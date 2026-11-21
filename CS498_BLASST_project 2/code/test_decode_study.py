# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode correctness and token-region accounting, independent of model weights."""
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import blasst_attn as ba
from attention_study import AttentionRecorder


@pytest.fixture(autouse=True)
def reset_state():
    ba.set_threshold(None)
    ba.set_decode_threshold(inherit=True)
    ba.STATE["observer"] = None
    ba.reset_records(record=False)
    yield
    ba.STATE["observer"] = None
    ba.set_threshold(None)
    ba.set_decode_threshold(inherit=True)
    ba.reset_records(record=False)


@pytest.mark.parametrize("length", [129, 513, 700])
@pytest.mark.parametrize("masked", [False, True])
def test_decode_against_independent_reference(length: int, masked: bool) -> None:
    torch.manual_seed(12)
    q = torch.randn(1, 8, 1, 32)
    k = torch.randn(1, 2, length, 32)
    v = torch.randn_like(k)
    q[..., 0] += 5
    k[:, :, 0, 0] += 15
    mask = torch.ones(1, 1, 1, length, dtype=torch.bool)
    if masked:
        mask[..., 200:256] = False
    repeated_k, repeated_v = k.repeat_interleave(4, 1), v.repeat_interleave(4, 1)
    scores = (q @ repeated_k.transpose(-1, -2) / math.sqrt(32)).masked_fill(~mask, -torch.inf)
    # Explicit sequential tile oracle, without cummax or reshaping.
    running = torch.full((1, 8, 1), -torch.inf)
    expected_scores = scores.clone()
    skipped = 0
    for start in range(0, length, 128):
        tile = scores[..., start:start + 128]
        tile_max = tile.max(-1).values
        running = torch.maximum(running, tile_max)
        discard = tile_max - running < math.log(0.1)
        skipped += int(discard.sum())
        expected_scores[..., start:start + 128].masked_fill_(discard[..., None], -torch.inf)
    expected = (expected_scores.softmax(-1) @ repeated_v).transpose(1, 2)
    ba.set_threshold(0.1)
    ba.reset_records(record=True)
    actual, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, mask)
    torch.testing.assert_close(actual, expected)
    assert ba.STATE["phase_stats"]["decode"]["skipped"] == skipped
    assert skipped > 0
    assert ba.STATE["phase_stats"]["decode"]["causal"] == 8 * math.ceil(length / 128)
    ba.set_decode_threshold(enabled=False)
    dense, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, mask)
    reference = F.scaled_dot_product_attention(q, repeated_k, repeated_v, attn_mask=mask).transpose(1, 2)
    torch.testing.assert_close(dense, reference)


def test_separate_decode_threshold_and_calibration() -> None:
    q = torch.ones(1, 1, 1, 8)
    k = torch.zeros(1, 1, 257, 8)
    k[:, :, 0] = 10
    v = torch.randn_like(k)
    ba.set_threshold(0.1)
    ba.set_decode_threshold(None)
    dense, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, None)
    reference = F.scaled_dot_product_attention(q, k, v).transpose(1, 2)
    torch.testing.assert_close(dense, reference)
    ba.set_decode_threshold(a=0.1 * 257)
    calibrated, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, None)
    ba.set_decode_threshold(0.1)
    fixed, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, None)
    torch.testing.assert_close(calibrated, fixed)


def test_mixed_tiles_count_token_regions_exactly() -> None:
    length = 140
    visual = torch.zeros(length, dtype=torch.bool)
    visual[30:100] = True
    recorder = AttentionRecorder(visual, {0}, {0})
    ba.STATE["observer"] = recorder.observe
    q = torch.ones(1, 1, length, 8)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, q, q, None)
    view = next(iter(recorder.views.values()))
    counts = {r: s["count"] for r, s in view["regions"].items()}
    expected = {r: 0 for r in counts}
    for row in range(length):
        for col in range(row + 1):
            region = f"{'V' if visual[row] else 'T'}→{'V' if visual[col] else 'T'}"
            expected[region] += 1
    assert counts == expected
    assert sum(counts.values()) == length * (length + 1) // 2
    assert abs(sum(x["mass"] for x in view["regions"].values()) - length) < 1e-4
    assert all(sum(s["hist"]) + s["underflow"] + s["overflow"] == s["count"]
               for s in view["regions"].values())


def test_recording_prefill_and_growing_decode_cache() -> None:
    ba.reset_records(record=True)
    module = SimpleNamespace(layer_idx=0)
    q = torch.randn(1, 2, 128, 8)
    ba.blasst_attention_forward(module, q, q, q, None)
    for length in (129, 257):
        keys = torch.randn(1, 2, length, 8)
        ba.blasst_attention_forward(module, q[:, :, :1], keys, keys, None)
    assert ba.get_margins()[0].shape == (1, 2, 1, 1)
    assert [x.shape[-1] for x in ba.get_decode_margins()[0]] == [2, 3]
