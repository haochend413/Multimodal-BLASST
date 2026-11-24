"""Verify prefill masks, text-only selection and sparse decode against small fixtures."""
import math
from types import SimpleNamespace

import torch

import blasst_attn as ba
from decode_visual_links import DecodeVisualLinks


def test_prefill_links_drive_sparse_decode_and_head_local_fallback(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    module = SimpleNamespace(layer_idx=0)
    policy = DecodeVisualLinks(torch.tensor([False, False, True, True, True, True, False, False]), 2)
    q = torch.zeros(1, 2, 8, 2)
    q[..., 0] = 1
    k = torch.tensor([[2., 0.], [2., 0.], [6., 12.], [6., 12.], [0., 3.], [0., 3.], [2., 10.], [2., 10.]])
    k = k.view(1, 1, 8, 2).expand(1, 2, 8, 2).clone()
    # Head 1 only retains the early text positions, which never saw vision.
    k[0, 1, :2, 1] = 10
    k[0, 1, 6:, 1] = 0
    v = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2).expand_as(k)
    baseline, _ = ba.blasst_attention_forward(module, q, k, v, None, scaling=1.)
    monkeypatch.setitem(ba.STATE, "decode_visual_links", policy)
    prefill, _ = ba.blasst_attention_forward(module, q, k, v, None, scaling=1.)
    torch.testing.assert_close(prefill, baseline)
    links, history, *_ = policy.layers[0]
    assert links.dtype == torch.bool
    assert links[0, 6:].tolist() == [[True, False], [True, False]]
    assert not history[:, :2].any()
    assert not links[:, 2:6].any()

    decode_query = torch.tensor([0., 1.]).view(1, 1, 1, 2).expand(1, 2, 1, 2)
    k = torch.cat((k, torch.zeros(1, 2, 1, 2)), dim=-2)
    v = torch.cat((v, torch.zeros(1, 2, 1, 2)), dim=-2)
    out, _ = ba.blasst_attention_forward(module, decode_query, k, v, None, scaling=1.)
    expected_scores = k[0, :, :, 1].clone()
    # Early text is retained on head 0 by the online max rule.
    expected_scores[0, 4:6] = -torch.inf
    expected_scores[0, 8] = -torch.inf
    # Fallback on head 1 computes all visual QK, then BLASST skips the weaker tile.
    expected_scores[1, 4:] = -torch.inf
    expected = (expected_scores.softmax(-1).unsqueeze(1) @ v[0]).transpose(0, 1).unsqueeze(0)
    torch.testing.assert_close(out, expected)
    assert policy.counts["decode_link_fallback_heads"] == 1
    assert policy.counts["decode_visual_tiles"] == 4
    assert policy.counts["decode_visual_tiles_bypassed"] == 1
    assert policy.counts["decode_visual_keys_bypassed"] == 2


def test_mixed_key_tile_requires_visual_score_to_pass(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    policy = DecodeVisualLinks(torch.tensor([False, True, False, False]), 2)
    policy.start_layer(0, 1, "cpu")
    scores = torch.tensor([[[[10., 0., 0., 0.]]]])
    tiles = ba.score_tiles(scores, torch.ones(4, dtype=torch.bool))
    policy.record(0, 2, tiles, math.log(.1), None)
    links, history, *_ = policy.layers[0]
    assert history[0, 2].item()
    assert not links[0, 2, 0].item()
