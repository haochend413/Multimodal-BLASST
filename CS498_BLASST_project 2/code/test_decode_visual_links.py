"""Verify prefill masks, text-only selection and sparse decode against small fixtures."""
import math
from types import SimpleNamespace

import pytest
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


@pytest.mark.parametrize("keep_previous", [True, False])
def test_generated_link_is_reused_only_for_a_kept_text_key(monkeypatch, keep_previous):
    monkeypatch.setattr(ba, "TILE", 2)
    visual = torch.tensor([False, False, True, True, True, True, False, False])
    old = DecodeVisualLinks(visual, 2)
    growing = DecodeVisualLinks(visual, 2, extend_history=True)
    q = torch.ones(1, 3, 1, 1)
    k = torch.tensor([10., 10., 0., 0., 12., 12., 0., 0., 10., 10.]).view(1, 1, 10, 1)
    k = k.expand(1, 3, 10, 1)
    v = torch.arange(10.).view(1, 1, 10, 1).expand_as(k)
    cutoff = math.log(.1)
    for policy in (old, growing):
        policy.start_layer(0, 3, "cpu")
        # Only late prompt text has history, and T→T drops its low-scoring tile.
        links, history, *_ = policy.layers[0]
        links[:, 6:8, 0] = True
        history[:, 6:8] = True
    fixed_first, _ = old.decode(0, q, k[..., :9, :], v[..., :9, :], None, 1., cutoff, ba.score_tiles)
    growing_first, _ = growing.decode(0, q, k[..., :9, :], v[..., :9, :], None, 1., cutoff, ba.score_tiles)
    torch.testing.assert_close(fixed_first, growing_first, atol=0, rtol=0)
    links, history, *_ = growing.layers[0]
    assert links[:, 8].tolist() == [[False, True]] * 3
    assert history[:, 8].all()
    assert old.layers[0][0].shape[1] == 8

    if not keep_previous:
        k = k.clone()
        k[:, :, 8:, :] = 0  # The generated text tile now loses the T→T BLASST test.
    mask = None
    fixed_second, _ = old.decode(0, q, k, v, mask, 1., cutoff, ba.score_tiles)
    growing_second, _ = growing.decode(0, q, k, v, mask, 1., cutoff, ba.score_tiles)
    torch.testing.assert_close(fixed_second, growing_second, atol=0, rtol=0)
    assert old.counts["decode_visual_tiles_bypassed"] == 0
    assert growing.counts["decode_generated_history_head_steps"] == (3 if keep_previous else 0)
    assert growing.counts["decode_visual_tiles_bypassed"] == (3 if keep_previous else 0)
    assert growing.counts["decode_link_fallback_heads"] == (3 if keep_previous else 6)


def test_fixed_budget_matches_visible_key_counts_and_ranks_link_votes():
    visual = torch.ones(5, dtype=torch.bool)
    columns = torch.tensor([0, 0, 1, 1, 2])
    allowed = torch.tensor([[True, True, True, True, False],
                            [True, False, True, False, True],
                            [False, False, False, False, False]])
    links = torch.tensor([[[True, False, False], [True, True, False]]] * 3)
    usable = torch.tensor([[True, True], [False, False], [True, True]])
    guided = DecodeVisualLinks(visual, 2, drop_percent=50, drop_seed="12:0")
    random = DecodeVisualLinks(visual, 2, drop_percent=50, random_drop=True, drop_seed="12:0")

    def budget(policy):
        return policy.budgeted_visual_keys(0, 1, allowed, columns, links, usable)

    guided_keys, random_keys = budget(guided), budget(random)
    for keys in (guided_keys, random_keys):
        assert keys.sum(dim=1).tolist() == [2, 2, 0]
        assert not (keys & ~allowed).any()
    assert guided_keys[0].tolist() == [True, True, False, False, False]
    assert guided_keys[1].equal(random_keys[1])
    assert budget(random).equal(random_keys)


@pytest.mark.parametrize("rate, expected_bypassed", [(5, 0), (99.9, 4)])
def test_fixed_decode_sweep_preserves_prefill_and_counts_actual_bypass(monkeypatch, rate, expected_bypassed):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "decode_enabled", True)
    module = SimpleNamespace(layer_idx=0)
    q = torch.ones(1, 1, 7, 1)
    k = torch.tensor([8., 0., 1., 5., 6., 3., 4., 0.]).view(1, 1, 8, 1)
    v = torch.arange(8.).view(1, 1, 8, 1)
    baseline, _ = ba.blasst_attention_forward(module, q, k[..., :7, :], v[..., :7, :], None, scaling=1.)
    baseline_decode, _ = ba.blasst_attention_forward(module, q[..., :1, :], k, v, None, scaling=1.)
    for random_drop in (False, True):
        policy = DecodeVisualLinks(torch.tensor([False, True, True, True, True, True, False]),
                                   2, extend_history=True, drop_percent=rate, random_drop=random_drop)
        monkeypatch.setitem(ba.STATE, "decode_visual_links", policy)
        prefill, _ = ba.blasst_attention_forward(module, q, k[..., :7, :], v[..., :7, :], None, scaling=1.)
        torch.testing.assert_close(prefill, baseline)
        output, _ = ba.blasst_attention_forward(module, q[..., :1, :], k, v, None, scaling=1.)
        assert policy.counts["decode_visual_keys"] == 5
        assert policy.counts["decode_visual_keys_bypassed"] == expected_bypassed
        assert policy.counts["decode_link_fallback_heads"] == 0
        assert policy.layers[0][0].shape[1] == 8
        if expected_bypassed == 0:
            torch.testing.assert_close(output, baseline_decode)
