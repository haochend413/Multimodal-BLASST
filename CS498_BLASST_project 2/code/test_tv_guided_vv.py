"""Check previous-layer causality, reactivation, and the actual compact QK matmul."""
import math
from types import SimpleNamespace

import torch

import blasst_attn as ba
from tv_guided_vv import TVGuidedVV


def test_previous_text_votes_bypass_and_reactivate_visual_queries(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    visual = torch.tensor([False, True, True, True, True, True, True, False])
    policy = TVGuidedVV(visual, 2, .5)
    snapshots = {}
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "collect_stats", True)
    monkeypatch.setitem(ba.STATE, "stats", {})
    monkeypatch.setitem(ba.STATE, "phase_stats", {})

    def observer(layer, q_start, scores, **_):
        snapshots[layer, q_start] = scores.clone()

    monkeypatch.setitem(ba.STATE, "observer", observer)
    q = torch.zeros(1, 2, 8, 3)
    q[..., 0] = 1
    k = torch.zeros(1, 1, 8, 3)
    v = torch.randn_like(k)
    widths = []
    matmul = torch.matmul

    def spy(left, right):
        if left.shape[-1] == 3:
            widths.append(right.shape[-1])
        return matmul(left, right)

    monkeypatch.setattr(torch, "matmul", spy)
    for layer in range(3):
        k[..., 0] = torch.tensor([8., 0., 0., 0., 9., 9., 0., 0.])
        if layer == 1:
            k[..., 2:4, 0] = 9
            k[..., 4:6, 0] = 0
        out, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, v, None, scaling=1.)
        assert torch.isfinite(out).all()
    assert widths == [8, 8, 8, 8, 8, 2, 8, 8, 8, 8, 2, 8]
    records = policy.layers
    assert [record["vote_layer"] for record in records] == [None, 0, 1]
    assert [record["vv_tiles_bypassed"] for record in records] == [0, 1, 2]
    assert ba.STATE["phase_stats"]["prefill"]["qk_bypassed"] == 6
    assert [v["decision"] for v in records[1]["votes"]] == ["skip", "keep"]
    assert [v["decision"] for v in records[2]["votes"]] == ["keep", "skip"]
    assert records[0]["votes"][1]["kept"] == records[0]["votes"][1]["eligible"] == 2
    assert snapshots[1, 2][..., 2:4].isneginf().all()
    assert snapshots[1, 2][..., :2].isfinite().all()
    assert snapshots[1, 6][..., 2:6].isfinite().all()
    distribution = policy.distribution()
    assert distribution["prompt_runs"] == [
        {"modality": "T", "tokens": 1}, {"modality": "V", "tokens": 6}, {"modality": "T", "tokens": 1}
    ]
    assert distribution["pairs_before"] == {"TT": 9, "TV": 18, "VT": 18, "VV": 63}
    assert distribution["pairs_after"] == {"TT": 9, "TV": 18, "VT": 18, "VV": 53}


def test_empty_vote_and_first_key_protection():
    policy = TVGuidedVV(torch.tensor([True, True, True, True, False, False]), 2, .5)
    layer = policy.start_layer(0, "cpu")
    layer.eligible[0] = layer.eligible[1] = 1
    layer.finish()
    next_layer = policy.start_layer(1, "cpu")
    assert next_layer.bypass_keys(0) == []
    assert next_layer.bypass_keys(1) == [1]
    assert next_layer.protected == {0, 1}

    policy = TVGuidedVV(torch.ones(4, dtype=torch.bool), 2, .5)
    policy.start_layer(0, "cpu").finish()
    assert policy.start_layer(1, "cpu").bypass_keys(1) == []


def test_counting_votes_leaves_the_baseline_unchanged(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.5))
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", None)
    torch.manual_seed(7)
    q, k, v = (torch.randn(1, 2, 8, 4) for _ in range(3))
    expected, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=1), q, k, v, None)
    visual = torch.tensor([False, False, True, True, True, True, False, False])
    policy = TVGuidedVV(visual, 2, None)
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, None)
    actual, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=1), q, k, v, None)
    torch.testing.assert_close(actual, expected)
    assert policy.layers[1]["vote_layer"] == 0
    assert policy.layers[1]["votes"][0]["eligible"] == 2
