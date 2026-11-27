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


def test_top_percent_ranks_previous_votes_with_a_fixed_quota():
    policy = TVGuidedVV(torch.tensor([False, True, True, True, True, True, True, False]),
                        1, None, top_percent=50)
    first = policy.start_layer(0, "cpu")
    assert first.dropped == set()
    first.eligible[1:7] = torch.tensor([4, 10, 4, 4, 4, 0])
    first.kept[1:7] = torch.tensor([3, 6, 3, 4, 1, 0])
    first.finish()
    second = policy.start_layer(1, "cpu")
    assert second.dropped == {2, 5, 6}
    assert second.bypass_keys(2) == [1, 2]
    assert second.bypass_keys(3) == []


def test_top_percent_selected_rows_still_use_blasst(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    policy = TVGuidedVV(torch.tensor([False, False, True, True]), 2, None, top_percent=100)
    policy.start_layer(0, "cpu").finish()
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
    observed = []
    monkeypatch.setitem(ba.STATE, "observer", lambda skip, **_: observed.append(skip))
    query = torch.ones(1, 1, 4, 1)
    key = torch.tensor([10., 10., 0., 0.]).view_as(query)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=1), query, key, query, None, scaling=1.)
    assert observed[1][0, 0, 1].item()
    assert policy.layers[1]["votes"][0]["decision"] == "keep"
    assert policy.layers[1]["vv_tiles_bypassed"] == 0


def test_top_token_percent_gates_individual_rows_even_in_mixed_tiles(monkeypatch):
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    visual = torch.tensor([False, *([True] * 128), False])
    policy = TVGuidedVV(visual, 128, None, token_percent=2)
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
    q = torch.ones(1, 1, len(visual), 1)
    k = torch.zeros_like(q)
    k[..., 126:129, :] = 10
    chunks = {}

    def observe(layer, q_start, scores, **_):
        chunks[layer, q_start] = scores

    monkeypatch.setitem(ba.STATE, "observer", observe)
    for layer in range(2):
        out, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, q, None)
        assert out.isfinite().all()
    warmup, gated = policy.layers
    assert warmup["visual_query_tokens_selected"] == 128
    assert gated["visual_query_tokens_selected"] == 3
    assert gated["dropped_token_positions"] == list(range(1, 126))
    assert gated["vv_pairs_bypassed"] == sum(range(1, 126))
    assert chunks[1, 0][..., 1:126, 1:129].isneginf().all()
    assert chunks[1, 0][..., 1:126, 0].isfinite().all()
    assert chunks[1, 128][..., 0, 1:129].isfinite().all()
    assert chunks[1, 128][..., 1, 1:129].isfinite().all()


def test_whole_visual_query_replays_vv_mask_and_keeps_visual_keys_readable(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "collect_stats", True)
    monkeypatch.setitem(ba.STATE, "stats", {})
    monkeypatch.setitem(ba.STATE, "phase_stats", {})
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    visual = torch.tensor([False, True, True, False, True, True])
    q = torch.ones(1, 1, 6, 1)
    k = torch.tensor([1., 8., 0., 1., 0., 0.]).view_as(q)
    v = torch.tensor([10., 20., 30., 40., 50., 60.]).view_as(q)
    vv = TVGuidedVV(visual, 2, None, token_gate="vv")
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", vv)
    for layer in range(2):
        vv_out, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, v, None)

    replay = {layer["layer"]: layer["dropped_token_positions"] for layer in vv.layers}
    assert replay == {0: [], 1: [2, 4, 5]}
    before = ba.STATE["phase_stats"]["prefill"].get("qk_bypassed", 0)
    whole = TVGuidedVV(visual, 2, None, token_gate="whole", replay=replay)
    monkeypatch.setitem(ba.STATE, "tv_guided_vv", whole)
    recorded = {}
    monkeypatch.setitem(ba.STATE, "observer",
                        lambda layer, q_start, scores, **_: recorded.update({(layer, q_start): scores}))
    for layer in range(2):
        whole_out, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, v, None)
    assert whole.layers[1]["dropped_token_positions"] == replay[1]
    assert whole.layers[1]["vv_pairs_bypassed"] == vv.layers[1]["vv_pairs_bypassed"] == 9
    assert whole.layers[1]["vt_pairs_bypassed"] == 5
    assert vv.layers[1]["vt_pairs_bypassed"] == 0
    assert ba.STATE["phase_stats"]["prefill"]["qk_bypassed"] - before == 3
    assert vv_out[:, replay[1]].ne(0).all()
    assert whole_out[:, replay[1]].eq(0).all()
    assert recorded[1, 2][..., 0, :].isneginf().all()
    assert recorded[1, 2][..., 1, 2].isfinite().all()
    assert whole_out.isfinite().all()


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


def test_prefill_drop_sweep_matches_query_count_and_ranks_previous_tv(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    visual = torch.tensor([False, True, True, False, True, True, False])
    q = torch.ones(1, 1, len(visual), 1)
    k = torch.tensor([1., 8., 0., 1., 8., 0., 1.]).view_as(q)
    for percent, count in ((50, 2), (99.9, 3)):
        random_masks = []
        for random_drop, drop_highest in ((False, False), (False, True), (True, False), (True, True)):
            policy = TVGuidedVV(visual, 2, None, token_gate="vv", token_drop_percent=percent,
                               random_drop=random_drop, drop_seed="fixture", drop_highest=drop_highest)
            monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
            for layer in range(2):
                out, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, q, None, scaling=1.)
                assert out.isfinite().all()
            assert policy.layers[0]["dropped_token_positions"] == []
            assert policy.layers[0]["visual_query_tokens_gateable"] == 0
            gated = policy.layers[1]
            dropped = gated["dropped_token_positions"]
            assert len(dropped) == count
            assert visual[dropped].all()
            assert gated["visual_query_tokens_gateable"] == 4
            assert gated["vt_pairs_bypassed"] == 0
            assert gated["vv_pairs_bypassed"] == int(visual.cumsum(0)[dropped].sum())
            if random_drop:
                random_masks.append(dropped)
            else:
                assert ({1, 4} if drop_highest else {2, 5}) <= set(dropped)
        assert random_masks[0] == random_masks[1]


def test_prefill_tile_size_changes_both_axes_and_preserves_decode_tiles(monkeypatch):
    monkeypatch.setitem(ba.STATE, "stats", {})
    monkeypatch.setitem(ba.STATE, "collect_stats", True)
    monkeypatch.setitem(ba.STATE, "ln_lam", 0.)
    visual = torch.arange(257) < 96
    q = torch.zeros(1, 1, len(visual), 1)

    for size, expected_tiles in ((32, 45), (64, 15), (128, 6), (256, 3)):
        monkeypatch.setitem(ba.STATE, "phase_stats", {})
        policy = TVGuidedVV(visual, size, None, token_gate="vv", token_drop_percent=50)
        monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
        for layer in range(2):
            module = SimpleNamespace(layer_idx=layer)
            output, _ = ba.blasst_attention_forward(module, q, q, q, None)
            assert output.isfinite().all()
        assert ba.STATE["phase_stats"]["prefill"]["causal"] == 2 * expected_tiles
        assert policy.layers[0]["visual_query_tokens_selected"] == 96
        assert policy.layers[1]["visual_query_tokens_selected"] == 48

        ba.blasst_attention_forward(module, q[:, :, :1], q, q, None)
        assert ba.STATE["phase_stats"]["decode"]["causal"] == 3
