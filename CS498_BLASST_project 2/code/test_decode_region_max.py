"""Check live decode max resets across boundaries on the physical tile grid."""
from types import SimpleNamespace

import torch

import blasst_attn as ba
from region_max import DecodeRegionMax


def test_region_decode_uses_one_softmax_and_keeps_boundary_tiles(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 4)
    monkeypatch.setitem(ba.STATE, "decode_enabled", True)
    monkeypatch.setitem(ba.STATE, "decode_override", True)
    monkeypatch.setitem(ba.STATE, "decode_lam", .1)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    logit = torch.tensor([10., 9., 8., 0., -1., -2., -3., -4., -5., -8., -9., -9.,
                          -12., -13., -14., -15.])
    q = torch.ones(1, 1, 1, 1)
    k = logit.reshape(1, 1, 16, 1)
    v = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)
    captured = []
    monkeypatch.setitem(ba.STATE, "observer", lambda **data: captured.append(data))
    module = SimpleNamespace(layer_idx=0)
    monkeypatch.setitem(ba.STATE, "decode_region_max", None)
    ba.blasst_attention_forward(module, q, k, v, None)
    assert captured[-1]["skip"].flatten().tolist() == [False, True, True, True]

    prompt_modalities = torch.tensor([False] * 3 + [True] * 6 + [False])
    monkeypatch.setitem(ba.STATE, "decode_region_max", DecodeRegionMax(prompt_modalities, 4))
    output, _ = ba.blasst_attention_forward(module, q, k, v, None)
    assert captured[-1]["skip"].flatten().tolist() == [False, False, False, True]
    torch.testing.assert_close(captured[-1]["margin"].flatten(), torch.tensor([0., -1., 0., -4.]))
    expected = logit[:12].softmax(0) @ v.flatten()[:12]
    torch.testing.assert_close(output.squeeze(), expected)

    # The first text suffix key (position 9) is masked. The next visible key sets
    # the new text run's maximum.
    mask = torch.ones(1, 1, 1, 16, dtype=torch.bool)
    mask[..., 9] = False
    ba.blasst_attention_forward(module, q, k, v, mask)
    assert captured[-1]["skip"].flatten().tolist() == [False, False, False, True]
    torch.testing.assert_close(captured[-1]["margin"].flatten(), torch.tensor([0., -1., 0., -3.]))


def test_prefill_regions_change_previous_layer_votes_and_leave_decode_shared(monkeypatch):
    from math import log

    from region_max import RegionMax
    from tv_guided_vv import TVGuidedVV

    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", log(.1))
    monkeypatch.setitem(ba.STATE, "decode_enabled", True)
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "decode_region_max", None)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    visual = torch.tensor([False, False, True, True, True, True, True, False])
    q = torch.ones(1, 1, 8, 1)
    k = torch.tensor([8., 7., 0., 0., -1., -1., -5., 2.]).reshape(1, 1, 8, 1)
    captured = {}
    monkeypatch.setitem(ba.STATE, "observer", lambda **data: captured.update({
        (data["layer"], data["phase"], data["q_start"]): data["skip"].flatten().tolist()
    }))
    for reset in (False, True):
        policy = TVGuidedVV(visual, 2, .5)
        monkeypatch.setitem(ba.STATE, "tv_guided_vv", policy)
        monkeypatch.setitem(ba.STATE, "prefill_region_max", RegionMax(visual, 2) if reset else None)
        for layer in range(2):
            ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, k, None)
        assert [vote["kept"] for vote in policy.layers[0]["votes"]] == [int(reset)] * 2
        assert [vote["eligible"] for vote in policy.layers[0]["votes"]] == [1, 1]
        assert policy.layers[1]["vv_tiles_bypassed"] == (0 if reset else 3)
        assert captured[0, "prefill", 2][2]  # The next visual key tile is still causal-masked.
        ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q[..., :1, :], k, k, None)
        assert captured[0, "decode", 7] == [False, True, True, True]
