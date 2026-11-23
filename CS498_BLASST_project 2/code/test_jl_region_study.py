import torch

import blasst_attn as ba
from jl_region_study import JLRegionStudy, Span, axis_tiles, region_margins


def test_pack_restarts_tiles_only_at_modality_boundaries():
    visual = torch.tensor([False, False, False, True, True])
    assert axis_tiles(visual, 2, packed=True) == [Span(0, 2, 0), Span(2, 3, 0), Span(3, 5, 1)]
    assert axis_tiles(visual, 2, packed=False) == [Span(0, 2, 0), Span(2, 4, 0), Span(4, 5, 0)]


def test_reset_uses_first_valid_score_in_each_contiguous_run():
    visual = torch.tensor([False, False, True, True, False, False])
    keys = axis_tiles(visual, 1, packed=True)
    scores = torch.tensor([[-2., -4., -8., -12., -9., -20.],
                           [-3., -6., -torch.inf, -11., -4., -10.]])
    margins = region_margins(scores, keys)
    torch.testing.assert_close(margins, torch.tensor([[0., -2., 0., -4., 0., -11.],
                                                     [0., -3., -torch.inf, 0., 0., -6.]]))
    assert torch.all(margins >= ba.row_tile_margins(scores))


def test_same_dense_inputs_preserve_coverage_and_reset_only_removes_skips(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    rng = torch.Generator().manual_seed(11)
    visual = torch.tensor([False] * 3 + [True] * 4 + [False] * 2)
    q = torch.randn(1, 4, len(visual), 8, generator=rng)
    k = torch.randn(1, 2, len(visual), 8, generator=rng)
    causal_mask = torch.ones(1, 4, len(visual), len(visual), dtype=torch.bool)
    causal_mask[..., 5, 1] = False
    study = JLRegionStudy(2, 0.3, 0, visual, {0}, {0, 2})
    study.observe(layer=0, phase="prefill", query=q, key=k, value=k,
                  attention_mask=causal_mask, scaling=8**-.5)
    results = study.summary({"model": "test", "synthetic": True})["phases"]["prefill"]
    original, packed, reset = (results[policy]["all"] for policy in
                               ("original_shared", "packed_shared", "packed_reset"))
    assert original["tiles"] == 30
    assert packed["tiles"] == reset["tiles"] == 30
    assert original["valid_pairs"] == packed["valid_pairs"] == reset["valid_pairs"] == 88
    assert results["packed_shared"]["mixed"]["tiles"] == 0
    assert results["original_shared"]["mixed"]["tiles"] > 0
    for score in ("full", "projected"):
        assert reset[f"{score}_skips"] <= packed[f"{score}_skips"]
        assert reset[f"{score}_skipped_dense_weight"] <= packed[f"{score}_skipped_dense_weight"]
