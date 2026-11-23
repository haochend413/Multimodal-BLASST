import torch

import blasst_attn as ba
from jl_region_study import JLRegionStudy, Span, axis_tiles, key_segments, physical_reset_margins, region_margins


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


def test_reset_inside_a_physical_tile_waits_for_each_rows_first_valid_key():
    visual = torch.tensor([False, False, False, True, True, True, True, False])
    keys = axis_tiles(visual, 2, packed=False)
    scores = torch.tensor([[10., 9., 2., 3., 2., 1., 2., 4.],
                           [10., 9., 2., -torch.inf, 1., 0.5, -torch.inf, -torch.inf]])
    tile_max = scores.unflatten(-1, (-1, 2)).amax(-1)
    margins = physical_reset_margins(scores, tile_max, key_segments(keys, visual))
    torch.testing.assert_close(margins, torch.tensor([[0., 0., -1., 0.],
                                                     [0., -8., 0., -torch.inf]]))
    assert margins.shape == tile_max.shape
    assert torch.all(margins >= ba.row_tile_margins(tile_max))


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
    original, original_reset, packed, reset = (results[policy]["all"] for policy in
        ("original_shared", "original_reset", "packed_shared", "packed_reset"))
    assert original["tiles"] == 30
    assert original_reset["tiles"] == original["tiles"]
    assert packed["tiles"] == reset["tiles"] == 30
    assert original["valid_pairs"] == original_reset["valid_pairs"] == packed["valid_pairs"] == reset["valid_pairs"] == 88
    assert results["packed_shared"]["mixed"]["tiles"] == 0
    assert results["original_shared"]["mixed"]["tiles"] > 0
    for score in ("full", "projected"):
        assert original_reset[f"{score}_skips"] <= original[f"{score}_skips"]
        assert reset[f"{score}_skips"] <= packed[f"{score}_skips"]
        assert reset[f"{score}_skipped_dense_weight"] <= packed[f"{score}_skipped_dense_weight"]
    assert original_reset["false_skips_vs_shared_full"] <= original["false_skips"]
    assert original_reset["missed_skips_vs_shared_full"] >= original["missed_skips"]
