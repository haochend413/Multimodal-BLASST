import copy

import torch

from jl_gap_study import decision_counts, gap_histograms, merge_sweeps, tile_groups


def test_mixed_key_gaps_use_shared_running_max_and_keep_overflow():
    scores = torch.tensor([[[[10., 8.], [1., 2.]], [[3., 4.], [-100., 95.]]]])
    query = torch.tensor([False, True])
    keys = torch.tensor([[0, 1], [0, 1]])
    hist = gap_histograms(scores, query, keys)
    assert hist[0, [0, 9]].tolist() == [1, 1]
    assert hist[1, [2, 8]].tolist() == [1, 1]
    assert hist[2, [1, 80]].tolist() == [1, 1]
    assert hist[3, 0] == 2
    assert hist.sum() == 8


def test_fully_masked_tiles_and_modalities_are_excluded():
    scores = torch.tensor([[[[5., -torch.inf], [-torch.inf, -torch.inf]]]])
    query, keys = torch.tensor([False]), torch.tensor([[0, 1], [0, -1]])
    hist = gap_histograms(scores, query, keys)
    assert hist.sum() == hist[0, 0] == 1
    groups = tile_groups(scores, query, keys)
    skips = torch.tensor([[False, True]])
    projected = torch.tensor([[True, True]])
    result = decision_counts(groups, skips, projected)
    assert result[0].tolist() == result[1].tolist() == [1, 0, 1, 0, 1, 0]
    assert result[2:].sum() == 0


def test_batch_pools_counts_and_preserves_number_of_pairs():
    report = {key: 0 for key in (
        "model", "synthetic", "head_width", "dimensions", "lambda", "seed", "histogram", "layers", "heads",
    )}
    gaps = {kind: [1, 0] for kind in ("T->T", "T->V", "V->T", "V->V")}
    report.update(pairs_completed=1, phases={"prefill": {
        "full_gaps": copy.deepcopy(gaps),
        "projections": {"32": {"gaps": gaps, "decisions": {"all": {"false_skips": 2}}}},
    }})
    original = copy.deepcopy(report)
    result = merge_sweeps(merge_sweeps(None, report), copy.deepcopy(report))
    phase = result["phases"]["prefill"]
    assert result["pairs_completed"] == 2
    assert phase["full_gaps"]["T->V"] == phase["projections"]["32"]["gaps"]["T->V"] == [2, 0]
    assert phase["projections"]["32"]["decisions"]["all"]["false_skips"] == 4
    assert report == original
