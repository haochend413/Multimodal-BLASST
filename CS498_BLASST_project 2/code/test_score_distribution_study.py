"""Regression check for score and row counts in the causal threshold audit."""
import math

import numpy as np

from score_distribution_study import EXCESS_EDGES, ScoreDistributions


def test_per_row_cutoff_and_masked_lanes():
    study = ScoreDistributions(math.exp(-1))
    scores = np.array([[6, 3, 5.5, -np.inf], [3, 0, 1, 1]], dtype=np.float32)
    study.chunk("prefill", scores, np.array([False, False]),
                np.array([False, False, True, True]), 2, "case")
    physical = study.report()["sources"]["physical_tiles"]["prefill"]
    tv = physical["T->V"]
    cutoff_bin = EXCESS_EDGES.index(0) + 1
    assert sum(tv["score_excess"]) == 3
    assert sum(tv["score_excess"][cutoff_bin:]) == 1
    assert sum(tv["row_excess"][cutoff_bin:]) == 1
    assert sum(tv["tile_excess"][cutoff_bin:]) == 1
    assert tv["row_support"][4] == 1  # One of two rows reaches the cutoff.
    assert tv["score_support"][3] == 1  # One of three valid scores reaches it.
