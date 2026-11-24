"""Checks for strict BLASST thresholds and shared calibration cutoffs."""
import math

from tile_calibration import best_threshold, calibrate
from tile_layer_study import LayerReport


def test_tied_and_zero_margins_limit_the_attainable_budget():
    margins = [-4.0, -2.0, -2.0, 0.0, 0.0]
    lowest = best_threshold(margins, 20)
    assert lowest["lambda"] == 1
    assert lowest["keep_rate"] == lowest["minimum_keep_rate"] == 0.4
    assert lowest["target_below_minimum"]
    eighty = best_threshold(margins, 80)
    assert eighty["kept"] == sum(value >= math.log(eighty["lambda"]) for value in margins) == 4


def test_phase_calibration_is_reused_for_every_evaluation_layer():
    def case(index, prefill, decode):
        return dict(sample_id=str(index), sample_index=index, layers=list(range(len(prefill))),
                    tiles=[["prefill", -1, 0, 0, 0, 2, 2, "T->T", prefill],
                           ["decode", 0, 0, 2, 0, 1, 2, "T->T", decode]])

    metadata = dict(model="demo", synthetic=True, tile_size=2)
    fitted = calibrate(metadata, [case(0, [-1, -.5, 0], [-10, -4, 0])], "phase")
    thresholds = fitted["targets"]["50"]
    calibration = dict(metadata=metadata, thresholds=thresholds, calibration_case_ids=["0"])
    report = LayerReport("demo", True, 0.1, 2, calibration=calibration)
    report.add(case(1, [-5, -5], [-1, -1]))
    rates = report.cases[0]["keep_rates"]
    assert rates["prefill"]["all"]["keep_rate"] == 0
    assert rates["decode"]["all"]["keep_rate"] == 1
    assert report.totals["prefill"]["T->T"][0, 1]["skip_skip"] == 1
    assert report.totals["decode"]["T->T"][0, 1]["keep_keep"] == 1
    assert not report.cases[0]["used_for_calibration"]
