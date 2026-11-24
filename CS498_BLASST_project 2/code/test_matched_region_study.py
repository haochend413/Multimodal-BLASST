"""Check live budget fitting, fixed gate replay, and split evaluation."""
import json
from argparse import ArgumentParser, Namespace

import pytest

from matched_region_study import fit, run
from tv_guided_vv import TVGuidedVV


def test_fit_reports_an_unreachable_target():
    result = fit(lambda lam: 1 - .3 * lam, .5, .7, 8, .01)
    assert result["lambda_"] == 1
    assert result["measured_keep_rate"] == pytest.approx(.7)
    assert not result["within_tolerance"]


def test_replayed_gate_uses_recorded_query_tiles_instead_of_new_votes():
    import torch

    policy = TVGuidedVV(torch.tensor([False, False, True, True, True, True, False, False]),
                        2, .5, replay={0: [], 1: [2]})
    first = policy.start_layer(0, "cpu")
    first.eligible[1:3] = 1
    first.kept[2] = 1
    first.finish()
    layer = policy.start_layer(1, "cpu")
    assert layer.dropped == {2}
    assert layer.bypass_keys(2) == [1, 2]
    assert layer.bypass_keys(1) == []


def test_four_policies_preserve_case_gates_and_reuse_calibration(tmp_path):
    phases = ("prefill", "decode")
    state = {"phase_stats": {phase: {"causal": 0, "skipped": 0} for phase in phases}}
    calls = []
    samples = [
        {"id": str(index), "benchmark": {"question_id": index, "question_text": "Color?",
                                        "reference_answers": ["red"] * 10}}
        for index in range(3)
    ]

    def collect(args, _parser, _benchmark, sample_id):
        guided = args.tv_vv_threshold is not None
        reset = args.prefill_region_max or args.decode_region_max
        calls.append((sample_id, guided, reset, args.gate_replay))
        dropped = [1] if guided else []
        if reset or args.gate_replay is not None:
            assert args.gate_replay == {0: [], 1: [1]}
        for phase in phases:
            if phase == "decode" and args.decode_tokens == 1:
                continue
            lam = args.lam if phase == "prefill" else args.decode_lambda
            phase_reset = args.prefill_region_max if phase == "prefill" else args.decode_region_max
            multiplier = (36 if phase == "prefill" else 45) if phase_reset else 90
            bypassed = 10 if guided and phase == "prefill" else 0
            count = round(multiplier * lam) + bypassed
            stats = state["phase_stats"][phase]
            stats["qk_bypassed"] = stats.get("qk_bypassed", 0) + bypassed
            state["phase_stats"][phase]["causal"] += 100
            state["phase_stats"][phase]["skipped"] += count
        layers = [{"layer": 0, "votes": []}, {"layer": 1, "votes": [
            {"tile": index, "decision": "skip"} for index in dropped
        ]}]
        return dict(answer="red", tv_guided_vv=dict(layers=layers, pure_vv_tiles=6, vv_tiles_bypassed=len(dropped)))

    output = tmp_path / "report.json"
    config = tmp_path / "budget.json"
    args = Namespace(calibration_count=1, tv_vv_threshold=.5, budget_tolerance_pp=2.,
                     budget_probes=14, budget_keep_percent=50., gate_lambda=1., output=output, budget_config=config, budget_manifest_sha="fixture",
                     model="test", image_max_pixels=1280, max_tokens=8192, decode_tokens=4, sample_index=0)
    run(args, ArgumentParser(), samples, collect, lambda *_: None, state, 128)
    report = json.loads(output.read_text())
    budget = json.loads(config.read_text())
    assert "baseline" not in report["variants"]
    assert len(report["variants"]) == 4
    assert budget["settings"]["requested_keep_rate"] == .5
    assert budget["common_keep_targets"] == pytest.approx({"prefill": .6, "decode": .55})
    assert budget["lambdas"]["decode_region"]["prefill"] == budget["lambdas"]["qk_skip"]["prefill"]
    assert budget["lambdas"]["both_regions"]["prefill"] == budget["lambdas"]["prefill_region"]["prefill"]
    assert [case["question_id"] for case in report["cases"]] == [1, 2]
    for result in report["summary"].values():
        assert result["vv_tiles_bypassed"] == 2
        assert result["prefill_tiles_after_qk"] == 180
        assert result["prefill_kept_after_qk"] == 200 - result["prefill_tiles_skipped"]
        assert result["decode_tiles_after_qk"] == 200
    assert all(budget["status"] == "within tolerance" for budget in report["evaluation_budget"].values())
    assert all(call[1] for call in calls)
    assert sum(call[3] is None for call in calls) == len(samples)
    assert len([call for call in calls if call[0] != "0"]) == 10
    calls.clear()
    run(args, ArgumentParser(), samples, collect, lambda *_: None, state, 128)
    assert not calls
