"""Match live BLASST skip rates while replaying a fixed V→V QK gate."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import vqa_shards
from tile_calibration import write_json
from tv_guided_vv import SWEEP_SCHEMA
from vqav2_accuracy import SHORT_ANSWER_INSTRUCTION, score_answer

PHASES = ("prefill", "decode")
POLICIES = {
    "baseline": (False, False, False),
    "qk_skip": (True, False, False),
    "prefill_region": (True, True, False),
    "decode_region": (True, False, True),
    "both_regions": (True, True, True),
}
LABELS = {
    "baseline": "Original BLASST · 50% target",
    "qk_skip": "QK skip only",
    "prefill_region": "QK skip + prefill region max",
    "decode_region": "QK skip + decode region max",
    "both_regions": "QK skip + both region maxima",
}


def validate_args(args, parser):
    if not args.manifest or args.demo or args.calibration_count < 1 or args.decode_tokens < 2:
        parser.error("The matched study needs a VQA manifest, calibration cases and at least two generated tokens")
    if (args.tv_vv_sweep or args.vqa_shards is not None or args.vqa_shard_index is not None
            or args.lam or args.decode_lambda is not None or args.prefill_region_max or args.decode_region_max
            or args.vv_prefix_attention or args.jl_dim is not None or args.jl_sweep or args.jl_region_audit
            or args.tile_layer_study or args.score_distributions or args.score_data or args.raw_dir):
        parser.error("The matched study sets its own policies and lambdas; omit other sweep, mask and output modes")
    if not math.isfinite(args.budget_tolerance_pp) or args.budget_tolerance_pp < 0 or args.budget_probes < 2:
        parser.error("Use at least two budget probes and a finite nonnegative tolerance")


def rate(counts, phase):
    total = counts[f"{phase}_tiles"]
    if not total:
        raise ValueError(f"No {phase} tiles were generated; use more calibration cases or decode tokens")
    return counts[f"{phase}_tiles_skipped"] / total


def fit(probe, target, initial, max_probes, tolerance):
    """Bracket the target with live measurements and retain the closest measured setting."""
    trials = {}

    def measure(lam):
        if lam not in trials:
            trials[lam] = probe(lam)
        return trials[lam]

    def matched(value):
        return abs(value - target) <= tolerance

    start = measure(initial)
    if not matched(start):
        endpoint = 1. if start < target else 0.
        edge = measure(endpoint)
        if not matched(edge) and (start - target) * (edge - target) < 0:
            lo, hi = sorted((initial, endpoint))
            while len(trials) < max_probes:
                middle = math.sqrt(max(1e-12, lo) * hi)
                if not lo < middle < hi:
                    break
                actual = measure(middle)
                if matched(actual):
                    break
                if actual < target:
                    lo = middle
                else:
                    hi = middle
    lam, actual = min(trials.items(), key=lambda entry: (abs(entry[1] - target), entry[1]))
    return dict(lambda_=lam, target_skip_rate=target, measured_skip_rate=actual, within_tolerance=matched(actual),
                trials=[dict(lambda_=value, skip_rate=result) for value, result in trials.items()])


def gate_from(trace):
    return {layer["layer"]: [vote["tile"] for vote in layer["votes"] if vote["decision"] == "skip"]
            for layer in trace["layers"]}


def run(args, parser, samples, collect, select_sample, state, tile_size):
    calibration, evaluation = samples[:args.calibration_count], samples[args.calibration_count:]
    threshold = args.tv_vv_threshold if args.tv_vv_threshold is not None else .5
    tolerance = args.budget_tolerance_pp / 100
    settings = dict(model=args.model, manifest_sha256=args.budget_manifest_sha, tile_size=tile_size,
                    calibration_ids=[sample["id"] for sample in calibration],
                    gate_threshold=threshold, decode_tokens=args.decode_tokens,
                    image_max_pixels=args.image_max_pixels, max_prompt_tokens=args.max_tokens,
                    probes=args.budget_probes, tolerance_pp=args.budget_tolerance_pp)
    config_path = args.budget_config or args.output.with_suffix(".budget.json")
    if config_path.resolve() == args.output.resolve():
        raise ValueError("Use separate output and budget-config files")

    def forward(sample, policy, lambdas, gate=None, *, prefill_only=False):
        guided, prefill_reset, decode_reset = POLICIES[policy]
        options = argparse.Namespace(**dict(vars(args), lam=lambdas["prefill"],
                                            decode_lambda=lambdas["decode"],
                                            prefill_region_max=prefill_reset, decode_region_max=decode_reset,
                                            tv_vv_threshold=threshold if guided else None, gate_replay=gate,
                                            decode_tokens=1 if prefill_only else args.decode_tokens))
        select_sample(options, parser, sample)
        benchmark = sample["benchmark"]
        options.question = f"{benchmark['question_text']}\n{SHORT_ANSWER_INSTRUCTION}"
        before = {phase: state["phase_stats"].get(phase, {}).copy() for phase in PHASES}
        metadata = collect(options, parser, benchmark, sample["id"])
        trace = metadata["tv_guided_vv"]
        result = dict(answer=metadata["answer"], pure_vv_tiles=trace["pure_vv_tiles"],
                      vv_tiles_bypassed=trace["vv_tiles_bypassed"])
        for phase in PHASES:
            after = state["phase_stats"].get(phase, {})
            result[f"{phase}_tiles"] = after.get("causal", 0) - before[phase].get("causal", 0)
            result[f"{phase}_tiles_skipped"] = after.get("skipped", 0) - before[phase].get("skipped", 0)
        return result, gate_from(trace)

    def tune(policy, phase, target, initial, fixed, reference=None):
        def probe(lam):
            counts = defaultdict(int)
            trial = dict(fixed, **{phase: lam})
            for sample in calibration:
                ref, gate = reference[sample["id"]] if reference else (None, None)
                result, _ = forward(sample, policy, trial, gate, prefill_only=(phase == "prefill"))
                if ref:
                    check_gate(result, ref)
                for key in vqa_shards.TILE_COUNTS:
                    counts[key] += result[key]
            actual = rate(counts, phase)
            print(f"Calibrate {policy} {phase}: lambda={lam:.6g}, skip={actual:.2%}, target={target:.2%}", flush=True)
            return actual

        return fit(probe, target, initial, args.budget_probes, tolerance)

    with vqa_shards.claim(config_path):
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("schema") != "blasst-matched-region-budget-v1" or config.get("settings") != settings:
                raise ValueError(f"Calibration settings differ from {config_path}")
            print(f"Reusing matched-region calibration from {config_path}", flush=True)
        else:
            config = calibrate(calibration, forward, tune, tolerance, settings)
            write_json(config_path, config)
            print(f"Saved matched-region calibration to {config_path}", flush=True)

    eval_start = args.sample_index + len(calibration)
    report = dict(schema=SWEEP_SCHEMA, model=args.model, synthetic=False, tile_size=tile_size,
                  blasst_lambda=config["lambdas"]["baseline"]["prefill"],
                  decode_lambda=config["lambdas"]["baseline"]["decode"], pairs_requested=len(evaluation),
                  answer_metric="VQA v2 consensus", variants={name: threshold if guided else None
                      for name, (guided, _, _) in POLICIES.items()},
                  variant_labels=LABELS, budget_calibration=config, cases=[],
                  answer_protocol=dict(instruction=SHORT_ANSWER_INSTRUCTION, max_new_tokens=args.decode_tokens,
                                       min_new_tokens=0))
    for index, phase in enumerate(PHASES, start=1):
        report[f"{phase}_max_policies"] = {name: "per_region" if mode[index] else "shared"
                                         for name, mode in POLICIES.items()}
    vqa_shards.resume(report, args.output, evaluation, eval_start)
    totals = defaultdict(lambda: defaultdict(int))
    for case in report["cases"]:
        vqa_shards.accumulate(totals, case)
    completed = len(report["cases"])
    for index, sample in enumerate(evaluation[completed:], start=eval_start + completed):
        benchmark = sample["benchmark"]
        case = dict(sample_index=index, sample_id=sample["id"], question_id=benchmark["question_id"], results={})
        gate = None
        for policy in POLICIES:
            result, observed_gate = forward(sample, policy, config["lambdas"][policy], gate)
            if policy == "qk_skip":
                gate = observed_gate
            elif gate is not None:
                check_gate(result, case["results"]["qk_skip"])
            result["vqa_score"] = score_answer(result["answer"], benchmark["reference_answers"])
            if policy != "baseline":
                result["answer_changed"] = result["answer"] != case["results"]["baseline"]["answer"]
            case["results"][policy] = result
        report["cases"].append(case)
        vqa_shards.accumulate(totals, case)
        done = len(report["cases"])
        if done % vqa_shards.CHECKPOINT_INTERVAL == 0 or done == len(evaluation):
            report["summary"] = summarize(totals)
            write_json(args.output, report)
            print(f"Evaluated {done}/{len(evaluation)} held-out questions across five policies", flush=True)
    predictions = {name: [{"question_id": case["question_id"], "answer": case["results"][name]["answer"]}
                          for case in report["cases"]] for name in POLICIES}
    vqa_shards.save_predictions(args.output, predictions)
    for name, result in summarize(totals).items():
        skips = " · ".join(f"{phase} skip {value:.2%}" for phase in PHASES
                           if (value := result[f"{phase}_skip_rate"]) is not None)
        print(f"{name}: VQA {result['vqa_accuracy']:.2f}% · {skips}")


def check_gate(result, reference):
    if (result["pure_vv_tiles"], result["vv_tiles_bypassed"]) != (
        reference["pure_vv_tiles"], reference["vv_tiles_bypassed"]
    ):
        raise ValueError("The replayed V→V QK budget differs from policy 2")


def calibrate(samples, forward, tune, tolerance, settings):
    lambdas = {name: {} for name in POLICIES}
    calibration = {name: {} for name in POLICIES}

    def set_threshold(policy, phase, result):
        calibration[policy][phase] = result
        lambdas[policy][phase] = result["lambda_"]

    set_threshold("baseline", "prefill", tune("baseline", "prefill", .5, .1, {"decode": .1}))
    set_threshold("baseline", "decode", tune("baseline", "decode", .5, .1, lambdas["baseline"]))
    lambdas["qk_skip"] = lambdas["baseline"].copy()
    reference, counts = {}, defaultdict(int)
    for sample in samples:
        result, gate = forward(sample, "qk_skip", lambdas["qk_skip"])
        reference[sample["id"]] = (result, gate)
        for key in vqa_shards.TILE_COUNTS:
            counts[key] += result[key]
    targets = {phase: rate(counts, phase) for phase in PHASES}
    for phase in PHASES:
        calibration["qk_skip"][phase] = dict(lambda_=lambdas["qk_skip"][phase], measured_skip_rate=targets[phase])
    set_threshold("prefill_region", "prefill", tune("prefill_region", "prefill", targets["prefill"],
                                                    lambdas["qk_skip"]["prefill"], lambdas["qk_skip"], reference))
    set_threshold("decode_region", "prefill", dict(calibration["qk_skip"]["prefill"], copied_from="qk_skip"))
    set_threshold("both_regions", "prefill", dict(calibration["prefill_region"]["prefill"], copied_from="prefill_region"))
    for policy in ("prefill_region", "decode_region", "both_regions"):
        set_threshold(policy, "decode", tune(policy, "decode", targets["decode"], lambdas["qk_skip"]["decode"],
                                              lambdas[policy], reference))
    return dict(schema="blasst-matched-region-budget-v1", settings=settings, lambdas=lambdas,
                calibration=calibration, qk_skip_targets=targets)


def summarize(totals):
    summary = vqa_shards.summarize(totals)
    reference = summary["qk_skip"]
    for result in summary.values():
        result["vqa_delta_vs_qk_pp"] = result["vqa_accuracy"] - reference["vqa_accuracy"]
        for phase in PHASES:
            current = rate(result, phase) if result[f"{phase}_tiles"] else None
            reference_rate = rate(reference, phase) if reference[f"{phase}_tiles"] else None
            result[f"{phase}_skip_rate"] = current
            result[f"{phase}_skip_delta_vs_qk_pp"] = (
                100 * (current - reference_rate) if current is not None and reference_rate is not None else None
            )
    return summary
