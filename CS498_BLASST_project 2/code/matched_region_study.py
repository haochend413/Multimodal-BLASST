"""Match live BLASST keep rates after QK skip while replaying a fixed V→V QK gate."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict

import vqa_shards
from tile_calibration import write_json
from tv_guided_vv import SWEEP_SCHEMA
from vqav2_accuracy import SHORT_ANSWER_INSTRUCTION, score_answer

BUDGET_SCHEMA = "blasst-matched-region-budget-v2"
REFERENCE = "qk_skip"
PHASES = ("prefill", "decode")
POLICIES = {
    "qk_skip": (False, False),
    "prefill_region": (True, False),
    "decode_region": (False, True),
    "both_regions": (True, True),
}
LABELS = {
    "qk_skip": "QK skip only",
    "prefill_region": "QK skip + prefill region max",
    "decode_region": "QK skip + decode region max",
    "both_regions": "QK skip + both region maxima",
}


def validate_args(args, parser):
    if args.gate_distribution and (args.matched_region_study or args.budget_config is not None):
        parser.error("Use --gate-distribution separately from matched-study calibration")
    if not args.manifest or args.demo or (not args.gate_distribution and
                                        (args.calibration_count < 1 or args.decode_tokens < 2)):
        parser.error("The matched study needs a VQA manifest, calibration cases and at least two generated tokens")
    if (args.tv_vv_sweep or args.vqa_shards is not None or args.vqa_shard_index is not None
            or args.lam or args.decode_lambda is not None or args.prefill_region_max or args.decode_region_max
            or args.vv_prefix_attention or args.jl_dim is not None or args.jl_sweep or args.jl_region_audit
            or args.tile_layer_study or args.score_distributions or args.score_data or args.raw_dir):
        parser.error("The matched study sets its own policies and lambdas; omit other sweep, mask and output modes")
    if not math.isfinite(args.budget_tolerance_pp) or args.budget_tolerance_pp < 0 or args.budget_probes < 2:
        parser.error("Use at least two budget probes and a finite nonnegative tolerance")
    if not 0 < args.budget_keep_percent <= 100 or not 0 < args.gate_lambda <= 1:
        parser.error("Use a keep percentage in (0, 100] and a gate lambda in (0, 1]")


def post_qk_counts(counts, phase):
    eligible = counts[f"{phase}_tiles"]
    bypassed = counts.get(f"{phase}_qk_bypassed", 0)
    return eligible - counts[f"{phase}_tiles_skipped"], eligible - bypassed


def fit(probe, target, minimum, max_probes, tolerance):
    """Search the live keep rate, including both endpoints in the measured trials."""
    trials = {1.: minimum}
    if abs(minimum - target) > tolerance:
        trials[0.] = probe(0.)
    if len(trials) == 2 and minimum < target < trials[0.]:
        lo, hi = 0., 1.
        while len(trials) < max_probes and min(abs(rate - target) for rate in trials.values()) > tolerance:
            middle = math.sqrt(lo * hi) if lo else hi * .1
            if not lo < middle < hi:
                break
            actual = trials[middle] = probe(middle)
            if actual > target:
                lo = middle
            else:
                hi = middle
    lam, actual = min(trials.items(), key=lambda entry: (abs(entry[1] - target), -entry[1]))
    return dict(lambda_=lam, target_keep_rate=target, measured_keep_rate=actual,
                within_tolerance=abs(actual - target) <= tolerance,
                trials=[dict(lambda_=value, keep_rate=result) for value, result in trials.items()])


def gate_from(trace):
    return {layer["layer"]: [vote["tile"] for vote in layer["votes"] if vote["decision"] == "skip"]
            for layer in trace["layers"]}


def run(args, parser, samples, collect, select_sample, state, tile_size):
    if getattr(args, "gate_distribution", False):
        return run_gate_distribution(args, parser, samples, collect, select_sample, tile_size)
    calibration, evaluation = samples[:args.calibration_count], samples[args.calibration_count:]
    top_percent = getattr(args, "tv_vv_top_percent", None)
    threshold = args.tv_vv_threshold if top_percent is not None or args.tv_vv_threshold is not None else .5
    settings = dict(model=args.model, manifest_sha256=args.budget_manifest_sha, tile_size=tile_size,
                    calibration_ids=[sample["id"] for sample in calibration],
                    gate_threshold=threshold, gate_lambda=args.gate_lambda,
                    requested_keep_rate=args.budget_keep_percent / 100, decode_tokens=args.decode_tokens,
                    image_max_pixels=args.image_max_pixels, max_prompt_tokens=args.max_tokens,
                    probes=args.budget_probes, tolerance_pp=args.budget_tolerance_pp)
    if top_percent is not None:
        settings["gate_top_percent"] = top_percent
    config_path = args.budget_config or args.output.with_suffix(".budget.json")
    if config_path.resolve() == args.output.resolve():
        raise ValueError("Use separate output and budget-config files")

    def forward(sample, policy, lambdas, gate=None, *, prefill_only=False):
        prefill_reset, decode_reset = POLICIES[policy]
        options = argparse.Namespace(**dict(vars(args), lam=lambdas["prefill"],
                                            decode_lambda=lambdas["decode"],
                                            prefill_region_max=prefill_reset, decode_region_max=decode_reset,
                                            tv_vv_threshold=threshold, gate_replay=gate,
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
        result["prefill_qk_bypassed"] = (state["phase_stats"].get("prefill", {}).get("qk_bypassed", 0)
                                         - before["prefill"].get("qk_bypassed", 0))
        return result, gate_from(trace)

    def capture_gate(sample):
        return forward(sample, REFERENCE, {"prefill": args.gate_lambda, "decode": 0.}, prefill_only=True)

    def probe(policy, phase, fixed, reference, lam):
        kept = total = 0
        trial = dict(fixed, **{phase: lam})
        for sample, (ref, gate) in zip(calibration, reference):
            result, observed_gate = forward(sample, policy, trial, gate, prefill_only=(phase == "prefill"))
            check_gate(result, ref, gate, observed_gate)
            case_kept, case_total = post_qk_counts(result, phase)
            kept += case_kept
            total += case_total
        if not total:
            raise ValueError(f"No {phase} tiles survived QK skip in calibration")
        actual = kept / total
        print(f"Calibrate {policy} {phase}: lambda={lam:.6g}, kept after QK={kept}/{total} ({actual:.2%})", flush=True)
        return actual

    with vqa_shards.claim(config_path):
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get("schema") != BUDGET_SCHEMA or config.get("settings") != settings:
                raise ValueError(f"Calibration settings or schema differ from {config_path}. Use a fresh --budget-config path.")
            print(f"Reusing matched-region calibration from {config_path}", flush=True)
        else:
            config = calibrate(probe, [capture_gate(sample) for sample in calibration], settings)
            write_json(config_path, config)
            print(f"Saved matched-region calibration to {config_path}", flush=True)

    eval_start = args.sample_index + len(calibration)
    report = dict(schema=SWEEP_SCHEMA, model=args.model, synthetic=False, tile_size=tile_size,
                  blasst_lambda=config["lambdas"][REFERENCE]["prefill"],
                  decode_lambda=config["lambdas"][REFERENCE]["decode"], pairs_requested=len(evaluation),
                  answer_metric="VQA v2 consensus",
                  variants=dict.fromkeys(POLICIES, threshold if top_percent is None else top_percent),
                  variant_labels=LABELS, budget_calibration=config, cases=[],
                  answer_protocol=dict(instruction=SHORT_ANSWER_INSTRUCTION, max_new_tokens=args.decode_tokens,
                                       min_new_tokens=0))
    for index, phase in enumerate(PHASES):
        report[f"{phase}_max_policies"] = {name: "per_region" if mode[index] else "shared"
                                         for name, mode in POLICIES.items()}
    vqa_shards.resume(report, args.output, evaluation, eval_start)
    totals = defaultdict(lambda: defaultdict(int))
    for case in report["cases"]:
        vqa_shards.accumulate(totals, case, reference=REFERENCE)
    completed = len(report["cases"])
    for index, sample in enumerate(evaluation[completed:], start=eval_start + completed):
        benchmark = sample["benchmark"]
        case = dict(sample_index=index, sample_id=sample["id"], question_id=benchmark["question_id"], results={})
        ref, gate = capture_gate(sample)
        for policy in POLICIES:
            result, observed_gate = forward(sample, policy, config["lambdas"][policy], gate)
            check_gate(result, ref, gate, observed_gate)
            result["vqa_score"] = score_answer(result["answer"], benchmark["reference_answers"])
            if policy != REFERENCE:
                result["answer_changed"] = result["answer"] != case["results"][REFERENCE]["answer"]
            case["results"][policy] = result
        report["cases"].append(case)
        vqa_shards.accumulate(totals, case, reference=REFERENCE)
        done = len(report["cases"])
        if done % vqa_shards.CHECKPOINT_INTERVAL == 0 or done == len(evaluation):
            report["summary"] = summarize(totals)
            report["evaluation_budget"] = evaluation_budget(report["summary"], config)
            write_json(args.output, report)
            print(f"Evaluated {done}/{len(evaluation)} held-out questions across four policies", flush=True)
    predictions = {name: [{"question_id": case["question_id"], "answer": case["results"][name]["answer"]}
                          for case in report["cases"]] for name in POLICIES}
    vqa_shards.save_predictions(args.output, predictions)
    summary = summarize(totals)
    for name, result in summary.items():
        keeps = " · ".join(
            f"{phase} kept after QK {result[f'{phase}_kept_after_qk']}/{result[f'{phase}_tiles_after_qk']} "
            f"({value:.2%})" for phase in PHASES if (value := result[f"{phase}_keep_rate_after_qk"]) is not None
        )
        print(f"{name}: VQA {result['vqa_accuracy']:.2f}% · {keeps}")
    for phase, budget in evaluation_budget(summary, config).items():
        spread = budget["spread_pp"]
        detail = f" · between-policy spread {spread:.2f} pp" if spread is not None else ""
        print(f"Held-out {phase}: target {budget['target_keep_rate']:.2%}{detail} · {budget['status']}")


def check_gate(result, reference, expected_gate, observed_gate):
    keys = ("pure_vv_tiles", "vv_tiles_bypassed", "prefill_qk_bypassed")
    if observed_gate != expected_gate or any(result[key] != reference[key] for key in keys):
        raise ValueError("The replayed V→V QK gate differs from the recorded gate")


def run_gate_distribution(args, parser, samples, collect, select_sample, tile_size):
    top_percent = args.tv_vv_top_percent
    threshold = args.tv_vv_threshold if top_percent is not None or args.tv_vv_threshold is not None else .5
    cases = []
    totals = {field: defaultdict(int) for field in ("prompt_tokens", "pairs_before", "pairs_after")}
    for sample in samples:
        options = argparse.Namespace(**dict(vars(args), lam=args.gate_lambda, decode_lambda=0.,
                                            tv_vv_threshold=threshold, decode_tokens=1))
        select_sample(options, parser, sample)
        benchmark = sample["benchmark"]
        options.question = f"{benchmark['question_text']}\n{SHORT_ANSWER_INSTRUCTION}"
        result = collect(options, parser, benchmark, sample["id"])["gate_distribution"]
        cases.append(dict(sample_id=sample["id"], question_id=benchmark["question_id"], **result))
        for field, counts in totals.items():
            for key, count in result[field].items():
                counts[key] += count
        if len(cases) % vqa_shards.CHECKPOINT_INTERVAL == 0 or len(cases) == len(samples):
            print(f"Measured V→V QK distribution on {len(cases)}/{len(samples)} questions", flush=True)
    write_json(args.output, dict(schema="blasst-gate-distribution-v1", model=args.model, tile_size=tile_size,
                                 gate_lambda=args.gate_lambda, gate_threshold=threshold,
                                 gate_top_percent=top_percent, totals=totals, cases=cases))
    print(f"Wrote V→V QK distribution to {args.output}", flush=True)


def calibrate(probe, reference, settings):
    lambdas = {name: dict.fromkeys(PHASES, 0.) for name in POLICIES}
    calibration = {name: {} for name in POLICIES}
    targets, minimums = {}, {}
    requested = settings["requested_keep_rate"]
    tolerance = settings["tolerance_pp"] / 100
    prefill_sources = {"decode_region": REFERENCE, "both_regions": "prefill_region"}
    for phase in PHASES:
        sources = prefill_sources if phase == "prefill" else {}
        policies = [name for name in POLICIES if name not in sources]
        endpoints = {name: probe(name, phase, lambdas[name], reference, 1.) for name in policies}
        minimums[phase] = dict(endpoints, **{name: endpoints[source] for name, source in sources.items()})
        target = targets[phase] = max(requested, *endpoints.values())
        print(f"Common {phase} keep target after QK: {target:.2%} (requested {requested:.2%})", flush=True)
        for name in policies:
            result = fit(lambda lam: probe(name, phase, lambdas[name], reference, lam),
                         target, endpoints[name], settings["probes"], tolerance)
            calibration[name][phase] = result
            lambdas[name][phase] = result["lambda_"]
            if not result["within_tolerance"]:
                print(f"{name} {phase}: closest measured keep rate "
                      f"{result['measured_keep_rate']:.2%} is outside tolerance", flush=True)
        for name, source in sources.items():
            calibration[name][phase] = dict(calibration[source][phase], copied_from=source)
            lambdas[name][phase] = lambdas[source][phase]
    return dict(schema=BUDGET_SCHEMA, settings=settings, lambdas=lambdas,
                calibration=calibration, common_keep_targets=targets, keep_rates_at_lambda_one=minimums)


def summarize(totals):
    summary = vqa_shards.summarize(totals, reference=REFERENCE)
    reference = summary[REFERENCE]
    for result in summary.values():
        result["vqa_delta_vs_qk_pp"] = result["vqa_accuracy"] - reference["vqa_accuracy"]
        for phase in PHASES:
            kept, total = post_qk_counts(result, phase)
            result[f"{phase}_kept_after_qk"] = kept
            result[f"{phase}_tiles_after_qk"] = total
            result[f"{phase}_keep_rate_after_qk"] = kept / total if total else None
    return summary


def evaluation_budget(summary, config):
    budgets = {}
    tolerance = config["settings"]["tolerance_pp"] / 100
    for phase, target in config["common_keep_targets"].items():
        rates = [result[f"{phase}_keep_rate_after_qk"] for result in summary.values()]
        spread, status = None, "no tiles in at least one policy"
        if None not in rates:
            spread = max(rates) - min(rates)
            status = ("within tolerance" if spread <= tolerance and
                      all(abs(rate - target) <= tolerance for rate in rates) else "outside tolerance")
        budgets[phase] = dict(target_keep_rate=target, spread_pp=None if spread is None else 100 * spread,
                              status=status)
    return budgets
