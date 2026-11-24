"""Choose fixed BLASST thresholds from saved dense tile margins."""
from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "blasst-keep-calibration-v1"
IDENTITY_FIELDS = ("model", "synthetic", "tile_size")
TARGETS = (20, 50, 80)


def case_id(case):
    return case["sample_id"] if case["sample_id"] is not None else case["sample_index"]


def tile_margins(case):
    for tile in case["tiles"]:
        for margin in tile[8]:
            if margin is not None:
                yield tile[0], tile[7], margin


def keep_summary(counts):
    kept, total = counts
    return dict(kept=kept, tiles=total, keep_rate=kept / total)


def case_keep_counts(case, cutoffs):
    counts = defaultdict(lambda: defaultdict(Counter))
    for phase, group, margin in tile_margins(case):
        keep = margin >= cutoffs[phase]
        for target in (counts[phase][group], counts[phase]["all"], counts["all"]["all"]):
            target["kept"] += keep
            target["tiles"] += 1
    return counts


def summarize_counts(counts):
    return {phase: {group: keep_summary((count["kept"], count["tiles"]))
                    for group, count in groups.items()}
            for phase, groups in counts.items()}


def best_threshold(margins, target):
    """Find the closest attainable keep rate under the strict margin < log(lambda) test."""
    margins = sorted(margins)
    unique = sorted(set(margins))
    min_lambda = math.ulp(0.0)
    # Search the intervals between observed margins so tied margins move together.
    min_cutoff = math.log(min_lambda)
    limits = [min_cutoff, *[value for value in unique if min_cutoff < value < 0], 0.0]
    candidates = [min_lambda, 1.0]
    for lower, upper in zip(limits, limits[1:]):
        lam = math.exp((lower + upper) / 2)
        candidates.extend((lam, math.nextafter(lam, 0.0), math.nextafter(lam, 1.0)))
    total = len(margins)
    scored = ((total - bisect_left(margins, math.log(lam)), lam) for lam in candidates if 0 < lam <= 1)
    kept, lam = min(scored, key=lambda pair: (abs(100 * pair[0] - target * total), -pair[0], -pair[1]))
    minimum = total - bisect_left(margins, 0)
    return dict(keep_summary((kept, total)), **{"lambda": lam}, cutoff=math.log(lam),
                minimum_keep_rate=minimum / total, target_below_minimum=100 * minimum > target * total)


def calibrate(metadata, cases, scope):
    groups = defaultdict(list)
    phases = set()
    for case in cases:
        for phase, _, margin in tile_margins(case):
            groups[phase if scope == "phase" else "all"].append(margin)
            phases.add(phase)
    if not groups:
        raise ValueError("Calibration needs recorded tile margins")
    ids = [case_id(case) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Calibration cases must have distinct sample IDs")
    return dict(schema=SCHEMA, scope=scope,
                metadata={field: metadata[field] for field in IDENTITY_FIELDS},
                calibration_case_ids=ids, phases=sorted(phases),
                targets={str(target): {group: best_threshold(margins, target)
                                       for group, margins in groups.items()}
                         for target in TARGETS})


def load_calibration(path, target, parser):
    try:
        config = json.loads(path.read_text())
        if config["schema"] != SCHEMA or config["scope"] not in ("global", "phase"):
            raise ValueError("unsupported calibration format")
        groups = config["targets"][str(target)]
        thresholds = {phase: groups[phase if config["scope"] == "phase" else "all"]
                      for phase in config["phases"]}
        for threshold in thresholds.values():
            if (not 0 < threshold["lambda"] <= 1
                    or threshold["cutoff"] != math.log(threshold["lambda"])):
                raise ValueError("saved cutoff must equal log(lambda)")
        return dict(target_keep_percent=target, scope=config["scope"],
                    metadata=config["metadata"], calibration_case_ids=config["calibration_case_ids"],
                    thresholds=thresholds)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"Invalid calibration config {path} ({error})")


def add_threshold_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--study-lambda", type=float, help="fixed offline threshold (default 0.1)")
    group.add_argument("--calibration", type=Path, help="saved keep-rate calibration JSON")
    parser.add_argument("--target-keep", type=int, choices=TARGETS, help="keep percentage from the saved config")


def selected_calibration(args, parser):
    if bool(args.calibration) != (args.target_keep is not None):
        parser.error("--calibration and --target-keep must be used together")
    if args.study_lambda is not None and not 0 < args.study_lambda <= 1:
        parser.error("--study-lambda must be in (0, 1]")
    return load_calibration(args.calibration, args.target_keep, parser) if args.calibration else None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    pending.replace(path)
