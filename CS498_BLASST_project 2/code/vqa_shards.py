"""Checkpoint paired VQA shards and merge their predictions and accuracy."""
from __future__ import annotations

import argparse
import fcntl
import json
from collections import defaultdict
from contextlib import contextmanager

from answer_benchmarks import ANSWER_BENCHMARKS, VQAV2
from binary_tv_pipeline import BINARY_TV_COUNTS
from decode_visual_links import DECODE_HISTORY_COUNTS, DECODE_LINK_COUNTS
from tile_calibration import write_json
from tv_guided_vv import TOKEN_GATE_COUNTS, VISUAL_QUERY_COUNTS

CHECKPOINT_INTERVAL = 25
MAX_POLICIES = ("prefill_max_policies", "decode_max_policies")
TILE_COUNTS = ("prefill_tiles", "prefill_tiles_skipped", "decode_tiles", "decode_tiles_skipped",
               "prefill_qk_bypassed", *DECODE_LINK_COUNTS, *DECODE_HISTORY_COUNTS, *TOKEN_GATE_COUNTS,
               *VISUAL_QUERY_COUNTS, *BINARY_TV_COUNTS)
CASE_FIELDS = ("answer", "vqa_score", "rouge_l", "generated_tokens", "pure_vv_tiles", "vv_tiles_bypassed",
               "answer_changed", *TILE_COUNTS)
PROTOCOLS_BY_METRIC = {protocol.metric: protocol for protocol in ANSWER_BENCHMARKS.values()}


@contextmanager
def claim(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another worker is writing {output}") from error
        yield


def partition(total, count, index):
    if not 1 <= count <= total or not 0 <= index < count:
        raise ValueError("shard count must fit the selected cases and shard index must be in [0, count)")
    return total * index // count, total * (index + 1) // count


def shard_bounds(shard):
    start = shard.get("selection_start", 0)
    total = shard.get("selection_count", shard["manifest_questions"])
    if not 0 <= start < start + total <= shard["manifest_questions"]:
        raise ValueError("requested sample range is outside the manifest")
    lo, hi = partition(total, shard["count"], shard["index"])
    return start + lo, start + hi


def resume(report, output, samples, start):
    if not output.exists():
        return
    saved = json.loads(output.read_text())
    expected = {key: value for key, value in report.items() if key != "cases"}
    if (any(saved.get(key) != value for key, value in expected.items())
            or any(saved.get(key) != report.get(key) for key in MAX_POLICIES)):
        raise ValueError("checkpoint settings differ from this run")
    cases = saved["cases"]
    if len(cases) > len(samples) or any(
        case["sample_index"] != start + index
        or case["question_id"] != samples[index]["benchmark"]["question_id"]
        for index, case in enumerate(cases)
    ):
        raise ValueError("checkpoint questions do not match the beginning of this shard")
    report["cases"] = cases


def accumulate(totals, case, *, reference="baseline", protocol=VQAV2):
    baseline = case["results"][reference][protocol.case_score]
    for name, result in case["results"].items():
        counts = totals[name]
        score = result[protocol.case_score]
        counts[f"{protocol.prefix}_score_sum"] += score
        counts["answers_scored"] += 1
        for key in ("pure_vv_tiles", "vv_tiles_bypassed", *TILE_COUNTS):
            if key in result:
                counts[key] += result[key]
        if "generated_tokens" in result:
            counts["generated_tokens"] += result["generated_tokens"]
            counts["generation_lengths_recorded"] += 1
            counts["decode_steps"] += max(result["generated_tokens"] - 1, 0)
        if name != reference:
            counts["answers_changed"] += result["answer_changed"]
            counts[f"{protocol.prefix}_better"] += score > baseline
            counts[f"{protocol.prefix}_worse"] += score < baseline


def summarize(totals, *, reference="baseline", protocol=VQAV2):
    summary = {}
    score_field = f"{protocol.prefix}_score_sum"
    baseline = totals[reference][score_field] / totals[reference]["answers_scored"]
    for name, totals_for_variant in totals.items():
        score = totals_for_variant[score_field] / totals_for_variant["answers_scored"]
        summary[name] = dict(totals_for_variant, **{protocol.summary_score: score})
        if observed := totals_for_variant.get("generation_lengths_recorded"):
            summary[name]["mean_generated_tokens"] = totals_for_variant["generated_tokens"] / observed
            summary[name]["mean_decode_steps"] = totals_for_variant["decode_steps"] / observed
        if name != reference:
            summary[name][f"{protocol.prefix}_delta_pp"] = score - baseline
    if "whole_visual_query" in summary and "visual_vv_only" in summary:
        full = summary["whole_visual_query"]
        full[f"{protocol.prefix}_delta_vs_vv_only_pp"] = (
            full[protocol.summary_score] - summary["visual_vv_only"][protocol.summary_score]
        )
    return summary


def save_predictions(output, predictions):
    for name, rows in predictions.items():
        write_json(output.with_name(f"{output.stem}-{name}-predictions.json"), rows)


def merge(paths, output):
    config_keys = ("schema", "model", "synthetic", "tile_size", "blasst_lambda", "decode_lambda",
                   "answer_metric", "variants", "answer_protocol", "image_max_pixels", "max_prompt_tokens")
    totals = defaultdict(lambda: defaultdict(int))
    predictions = defaultdict(list)
    seen_shards = set()
    question_ids = set()
    config, manifest = None, None
    for path in paths:
        report = json.loads(path.read_text())
        shard = report.get("vqa_shard")
        if not shard:
            raise ValueError(f"{path} is not a VQA shard report")
        expected_config = {key: report[key] for key in config_keys}
        for key in (*MAX_POLICIES, "benchmark", "variant_labels", "decode_visual_link_study", "token_top_k",
                    "binary_tv_pipeline", "prefill_visual_query_study"):
            if key in report:
                expected_config[key] = report[key]
        expected_manifest = {key: shard[key] for key in ("count", "manifest_questions", "manifest_sha256")}
        for key in ("selection_start", "selection_count"):
            if key in shard:
                expected_manifest[key] = shard[key]
        if config is None:
            config, manifest = expected_config, expected_manifest
        elif config != expected_config or manifest != expected_manifest:
            raise ValueError(f"{path} uses different run or manifest settings")
        if shard["index"] in seen_shards:
            raise ValueError(f"duplicate shard index {shard['index']}")
        start, stop = shard_bounds(shard)
        cases = report["cases"]
        if (len(cases) != stop - start or report["pairs_requested"] != stop - start
                or any(case["sample_index"] != index for index, case in enumerate(cases, start=start))):
            raise ValueError(f"{path} is incomplete or has an invalid case range")
        for case in cases:
            qid = case["question_id"]
            if qid in question_ids or case["results"].keys() != config["variants"].keys():
                raise ValueError(f"{path} has a duplicate question ID or incomplete pair")
            question_ids.add(qid)
            accumulate(totals, case, protocol=PROTOCOLS_BY_METRIC[config["answer_metric"]])
            for name, result in case["results"].items():
                predictions[name].append({"question_id": qid, "answer": result["answer"]})
        seen_shards.add(shard["index"])
    missing = set(range(manifest["count"])) - seen_shards
    if missing:
        raise ValueError(f"{len(missing)} missing shards; first missing shard is {min(missing)}")
    total = manifest.get("selection_count", manifest["manifest_questions"])
    merged = dict(config, pairs_requested=total, pairs_completed=len(question_ids), cases=[],
                  summary=summarize(totals, protocol=PROTOCOLS_BY_METRIC[config["answer_metric"]]), vqa_merged=manifest)
    for pairs in predictions.values():
        pairs.sort(key=lambda row: row["question_id"])
    save_predictions(output, predictions)
    write_json(output, merged)
    return merged


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = merge(args.shards, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    protocol = PROTOCOLS_BY_METRIC[report["answer_metric"]]
    for policy, values in report["summary"].items():
        print(f"{policy}: {protocol.format_score(values)} on {values['answers_scored']} questions")
    print(f"Wrote merged VQA report to {args.output}")


if __name__ == "__main__":
    main()
