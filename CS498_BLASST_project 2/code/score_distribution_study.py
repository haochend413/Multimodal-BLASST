"""Measure causal BLASST score distributions relative to each row's running cutoff."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from tile_calibration import add_threshold_arguments, selected_calibration, write_json

SCHEMA = "blasst-score-distributions-v1"
GROUPS = ("T->T", "T->V", "V->T", "V->V", "mixed")
EXCESS_EDGES = (-40, -20, -12, -8, -5, -3, -2, -1, -.5, -.1, 0, .1, .5, 1, 2, 3, 5, 10)
RAW_EDGES = (-80, -40, -20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20, 40, 80)
SUPPORT_EDGES = (0, .1, .25, .5, .75, 1)


def bucket(values, edges):
    """Bins include their left edge, with an explicit underflow and overflow."""
    return np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)


class ScoreDistributions:
    def __init__(self, lam, calibration=None):
        self.lam = lam
        self.calibration = calibration
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.case_sets = defaultdict(set)
        self.cases = {}
        self.margin_cases = set()

    def threshold(self, phase):
        if self.calibration:
            return self.calibration["thresholds"][phase]["cutoff"]
        return math.log(self.lam)

    def observer(self, recorder, tile_size, case):
        visual = recorder.visual.numpy()
        heads = sorted(recorder.heads)
        self.cases[case] = {"layers": sorted(recorder.layers), "heads": heads}

        def observe(*, layer, phase, q_start, scores, margin, skip, ln_lambda, key_length):
            if layer not in recorder.layers:
                return
            width = min(key_length, q_start + scores.shape[-2])
            keys = np.zeros(width, dtype=bool)
            keys[:min(len(visual), width)] = visual[:width]
            for block in scores[0, heads, :, :width].detach().float().cpu().numpy():
                self.chunk(phase, block, keys[q_start:width], keys, tile_size, case)

        return observe

    def add_histogram(self, source, phase, group, metric, values, edges, case):
        if not np.size(values):
            return
        hist = bucket(values, edges)
        distributions = self.data[source][phase][group]
        if metric in distributions:
            distributions[metric] += hist
        else:
            distributions[metric] = hist
        self.case_sets[source, phase, group].add(case)

    def chunk(self, phase, scores, q_visual, k_visual, tile_size, case):
        cutoff = self.threshold(phase)
        running = np.full(scores.shape[0], -np.inf)
        for start in range(0, scores.shape[1], tile_size):
            block = scores[:, start:start + tile_size]
            valid = np.isfinite(block)
            if not valid.any():
                continue
            row_valid, key_valid = valid.any(1), valid.any(0)
            keys = k_visual[start:start + block.shape[1]]
            row_max = np.max(np.where(valid, block, -np.inf), axis=1)
            np.maximum(running, row_max, out=running)
            excess = np.full(block.shape, -np.inf)
            np.subtract(block, running[:, None], out=excess, where=valid)
            excess += -cutoff
            group = "mixed"
            q_types, k_types = np.unique(q_visual[row_valid]), np.unique(keys[key_valid])
            if len(q_types) == len(k_types) == 1:
                group = f"{'V' if q_types[0] else 'T'}->{'V' if k_types[0] else 'T'}"
            self.measure("physical_tiles", phase, group, block, excess, valid, case)
            if group != "mixed":
                self.measure("token_regions", phase, group, block, excess, valid, case)
                continue
            for qv, kv in ((False, False), (False, True), (True, False), (True, True)):
                region = f"{'V' if qv else 'T'}->{'V' if kv else 'T'}"
                mask = valid & (q_visual[:, None] == qv) & (keys[None, :] == kv)
                if mask.any():
                    self.measure("token_regions", phase, region, block, excess, mask, case)

    def measure(self, source, phase, group, scores, excess, valid, case):
        def add(metric, values, edges):
            self.add_histogram(source, phase, group, metric, values, edges, case)

        values = excess[valid]
        row_values = np.where(valid, excess, -np.inf).max(1)[valid.any(1)]
        add("score_excess", values, EXCESS_EDGES)
        add("raw_scores", scores[valid], RAW_EDGES)
        add("row_excess", row_values, EXCESS_EDGES)
        if source == "physical_tiles":
            add("tile_excess", [row_values.max()], EXCESS_EDGES)
            add("row_support", [np.count_nonzero(row_values >= 0) / len(row_values)], SUPPORT_EDGES)
            add("score_support", [np.count_nonzero(values >= 0) / len(values)], SUPPORT_EDGES)
            # Isolate exactly zero support from the (0, 10%) histogram bin.
            dist = self.data[source][phase][group]
            for metric, sample in (("row_support", row_values), ("score_support", values)):
                if not (sample >= 0).any():
                    dist[metric][1] -= 1
                    dist[metric][0] += 1

    def add_tile_report(self, report):
        metadata = report["metadata"]
        if (report.get("schema") != "blasst-tile-layer-study-v1"
                or metadata.get("attention_mask_policy", "causal") != "causal"):
            raise ValueError("Tile margins must come from a causal cross-layer report")
        for case in report["cases"]:
            identity = case["sample_id"] or case["sample_index"]
            self.margin_cases.add(identity)
            groups = defaultdict(list)
            for tile in case["tiles"]:
                groups[tile[0], tile[7]].extend(margin for margin in tile[8] if margin is not None)
            for (phase, group), margins in groups.items():
                self.add_histogram("saved_tile_margins", phase, group, "tile_excess",
                                   np.asarray(margins) - self.threshold(phase), EXCESS_EDGES, identity)

    def add_score_export(self, path):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        reference = data.get("score_file")
        if (data.get("schema") != "blasst-attention-v1" or not reference
                or metadata["recorded_trajectory"] != "dense"
                or metadata.get("attention_mask_policy", "causal") != "causal"):
            raise ValueError(f"{path} needs a dense causal export with --score-data")
        identity = metadata.get("sample_id") or path.stem
        self.cases[identity] = {"model": metadata["model"], "synthetic": metadata["synthetic"],
                                "layers": sorted({v["layer"] for v in data["views"]}),
                                "heads": sorted({v["head"] for v in data["views"]})}
        visual = np.zeros(max(v["key_length"] for v in data["views"]), dtype=bool)
        for start, end in data["visual_spans"]:
            visual[start:end] = True
        companion = path.with_suffix(".scores.bin")
        with companion.open("rb") as stream:
            if stream.read(32).decode("ascii") != reference["id"] or companion.stat().st_size != reference["bytes"]:
                raise ValueError(f"{companion} does not match the JSON score reference")
            for view in data["views"]:
                rows_by_start = {tile["q"]: tile["rows"] for tile in view["tiles"]}
                for q_start, offset in view["score_chunks"].items():
                    q_start = int(q_start)
                    rows = rows_by_start[q_start]
                    width = min(view["key_length"], q_start + rows)
                    stream.seek(offset)
                    scores = np.frombuffer(stream.read(rows * width * 4), dtype="<f4").reshape(rows, width)
                    self.chunk(view["phase"], scores, visual[q_start:q_start + rows], visual[:width],
                               metadata["tile_size"], identity)

    def report(self):
        sources = {}
        for source, phases in self.data.items():
            sources[source] = {
                phase: {group: {"cases_observed": len(self.case_sets[source, phase, group]),
                                **{metric: bins.tolist() for metric, bins in h.items()}}
                        for group, h in groups.items()}
                for phase, groups in phases.items()}
        phases = {phase for source in sources.values() for phase in source}
        return dict(schema=SCHEMA, exact_score_cases=self.cases, saved_tile_margin_cases=len(self.margin_cases),
                    sources=sources, calibration=self.calibration,
                    lambda_by_phase={phase: math.exp(self.threshold(phase)) for phase in sorted(phases)},
                    bins=dict(excess=EXCESS_EDGES, raw=RAW_EDGES, support=SUPPORT_EDGES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-export", type=Path, action="append", default=[])
    parser.add_argument("--tile-report", type=Path, help="additional causal cross-layer tile margins")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", type=Path)
    add_threshold_arguments(parser)
    args = parser.parse_args()
    if not (args.score_export or args.tile_report):
        parser.error("provide --score-export, --tile-report, or both")
    study = ScoreDistributions(args.study_lambda or .1, selected_calibration(args, parser))
    for path in args.score_export:
        study.add_score_export(path)
    if args.tile_report:
        study.add_tile_report(json.loads(args.tile_report.read_text()))
    report = study.report()
    write_json(args.output, report)
    if args.plot:
        from plot_score_distributions import plot_report
        plot_report(report, args.plot)
    print(f"Wrote causal score distributions to {args.output}")


if __name__ == "__main__":
    main()
