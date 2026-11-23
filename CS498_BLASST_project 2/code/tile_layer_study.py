"""Track standard BLASST decisions for the same physical tile across model layers."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

SCHEMA = "blasst-tile-layer-study-v1"
OUTCOMES = ("keep_keep", "keep_skip", "skip_keep", "skip_skip")


class TileLayerRecorder:
    """Keep only margins and modality labels from the dense attention observer."""

    def __init__(self, recorder):
        self.visual = recorder.visual
        self.layers = recorder.layers
        self.heads = sorted(recorder.heads)
        self.max_tiles = recorder.max_tiles
        self.count = 0
        self.tiles = defaultdict(dict)
        self.geometry = {}

    def observe(self, *, layer, phase, q_start, scores, margin, skip, ln_lambda, key_length):
        if layer not in self.layers:
            return

        import torch
        from torch.nn.functional import pad

        import blasst_attn as ba
        from jl_gap_study import TILE_GROUPS, tile_groups

        n_rows = scores.shape[-2]
        step = -1 if phase == "prefill" else key_length - len(self.visual) - 1
        chunk = (phase, step, q_start, n_rows, key_length)
        if chunk not in self.geometry:
            visual = torch.zeros(key_length, dtype=torch.bool, device=scores.device)
            visual[:len(self.visual)] = self.visual[:key_length]
            keys = pad(visual.to(torch.int8), (0, -key_length % ba.TILE), value=-1)
            full = pad(scores[0, self.heads], (0, -key_length % ba.TILE), value=-torch.inf)
            groups = tile_groups(full.unflatten(-1, (-1, ba.TILE)),
                                 visual[q_start:q_start + n_rows],
                                 keys.unflatten(-1, (-1, ba.TILE)))
            entries = []
            for group, head_index, key_tile in groups[1:].nonzero().tolist():
                start = key_tile * ba.TILE
                key = (phase, step, self.heads[head_index], q_start, start,
                       n_rows, min(ba.TILE, key_length - start), TILE_GROUPS[group + 1])
                entries.append((head_index, key_tile, key))
            self.geometry[chunk] = entries

        values = margin[0, self.heads].tolist()
        entries = self.geometry[chunk]
        self.count += len(entries)
        if self.count > self.max_tiles:
            raise ValueError("Layer study exceeds --max-tiles; increase the limit or select fewer heads/tokens")
        for head_index, key_tile, key in entries:
            self.tiles[key][layer] = values[head_index][key_tile]

    def case(self, index, sample_id):
        return make_case(index, sample_id, self.tiles)


def make_case(index, sample_id, tiles):
    layers = sorted({layer for margins in tiles.values() for layer in margins})
    if len(layers) < 2:
        raise ValueError("Layer study requires margins from at least two recorded layers")
    return {
        "sample_index": index, "sample_id": sample_id, "layers": layers,
        "tiles": [[*key, [margins.get(layer) for layer in layers]]
                  for key, margins in sorted(tiles.items())],
    }


def _axis_kind(start, stop, spans):
    visual = sum(max(0, min(stop, end) - max(start, begin)) for begin, end in spans)
    return "T" if not visual else "V" if visual == stop - start else "mixed"


def recorded_case(data, index):
    """Import an existing dense Attention Lab export with a standard causal mask."""
    if (data.get("schema") != "blasst-attention-v1"
            or data["metadata"]["recorded_trajectory"] != "dense"):
        raise ValueError("Expected a dense blasst-attention-v1 export with per-tile margins")
    spans = data["visual_spans"]
    tiles = defaultdict(dict)
    for view in data["views"]:
        for tile in view["tiles"]:
            q, k, rows, cols = (tile[field] for field in ("q", "k", "rows", "cols"))
            # A causal tile can contain masked lanes even when it has a decision.
            q_kind = _axis_kind(max(q, k), q + rows, spans)
            k_kind = _axis_kind(k, min(k + cols, q + rows), spans)
            group = "mixed" if "mixed" in (q_kind, k_kind) else f"{q_kind}->{k_kind}"
            key = (view["phase"], view["step"], view["head"], q, k, rows, cols, group)
            if view["layer"] in tiles[key]:
                raise ValueError(f"Duplicate tile measurement for {key} at layer {view['layer']}")
            tiles[key][view["layer"]] = tile["margin"]
    return make_case(index, data["metadata"].get("sample_id"), tiles)


def layer_pairs(case, cutoff):
    """Compare the same tile, head and decode step at every observed layer pair."""
    by_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])))
    layers = case["layers"]
    for phase, _step, _head, _q, _k, _rows, _cols, group, margins in case["tiles"]:
        decisions = [(layer, int(value < cutoff))
                     for layer, value in zip(layers, margins, strict=True) if value is not None]
        pairs = by_phase[phase][group]
        for i, (earlier, from_skip) in enumerate(decisions):
            for later, to_skip in decisions[i + 1:]:
                pairs[earlier, later][2 * from_skip + to_skip] += 1
    return by_phase


def _reliability(values):
    keep_keep, keep_skip, skip_keep, skip_skip = values
    total = sum(values)
    earlier_skips = skip_keep + skip_skip
    later_skips = keep_skip + skip_skip
    return dict(zip(OUTCOMES, values, strict=True), observations=total,
                agreement=(keep_keep + skip_skip) / total,
                later_skip_rate=later_skips / total,
                skip_persistence=skip_skip / earlier_skips if earlier_skips else None,
                later_skips_recovered=skip_skip / later_skips if later_skips else None)


class LayerReport:
    def __init__(self, model, synthetic, lam, tile_size):
        if not 0 < lam <= 1:
            raise ValueError("Layer study lambda must be in (0, 1]")
        self.metadata = dict(model=model, synthetic=synthetic, tile_size=tile_size,
                             cutoff=math.log(lam), recorded_trajectory="dense", **{"lambda": lam})
        self.totals = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])))
        self.cases = []

    def add(self, case):
        for phase, groups in layer_pairs(case, self.metadata["cutoff"]).items():
            for group, pairs in groups.items():
                for pair, counts in pairs.items():
                    self._add(self.totals[phase][group][pair], counts)
                    self._add(self.totals[phase]["all"][pair], counts)
        self.cases.append(case)

    @staticmethod
    def _add(total, values):
        for index, count in enumerate(values):
            total[index] += count

    def write(self, output, pairs_requested):
        phases = {}
        for phase, groups in self.totals.items():
            phases[phase] = {}
            for group, pairs in groups.items():
                lags = defaultdict(lambda: [0, 0, 0, 0])
                for (earlier, later), counts in pairs.items():
                    self._add(lags[later - earlier], counts)
                phases[phase][group] = {
                    "by_lag": [dict(lag=lag, **_reliability(counts)) for lag, counts in sorted(lags.items())],
                    "by_layer_pair": [dict(earlier=a, later=b, **_reliability(counts))
                                      for (a, b), counts in sorted(pairs.items())],
                }
        payload = dict(schema=SCHEMA, metadata=self.metadata, pairs_requested=pairs_requested,
                       pairs_completed=len(self.cases), phases=phases, cases=self.cases)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
        pending.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="dense Attention Lab JSON exports")
    parser.add_argument("--study-lambda", type=float, default=0.1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = None
    for index, path in enumerate(args.inputs):
        data = json.loads(path.read_text())
        case = recorded_case(data, index)
        metadata = data["metadata"]
        identity = (metadata["model"], metadata["synthetic"], metadata["tile_size"])
        if report is None:
            report = LayerReport(identity[0], identity[1], args.study_lambda, identity[2])
        elif identity != (report.metadata["model"], report.metadata["synthetic"], report.metadata["tile_size"]):
            parser.error(f"{path} has a different model or tile size")
        report.add(case)
    report.write(args.output, len(args.inputs))
    print(f"Wrote layer study for {len(report.cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
