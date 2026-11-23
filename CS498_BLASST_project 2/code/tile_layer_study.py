"""Track standard BLASST decisions for the same physical tile across model layers."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "blasst-tile-layer-study-v1"
OUTCOMES = ("keep_keep", "keep_skip", "skip_keep", "skip_skip")
INDEX_MATCHES = ("same_query", "same_tile_pair", "same_running_pair", "same_all_indices")


def margin_indices(scores, tile_size, anchor, previous_keys):
    """Collect exact argmaxes and replay one prior-layer key per valid query row."""
    import torch
    from torch.nn.functional import pad

    width = scores.shape[-1]
    tiles = pad(scores, (0, -width % tile_size), value=-torch.inf).unflatten(-1, (-1, tile_size))
    tile_scores, local_keys = tiles.max(-1)
    running_scores, running_tiles = tile_scores.cummax(-1)
    row_margins = (tile_scores - running_scores).masked_fill(~torch.isfinite(running_scores), -torch.inf)
    margins, queries = row_margins.max(-2)
    row = queries.unsqueeze(-2)
    winning_tile = running_tiles.gather(-2, row).squeeze(-2)
    decisive_keys = local_keys.gather(-2, row).squeeze(-2)
    running_keys = decisive_keys.gather(-1, winning_tile) + winning_tile * tile_size
    tile_keys = decisive_keys + torch.arange(tiles.shape[-2], device=scores.device) * tile_size
    indices = torch.stack((queries, tile_keys, running_keys), dim=-1)
    if anchor is None:
        replay = margins
    else:
        heads = torch.arange(scores.shape[0], device=scores.device)[:, None]
        query, key, running = anchor.unbind(-1)
        # A padded key index can occur only on an invalid tile, which has no vote.
        replay = scores[heads, query, key.clamp(max=width - 1)] - scores[heads, query, running]
    snapshots = torch.stack((tile_scores.gather(-2, row).squeeze(-2),
                             running_scores.gather(-2, row).squeeze(-2), replay), dim=-1)
    valid = torch.isfinite(tile_scores)
    row_keys = local_keys.masked_fill(~valid, 255)
    row_replay = None
    if previous_keys is not None:
        # Evaluate the SAME query and key positions using the target layer's scores.
        candidates = tiles.gather(-1, previous_keys.clamp(max=tile_size - 1).unsqueeze(-1)).squeeze(-1)
        candidates = candidates.masked_fill((previous_keys == 255) | ~valid, -torch.inf)
        # A candidate that ties the current maximum is a hit even if its key differs.
        matches = (valid & (previous_keys == row_keys)).sum(-2)
        hits = (valid & (candidates == tile_scores)).sum(-2)
        import blasst_attn as ba
        approximate_margin = ba.row_tile_margins(candidates).amax(-2)
        row_replay = torch.stack((matches, hits, valid.sum(-2), approximate_margin), dim=-1)
    return indices, snapshots, row_keys, row_replay


class TileIndexTracker:
    def __init__(self):
        self.anchors = {}
        self.previous = {}

    def capture(self, chunk, layer, scores, tile_size):
        if not 0 < tile_size <= 255:
            raise ValueError("Row-key encoding requires a tile width between 1 and 255")
        previous_layer, previous_keys = self.previous.get(chunk, (None, None))
        if previous_layer != layer - 1:
            previous_keys = None
        indices, snapshots, row_keys, replay = margin_indices(
            scores, tile_size, self.anchors.get(chunk), previous_keys)
        self.anchors.setdefault(chunk, indices)
        self.previous[chunk] = (layer, row_keys)
        return (indices.tolist(), snapshots.tolist(), row_keys.transpose(-2, -1).tolist(),
                replay.tolist() if replay is not None else None)


def _snapshot(captured, head, tile, q_start):
    indices, scores, row_keys, replay = captured
    query, tile_key, running_key = indices[head][tile]
    replay_row = None
    if replay is not None:
        matches, hits, rows, margin = replay[head][tile]
        replay_row = [int(matches), int(hits), int(rows), margin if math.isfinite(margin) else None]
    return [query + q_start, tile_key, running_key, *scores[head][tile],
            bytes(row_keys[head][tile]).hex(), replay_row]


class TileLayerRecorder:
    """Keep only margins and modality labels from the dense attention observer."""

    def __init__(self, recorder, *, track_indices=False):
        self.visual = recorder.visual
        self.layers = recorder.layers
        self.heads = sorted(recorder.heads)
        self.max_tiles = recorder.max_tiles
        self.count = 0
        self.tiles = defaultdict(dict)
        self.geometry = {}
        self.indices = defaultdict(dict) if track_indices else None
        self.tracker = TileIndexTracker() if track_indices else None

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
        if self.tracker is not None:
            captured = self.tracker.capture(chunk, layer, scores[0, self.heads], ba.TILE)
        self.count += len(entries)
        if self.count > self.max_tiles:
            raise ValueError("Layer study exceeds --max-tiles; increase the limit or select fewer heads/tokens")
        for head_index, key_tile, key in entries:
            self.tiles[key][layer] = values[head_index][key_tile]
            if self.indices is not None:
                self.indices[key][layer] = _snapshot(captured, head_index, key_tile, q_start)

    def case(self, index, sample_id):
        return make_case(index, sample_id, self.tiles, self.indices)


def make_case(index, sample_id, tiles, indices=None):
    layers = sorted({layer for margins in tiles.values() for layer in margins})
    if len(layers) < 2:
        raise ValueError("Layer study requires margins from at least two recorded layers")
    return {
        "sample_index": index, "sample_id": sample_id, "layers": layers,
        "tiles": [[*key, [margins.get(layer) for layer in layers],
                   *([] if indices is None else [[indices[key].get(layer) for layer in layers]])]
                  for key, margins in sorted(tiles.items())],
    }


def _axis_kind(start, stop, spans):
    visual = sum(max(0, min(stop, end) - max(start, begin)) for begin, end in spans)
    return "T" if not visual else "V" if visual == stop - start else "mixed"


def recorded_case(data, index, score_file=None):
    """Import an existing dense Attention Lab export with a standard causal mask."""
    if (data.get("schema") != "blasst-attention-v1"
            or data["metadata"]["recorded_trajectory"] != "dense"):
        raise ValueError("Expected a dense blasst-attention-v1 export with per-tile margins")
    spans = data["visual_spans"]
    tiles = defaultdict(dict)
    indices = defaultdict(dict) if score_file else None
    tracker = TileIndexTracker() if score_file else None
    for view in sorted(data["views"], key=lambda view: view["layer"]):
        chunks = {}
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
            if score_file:
                if q not in chunks:
                    width = min(view["key_length"], q + rows)
                    block = read_score_chunk(score_file, view["score_chunks"][str(q)], rows, width)
                    chunk = (view["phase"], view["step"], view["head"], q)
                    chunks[q] = tracker.capture(chunk, view["layer"], block, data["metadata"]["tile_size"])
                k_tile = k // data["metadata"]["tile_size"]
                indices[key][view["layer"]] = _snapshot(chunks[q], 0, k_tile, q)
    return make_case(index, data["metadata"].get("sample_id"), tiles, indices)


def read_score_chunk(file, offset, rows, width):
    import torch

    file.seek(offset)
    raw = bytearray(file.read(rows * width * 4))
    if len(raw) != rows * width * 4:
        raise ValueError("Score companion is truncated")
    return torch.frombuffer(raw, dtype=torch.float32).reshape(1, rows, width)


def layer_pairs(case, cutoff):
    """Compare the same tile, head and decode step at every observed layer pair."""
    by_phase = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    layers = case["layers"]
    for phase, _step, _head, _q, _k, _rows, _cols, group, margins, *snapshots in case["tiles"]:
        winners = snapshots[0] if snapshots else [None] * len(layers)
        decisions = [(layer, value, int(value < cutoff), witness)
                     for layer, value, witness in zip(layers, margins, winners, strict=True) if value is not None]
        pairs = by_phase[phase][group]
        for i, (earlier, _, from_skip, source) in enumerate(decisions):
            for later, exact_margin, to_skip, target in decisions[i + 1:]:
                counts = pairs[earlier, later]
                counts[OUTCOMES[2 * from_skip + to_skip]] += 1
                if source is None or target is None:
                    continue
                same_query = source[0] == target[0]
                same_tile = same_query and source[1] == target[1]
                same_running = same_query and source[2] == target[2]
                counts["same_query"] += same_query
                counts["same_tile_pair"] += same_tile
                counts["same_running_pair"] += same_running
                counts["same_all_indices"] += same_tile and same_running
                if earlier == layers[0]:
                    replay = target[5]
                    outcome = OUTCOMES[2 * int(replay < cutoff) + to_skip]
                    counts[f"replay_{outcome}"] += 1
                    counts["replay_absolute_error"] += abs(replay - exact_margin)
                if later == earlier + 1 and len(target) > 7 and target[7] is not None:
                    matches, hits, rows, replay = target[7]
                    if replay is None:
                        continue
                    counts["row_matches"] += matches
                    counts["row_hits"] += hits
                    counts["row_observations"] += rows
                    outcome = OUTCOMES[2 * int(replay < cutoff) + to_skip]
                    counts[f"row_replay_{outcome}"] += 1
                    counts["row_replay_absolute_error"] += abs(replay - exact_margin)
    return by_phase


def _replay_metrics(values, prefix):
    keeps, misses, false, skips = (values[f"{prefix}_{name}"] for name in OUTCOMES)
    total = keeps + misses + false + skips
    if not total:
        return None
    return dict(observations=total, false_skips=false, missed_skips=misses,
                false_skip_rate=false / (keeps + false) if keeps + false else None,
                missed_skip_rate=misses / (misses + skips) if misses + skips else None,
                mean_absolute_margin_error=values[f"{prefix}_absolute_error"] / total)


def _reliability(values, track_indices):
    decisions = [values[name] for name in OUTCOMES]
    keep_keep, keep_skip, skip_keep, skip_skip = decisions
    total = sum(decisions)
    earlier_skips = skip_keep + skip_skip
    later_skips = keep_skip + skip_skip
    result = dict(zip(OUTCOMES, decisions, strict=True), observations=total,
                  agreement=(keep_keep + skip_skip) / total,
                  later_skip_rate=later_skips / total,
                  skip_persistence=skip_skip / earlier_skips if earlier_skips else None,
                  later_skips_recovered=skip_skip / later_skips if later_skips else None)
    if track_indices:
        result.update({name: values[name] / total for name in INDEX_MATCHES})
        anchor = _replay_metrics(values, "replay")
        if anchor:
            result["anchor_replay"] = anchor
        row_replay = _replay_metrics(values, "row_replay")
        if row_replay:
            rows = values["row_observations"]
            result["row_key_replay"] = dict(row_replay, row_observations=rows,
                                            matched_rows=values["row_matches"],
                                            max_score_hits=values["row_hits"],
                                            key_match_rate=values["row_matches"] / rows,
                                            max_score_hit_rate=values["row_hits"] / rows)
    return result


class LayerReport:
    def __init__(self, model, synthetic, lam, tile_size, *, track_indices=False):
        if not 0 < lam <= 1:
            raise ValueError("Layer study lambda must be in (0, 1]")
        self.metadata = dict(model=model, synthetic=synthetic, tile_size=tile_size,
                             cutoff=math.log(lam), recorded_trajectory="dense", **{"lambda": lam})
        if track_indices:
            self.metadata["max_indices"] = ["query", "tile_key", "running_key", "tile_score",
                                            "running_score", "first_layer_replayed_margin",
                                            "row_key_offsets_hex", "previous_layer_row_replay"]
            self.metadata["row_key_replay"] = True
        self.totals = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        self.cases = []

    def add(self, case):
        for phase, groups in layer_pairs(case, self.metadata["cutoff"]).items():
            for group, pairs in groups.items():
                for pair, counts in pairs.items():
                    self.totals[phase][group][pair].update(counts)
                    self.totals[phase]["all"][pair].update(counts)
        self.cases.append(case)

    def write(self, output, pairs_requested):
        phases = {}
        track_indices = "max_indices" in self.metadata
        for phase, groups in self.totals.items():
            phases[phase] = {}
            for group, pairs in groups.items():
                lags = defaultdict(Counter)
                for (earlier, later), counts in pairs.items():
                    lags[later - earlier].update(counts)
                phases[phase][group] = {
                    "by_lag": [dict(lag=lag, **_reliability(counts, track_indices))
                               for lag, counts in sorted(lags.items())],
                    "by_layer_pair": [dict(earlier=a, later=b, **_reliability(counts, track_indices))
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
    parser.add_argument("--study-max-indices", action="store_true",
                        help="read matching .scores.bin files to reconstruct max indices")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = None
    for index, path in enumerate(args.inputs):
        data = json.loads(path.read_text())
        metadata = data["metadata"]
        if args.study_max_indices:
            companion = path.with_suffix(".scores.bin")
            reference = data.get("score_file")
            if not reference or not companion.is_file():
                parser.error(f"{path} needs its matching .scores.bin companion")
            with companion.open("rb") as scores:
                if (scores.read(32) != reference["id"].encode("ascii")
                        or companion.stat().st_size != reference["bytes"]):
                    parser.error(f"{companion} does not match the JSON score reference")
                case = recorded_case(data, index, scores)
        else:
            case = recorded_case(data, index)
        identity = (metadata["model"], metadata["synthetic"], metadata["tile_size"])
        if report is None:
            report = LayerReport(identity[0], identity[1], args.study_lambda, identity[2],
                                 track_indices=args.study_max_indices)
        elif identity != (report.metadata["model"], report.metadata["synthetic"], report.metadata["tile_size"]):
            parser.error(f"{path} has a different model or tile size")
        report.add(case)
    report.write(args.output, len(args.inputs))
    print(f"Wrote layer study for {len(report.cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
