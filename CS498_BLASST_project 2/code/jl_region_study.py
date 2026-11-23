"""Paired audit of modality packing and per-key-region running maxima."""
from __future__ import annotations

import math
from collections import defaultdict
from copy import deepcopy
from typing import NamedTuple

import torch

import blasst_attn as ba
from jl_gap_study import TILE_GROUPS, nested_basis

SCHEMA = "blasst-jl-region-audit-v1"
METRICS = (
    "tiles", "full_skips", "projected_skips", "matched_skips", "false_skips", "missed_skips",
    "valid_pairs", "full_skipped_pairs", "projected_skipped_pairs", "dense_weight",
    "full_skipped_dense_weight", "projected_skipped_dense_weight", "false_skip_dense_weight",
    "missed_skip_dense_weight",
)


class Span(NamedTuple):
    start: int
    end: int
    region: int


def axis_tiles(visual, size, *, packed):
    """Start each contiguous modality run at a tile boundary, padding only its tail."""
    end = len(visual)
    starts = [0]
    if packed:
        starts.extend((visual[1:] != visual[:-1]).nonzero().flatten().add(1).tolist())
    return [Span(start, min(start + size, hi), region)
            for region, (lo, hi) in enumerate(zip(starts, [*starts[1:], end]))
            for start in range(lo, hi, size)]


def region_margins(tile_max, keys):
    """Reset at contiguous key-run transitions; the first valid tile establishes the bar."""
    cuts = [0, *(i for i in range(1, len(keys)) if keys[i - 1].region != keys[i].region), len(keys)]
    return torch.cat([ba.row_tile_margins(tile_max[..., lo:hi])
                      for lo, hi in zip(cuts, cuts[1:])], dim=-1)


def _tile_geometry(scores, probabilities, q_visual, key_visual, keys):
    groups, pairs, weights = [], [], []
    for lo, hi, _ in keys:
        valid = torch.isfinite(scores[..., lo:hi])
        rows, cols = valid.any(-1), valid.any(-2)
        q_types = [(rows & (q_visual == kind)).any(-1) for kind in (False, True)]
        k_types = [(cols & (key_visual[lo:hi] == kind)).any(-1) for kind in (False, True)]
        mixed = (q_types[0] & q_types[1]) | (k_types[0] & k_types[1])
        pure = [q & k & ~mixed for q in q_types for k in k_types]
        groups.append(torch.stack([rows.any(-1), *pure, mixed]))
        pairs.append(valid.sum(dim=(-1, -2)))
        weights.append(probabilities[..., lo:hi].sum(dim=(-1, -2)))
    return (torch.stack(groups, dim=-1), torch.stack(pairs, dim=-1), torch.stack(weights, dim=-1))


def _counts(geometry, margins, projected_margins, ln_lam):
    groups, pairs, weights = geometry
    full = margins.amax(-2) < ln_lam
    projected = projected_margins.amax(-2) < ln_lam
    all_tiles = torch.ones_like(full)
    false_skips, missed = projected & ~full, full & ~projected
    counts = torch.stack([all_tiles, full, projected, full & projected, false_skips, missed])
    weighted = torch.stack([pairs, pairs * full, pairs * projected, weights,
                            weights * full, weights * projected, weights * false_skips, weights * missed])
    return torch.cat(((groups[:, None] & counts).sum(dim=(-2, -1)),
                      (groups[:, None] * weighted).sum(dim=(-2, -1), dtype=torch.float64)), dim=-1)


class JLRegionStudy:
    """Compare original, packed, and region-reset masks on one dense trajectory."""

    def __init__(self, dim, lam, seed, visual, layers, heads):
        self.dim, self.lam, self.seed = dim, lam, seed
        self.visual, self.layers, self.heads = visual, layers, sorted(heads)
        self.phases = defaultdict(dict)

    @torch.no_grad()
    def observe(self, *, layer, phase, query, key, value, attention_mask, scaling):
        if layer not in self.layers:
            return
        q_len, width = query.shape[-2:]
        if self.dim >= width:
            raise ValueError(f"JL dimension {self.dim} must be smaller than head width {width}")
        self.width = width
        if not hasattr(self, "basis"):
            self.basis = nested_basis(width, self.dim, self.seed, query.device)
        key_len, offset = key.shape[-2], key.shape[-2] - q_len
        visual = torch.zeros(key_len, dtype=torch.bool)
        visual[:min(key_len, len(self.visual))] = self.visual[:key_len]
        q_visual = visual[offset:].to(query.device)
        k_visual = visual.to(query.device)
        q = query[0, self.heads].float()
        groups = query.shape[1] // key.shape[1]
        k = key[0, [head // groups for head in self.heads]].float()
        q_short, k_short = q.double() @ self.basis, k.double() @ self.basis
        k_t, k_short_t = k.transpose(-1, -2), k_short.transpose(-1, -2)
        if attention_mask is not None and attention_mask.shape[1] != 1:
            attention_mask = attention_mask[:, self.heads]
        positions = torch.arange(key_len, device=query.device)
        results = self.phases[phase]
        ln_lam = math.log(self.lam)
        for packed in (False, True):
            keys = axis_tiles(visual, ba.TILE, packed=packed)
            queries = axis_tiles(visual[offset:], ba.TILE, packed=packed)
            for start, end, _ in queries:
                # Future-only tiles receive no vote. Keep the full last eligible tile's span.
                limit = end + offset
                active_keys = [span for span in keys if span.start < limit]
                last = active_keys[-1].end
                allowed = positions[:last] <= positions[start + offset:end + offset, None]
                mask = ba.mask_chunk(attention_mask, start, end, q_len, last)
                exact = (q[:, start:end] @ k_t[..., :last]) * scaling
                jl = ((q_short[:, start:end] @ k_short_t[..., :last]) *
                      (scaling * width / self.dim)).float()
                exact = _mask_scores(exact, allowed, mask)
                jl = _mask_scores(jl, allowed, mask)
                probabilities = torch.softmax(exact, dim=-1).nan_to_num()
                geometry = _tile_geometry(exact, probabilities, q_visual[start:end], k_visual, active_keys)
                exact_max, jl_max = (_row_max(scores, active_keys) for scores in (exact, jl))
                name = "packed_shared" if packed else "original_shared"
                self._add(results, name, _counts(geometry, ba.row_tile_margins(exact_max),
                                                ba.row_tile_margins(jl_max), ln_lam))
                if packed:
                    self._add(results, "packed_reset", _counts(geometry, region_margins(exact_max, active_keys),
                                                               region_margins(jl_max, active_keys), ln_lam))

    @staticmethod
    def _add(results, name, counts):
        if name in results:
            results[name] += counts
        else:
            results[name] = counts

    def summary(self, metadata):
        phases = {}
        for phase, policies in self.phases.items():
            phases[phase] = {}
            for policy, counts in policies.items():
                phases[phase][policy] = {}
                for group, values in zip(TILE_GROUPS, counts.tolist(), strict=True):
                    result = {key: value if key.endswith("weight") else int(value)
                              for key, value in zip(METRICS, values, strict=True)}
                    result.update(pairs_observed=int(result["tiles"] > 0),
                                  pairs_with_false_skips=int(result["false_skips"] > 0))
                    phases[phase][policy][group] = result
        return {
            "schema": SCHEMA, "model": metadata["model"], "synthetic": metadata["synthetic"],
            "head_width": self.width, "dimension": self.dim, "lambda": self.lam, "seed": self.seed,
            "layers": sorted(self.layers), "heads": self.heads, "tile_size": ba.TILE,
            "pairs_completed": 1, "phases": phases,
        }


def _mask_scores(scores, allowed, mask):
    scores = scores.masked_fill(~allowed, -torch.inf)
    if mask is None:
        return scores
    mask = mask[0]
    return scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask.float()


def _row_max(scores, keys):
    return torch.stack([scores[..., lo:hi].amax(-1) for lo, hi, _ in keys], dim=-1)


def merge_region_audits(total, sample):
    if total is None:
        return deepcopy(sample)
    if any(total[key] != value for key, value in sample.items() if key not in ("phases", "pairs_completed")):
        raise ValueError("Region audit settings must match before pooling")
    for phase, policies in sample["phases"].items():
        if phase not in total["phases"]:
            total["phases"][phase] = deepcopy(policies)
            continue
        for policy, groups in policies.items():
            for group, counts in groups.items():
                for key, value in counts.items():
                    total["phases"][phase][policy][group][key] += value
    total["pairs_completed"] += sample["pairs_completed"]
    return total
