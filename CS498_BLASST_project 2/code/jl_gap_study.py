"""Paired JL dimension sweeps and modality-resolved BLASST gap histograms."""
from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn.functional as F

import blasst_attn as ba

SCHEMA = "blasst-jl-sweep-v1"
REGIONS = ("T->T", "T->V", "V->T", "V->V")
TILE_GROUPS = ("all", *REGIONS, "mixed")
OVERFLOW = 80
COUNTS = ("tiles", "full_skips", "projected_skips", "matched_skips", "false_skips", "missed_skips")


def nested_basis(width, dim, seed, device):
    rng = torch.Generator().manual_seed(seed)
    rotation = torch.randn(width, width, generator=rng, dtype=torch.float64)
    return torch.linalg.qr(rotation).Q[:, :dim].to(device)


def gap_histograms(tiles, q_visual, k_visual):
    """Pool running max minus tile-row max within each token modality pair."""
    running = tiles.amax(-1).cummax(-1).values
    bins = []
    for k_kind in (0, 1):
        # Restrict the current tile's max to this modality; keep the original mixed-modality bar.
        tile_max = tiles.masked_fill(k_visual != k_kind, -torch.inf).amax(-1)
        gaps = (running - tile_max).clamp(min=0, max=OVERFLOW)
        row_bins = gaps[torch.isfinite(tile_max)].long()
        region_offsets = (2 * q_visual[:, None].long() + k_kind) * (OVERFLOW + 1)
        bins.append(row_bins + region_offsets.expand_as(tile_max)[torch.isfinite(tile_max)])
    return torch.bincount(torch.cat(bins), minlength=len(REGIONS) * (OVERFLOW + 1)).reshape(len(REGIONS), -1)


def tile_groups(tiles, q_visual, k_visual):
    """Assign whole-tile votes to pure modality groups or mixed using valid lanes."""
    valid = torch.isfinite(tiles)
    rows, columns = valid.any(-1), valid.any(-3)
    q_has_v = (rows & q_visual[:, None]).any(-2)
    q_has_t = (rows & ~q_visual[:, None]).any(-2)
    k_has_v = (columns & (k_visual == 1)).any(-1)
    k_has_t = (columns & (k_visual == 0)).any(-1)
    active = rows.any(-2)
    mixed = (q_has_t & q_has_v) | (k_has_t & k_has_v)
    pure = tuple(active & ~mixed & q & k for q in (q_has_t, q_has_v) for k in (k_has_t, k_has_v))
    return torch.stack((active, *pure, mixed))


def decision_counts(groups, full_skip, projected_skip):
    masks = torch.stack((torch.ones_like(full_skip), full_skip, projected_skip,
                         full_skip & projected_skip, ~full_skip & projected_skip,
                         full_skip & ~projected_skip))
    return (groups[:, None] & masks).sum(dim=(-1, -2))


class JLGapStudy:
    """Reuse each exact score chunk and project the same post-RoPE Q/K at every width."""

    def __init__(self, dimensions, lam, seed, visual, layers, heads):
        self.dimensions = dimensions
        self.lam, self.seed = lam, seed
        self.ln_lam = math.log(lam)
        self.visual, self.layers, self.heads = visual, layers, sorted(heads)
        self.phases = {}

    @torch.no_grad()
    def observe(self, *, layer, phase, query, key, value, attention_mask, scaling):
        if layer not in self.layers:
            return
        self.query_len, self.width = query.shape[-2:]
        if self.dimensions[-1] >= self.width:
            raise ValueError(f"JL sweep dimensions must be smaller than head width {self.width}")
        if not hasattr(self, "basis"):
            self.basis = nested_basis(self.width, self.dimensions[-1], self.seed, query.device)
        groups = query.shape[1] // key.shape[1]
        self.q_short = query[:, self.heads].double() @ self.basis
        self.k_short = key[:, [head // groups for head in self.heads]].double() @ self.basis
        self.mask = attention_mask
        if self.mask is not None and self.mask.shape[1] != 1:
            self.mask = self.mask[:, self.heads]
        self.scaling = scaling
        key_len = key.shape[-2]
        self.key_positions = torch.arange(key_len, device=query.device)
        self.key_visual = torch.zeros(key_len, device=query.device, dtype=torch.bool)
        self.key_visual[:min(key_len, len(self.visual))] = self.visual[:key_len]
        self.key_types = F.pad(self.key_visual.to(torch.int8), (0, -key_len % ba.TILE), value=-1)
        self.key_types = self.key_types.unflatten(-1, (-1, ba.TILE))

    @torch.no_grad()
    def record(self, *, layer, phase, q_start, scores, margin, skip, ln_lambda, key_length):
        if layer not in self.layers:
            return
        n_rows = scores.shape[-2]
        start = q_start - (key_length - self.query_len)
        end = start + n_rows
        q_visual = self.key_visual[q_start:q_start + n_rows]
        full = scores[0, self.heads]
        full_tiles = F.pad(full, (0, -key_length % ba.TILE), value=-torch.inf).unflatten(-1, (-1, ba.TILE))
        groups = tile_groups(full_tiles, q_visual, self.key_types)
        full_skip = margin[0, self.heads] < self.ln_lam
        allowed = self.key_positions[None, :] <= self.key_positions[q_start:q_start + n_rows, None]
        mask = ba.mask_chunk(self.mask, start, end, self.query_len, key_length)
        phase_data = self.phases.setdefault(phase, {})
        self._accumulate(phase_data, "full", gap_histograms(full_tiles, q_visual, self.key_types))
        for dim in self.dimensions:
            approx = self.q_short[:, :, start:end, :dim] @ self.k_short[..., :dim].transpose(-1, -2)
            approx = (approx * (self.scaling * self.width / dim)).float()
            tiles = ba.score_tiles(approx, allowed, mask)[0]
            projected_skip = ba.tile_margins(tiles) < self.ln_lam
            self._accumulate(phase_data, dim, gap_histograms(tiles, q_visual, self.key_types))
            self._accumulate(phase_data, ("decisions", dim), decision_counts(groups, full_skip, projected_skip))

    @staticmethod
    def _accumulate(phase_data, key, values):
        if key in phase_data:
            phase_data[key] += values
        else:
            phase_data[key] = values

    def summary(self, metadata):
        phases = {}
        for phase, data in self.phases.items():
            projections = {}
            for dim in self.dimensions:
                decisions = {}
                for group, values in zip(TILE_GROUPS, data["decisions", dim].tolist()):
                    counts = dict(zip(COUNTS, values))
                    counts.update(pairs_observed=int(counts["tiles"] > 0),
                                  pairs_with_false_skips=int(counts["false_skips"] > 0))
                    decisions[group] = counts
                projections[str(dim)] = {
                    "gaps": dict(zip(REGIONS, data[dim].tolist())), "decisions": decisions,
                }
            phases[phase] = {"full_gaps": dict(zip(REGIONS, data["full"].tolist())), "projections": projections}
        return {
            "schema": SCHEMA, "model": metadata["model"], "synthetic": metadata["synthetic"],
            "head_width": self.width, "dimensions": list(self.dimensions), "lambda": self.lam,
            "seed": self.seed, "histogram": {"bin_width": 1, "overflow_from": OVERFLOW},
            "layers": sorted(self.layers), "heads": self.heads, "pairs_completed": 1,
            "phases": phases,
        }


def merge_sweeps(total, sample):
    """Pool histogram and decision counts across fresh prompts on identical settings."""
    if total is None:
        return deepcopy(sample)
    identity = ("model", "synthetic", "head_width", "dimensions", "lambda", "seed", "histogram", "layers", "heads")
    if any(total[key] != sample[key] for key in identity):
        raise ValueError("JL sweep settings must match before pooling")
    for phase, sample_phase in sample["phases"].items():
        if phase not in total["phases"]:
            total["phases"][phase] = deepcopy(sample_phase)
            continue
        merged = total["phases"][phase]
        for region in REGIONS:
            merged["full_gaps"][region] = _add_bins(merged["full_gaps"][region], sample_phase["full_gaps"][region])
        for dim, sample_projection in sample_phase["projections"].items():
            projection = merged["projections"][dim]
            for region in REGIONS:
                projection["gaps"][region] = _add_bins(projection["gaps"][region], sample_projection["gaps"][region])
            for group, counts in sample_projection["decisions"].items():
                for key, value in counts.items():
                    projection["decisions"][group][key] += value
    total["pairs_completed"] += sample["pairs_completed"]
    return total


def _add_bins(left, right):
    return [a + b for a, b in zip(left, right, strict=True)]
