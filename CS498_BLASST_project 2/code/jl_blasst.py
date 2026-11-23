"""Tile-wise JL precheck and a same-input audit against BLASST."""
from __future__ import annotations

import math
from collections import defaultdict

import torch

import blasst_attn as ba


class JLPrecheck:
    """A seeded random orthogonal sketch, shared by queries and keys."""

    def __init__(self, dim: int, *, epsilon: float | None = None, seed: int = 0):
        if dim < 1 or (epsilon is not None and (not math.isfinite(epsilon) or epsilon < 0)):
            raise ValueError("JL dimension must be positive and epsilon finite and nonnegative")
        self.dim = dim
        self.epsilon = epsilon
        self.seed = seed
        self.bases = {}

    def basis(self, width, device):
        if not self.dim < width:
            raise ValueError(f"JL dimension {self.dim} must be smaller than head dimension {width}")
        identity = (width, device)
        if identity not in self.bases:
            rng = torch.Generator().manual_seed(self.seed)
            samples = torch.randn(width, self.dim, generator=rng, dtype=torch.float64)
            self.bases[identity] = torch.linalg.qr(samples).Q.to(device)
        return self.bases[identity]

    @torch.no_grad()
    def forward(self, q, k, v, *, lam, tile=128, scaling=None, mask=None,
                visual=None, audit=None, usage=None):
        """Run one attention head on [sequence, head_dim] tensors, queries at the KV tail.

        Audit computes exact logits for early skips as well, solely for comparison.
        Mask entries follow SDPA (boolean keep or additive logit bias).
        """
        q_len, width = q.shape
        k_len = k.shape[0]
        if not 0 < lam <= 1 or not 0 < q_len <= k_len or tile < 1:
            raise ValueError("Expected 0 < lambda <= 1, 0 < queries <= keys, and a positive tile size")
        if scaling is None:
            scaling = 1 / math.sqrt(width)
        basis = self.basis(width, q.device)
        q64, k64 = q.double(), k.double()
        q_short, k_short = q64 @ basis, k64 @ basis
        q_norm, k_norm = q64.norm(dim=-1), k64.norm(dim=-1)
        if self.epsilon is None:
            # Explicit residuals avoid subtracting nearly equal squared norms.
            q_rest = (q64 - q_short @ basis.T).norm(dim=-1)
            k_rest = (k64 - k_short @ basis.T).norm(dim=-1)
        roundoff = 4 * width * torch.finfo(torch.float32).eps
        q_offset, ln_lam = k_len - q_len, math.log(lam)
        out = v.new_zeros((q_len, v.shape[-1]))
        for start in range(0, q_len, tile):
            end = min(start + tile, q_len)
            q_positions = torch.arange(start, end, device=q.device) + q_offset
            q_slice = slice(start, end)
            q_block = q[q_slice].float()
            bar = q_block.new_full((end - start,), -torch.inf)
            baseline_bar = bar
            kept_scores, kept_values = [], []
            for key_start in range(0, end + q_offset, tile):
                key_end = min(key_start + tile, k_len)
                keys = slice(key_start, key_end)
                allowed = q_positions[:, None] >= torch.arange(key_start, key_end, device=q.device)
                bias = 0.0
                if mask is not None:
                    block_mask = mask[..., keys]
                    if mask.shape[-2] != 1:
                        block_mask = block_mask[q_slice]
                    if mask.dtype == torch.bool:
                        allowed = allowed & block_mask
                    else:
                        bias = block_mask.float()
                        allowed = allowed & torch.isfinite(bias)
                if not bool(allowed.any()):
                    continue
                product_norm = q_norm[q_slice, None] * k_norm[None, keys]
                short_dot = q_short[q_slice] @ k_short[keys].T
                estimate = short_dot * (scaling * width / self.dim) + bias
                if self.epsilon is None:
                    upper = short_dot * scaling + q_rest[q_slice, None] * k_rest[None, keys] * abs(scaling) + bias
                else:
                    upper = estimate + self.epsilon * abs(scaling) * product_norm
                # Allow for FP32 dot-product, scaling and bias rounding in the fallback.
                upper = upper + roundoff * (abs(scaling) * product_norm + abs(bias))
                upper_max = upper.masked_fill(~allowed, -torch.inf).amax(-1).to(bar.dtype)
                early = bool(((~allowed.any(-1)) | (upper_max - bar < ln_lam)).all())

                if not early or audit is not None:
                    exact = ((q_block @ k[keys].float().T) * scaling + bias).masked_fill(~allowed, -torch.inf)
                    exact_max = exact.amax(-1)
                skip = early
                if not early:
                    bar = torch.maximum(bar, exact_max)
                    skip = _skip(exact_max, bar, ln_lam)
                    if not skip:
                        kept_scores.append(exact)
                        kept_values.append(v[keys])
                if usage is not None:
                    usage["tiles"] += 1
                    usage["early_skips"] += int(early)
                    usage["fallbacks"] += int(not early)
                    usage["skips"] += int(skip)
                if audit is not None:
                    baseline_bar = torch.maximum(baseline_bar, exact_max)
                    baseline_skip = _skip(exact_max, baseline_bar, ln_lam)
                    audit.record(visual[q_positions], visual[keys], q_norm[q_slice], k_norm[keys],
                                 allowed, estimate, upper, exact, early, skip, baseline_skip)
            if kept_scores:
                scores = torch.cat(kept_scores, dim=-1)
                probs = torch.softmax(scores, dim=-1).nan_to_num()
                out[q_slice] = probs.to(v.dtype) @ torch.cat(kept_values)
        return out


def _skip(exact_max, bar, ln_lam):
    margin = torch.where(torch.isfinite(bar), exact_max - bar, -torch.inf)
    return bool((margin < ln_lam).all())


class JLAudit:
    """Counts head-level tile decisions and token-pair errors by modality."""

    def __init__(self):
        self.tiles = defaultdict(lambda: defaultdict(int))
        self.pairs = defaultdict(lambda: defaultdict(float))

    def record(self, q_visual, k_visual, q_norm, k_norm, allowed, estimate, upper, exact,
               early, skip, baseline_skip):
        q_used = q_visual[allowed.any(-1)]
        k_used = k_visual[allowed.any(-2)]
        q_kind = _tile_kind(q_used)
        k_kind = _tile_kind(k_used)
        for group in ("all", f"key:{k_kind}", f"{q_kind}->{k_kind}"):
            counts = self.tiles[group]
            counts["tiles"] += 1
            counts["early_skips"] += int(early)
            counts["fallbacks"] += int(not early)
            counts["baseline_skips"] += int(baseline_skip)
            counts["false_early_skips"] += int(early and not baseline_skip)
            counts["correct_early_skips"] += int(early and baseline_skip)
            counts["skip_mask_disagreements"] += int(skip != baseline_skip)
        for qv, q_name in ((False, "T"), (True, "V")):
            for kv, k_name in ((False, "T"), (True, "V")):
                selected = allowed & (q_visual[:, None] == qv) & (k_visual[None, :] == kv)
                rows, cols = selected.nonzero(as_tuple=True)
                if not rows.numel():
                    continue
                result = self.pairs[f"{q_name}->{k_name}"]
                error = (estimate[rows, cols] - exact[rows, cols]).abs()
                result["pairs"] += rows.numel()
                result["abs_error"] += error.sum().item()
                result["q_norm"] += q_norm[rows].sum().item()
                result["k_norm"] += k_norm[cols].sum().item()
                result["upper_violations"] += (upper[rows, cols] < exact[rows, cols]).sum().item()

    def summary(self):
        tiles = {}
        for name, counts in self.tiles.items():
            baseline = counts["baseline_skips"]
            tiles[name] = dict(counts, early_skip_rate=counts["early_skips"] / counts["tiles"],
                               fallback_rate=counts["fallbacks"] / counts["tiles"],
                               baseline_skips_caught=counts["correct_early_skips"] / baseline
                               if baseline else None)
        pairs = {}
        for name, stats in self.pairs.items():
            count = int(stats["pairs"])
            pairs[name] = {
                "pairs": count,
                "mean_abs_error": stats["abs_error"] / count,
                "mean_query_norm": stats["q_norm"] / count,
                "mean_key_norm": stats["k_norm"] / count,
                "upper_violations": int(stats["upper_violations"]),
            }
        return {"tile_decisions": tiles, "token_pair_errors": pairs}


def _tile_kind(visual):
    count = int(visual.sum())
    if count == 0:
        return "T"
    return "V" if count == visual.numel() else "mixed"


def _tile_spans(visual, split):
    """Split each original tile internally, preserving position and traversal order."""
    spans = []
    for start in range(0, len(visual), ba.TILE):
        end = min(start + ba.TILE, len(visual))
        cuts = [start]
        if split:
            changes = visual[start + 1:end] != visual[start:end - 1]
            cuts.extend(start + 1 + i for i in changes.nonzero().flatten().tolist())
        cuts.append(end)
        spans.extend(zip(cuts, cuts[1:]))
    return spans


def _split_margins(scores, query_spans, key_spans):
    """Reduce real lanes only, equivalent to padding each score tile with -inf."""
    maxima = torch.stack([scores[..., lo:hi].amax(-1) for lo, hi in key_spans], dim=-1)
    margins = ba.row_tile_margins(maxima)
    return torch.stack([margins[..., lo:hi, :].amax(-2) for lo, hi in query_spans], dim=-2)


def _record_mask_decision(groups, q_kind, k_kind, full_skip, projected_skip, valid_pairs, dense_weight):
    false_skip = projected_skip and not full_skip
    for name in ("all", f"key:{k_kind}", f"{q_kind}->{k_kind}"):
        counts = groups[name]
        counts["tiles"] += 1
        counts["valid_pairs"] += valid_pairs
        counts["false_skip_valid_pairs"] += valid_pairs if false_skip else 0
        counts["dense_weight"] += dense_weight
        counts["full_skipped_dense_weight"] += dense_weight if full_skip else 0.0
        counts["projected_skipped_dense_weight"] += dense_weight if projected_skip else 0.0
        counts["false_skip_dense_weight"] += dense_weight if false_skip else 0.0
        counts["full_skips"] += int(full_skip)
        counts["projected_skips"] += int(projected_skip)
        counts["matched_skips"] += int(full_skip and projected_skip)
        counts["false_skips"] += int(false_skip)
        counts["missed_skips"] += int(full_skip and not projected_skip)


class ProjectedJLStudy:
    """Build a skip mask from projected logits alone and compare it to full BLASST."""

    def __init__(self, probe, layers, heads, lam, *, split=None, visual=None):
        self.probe, self.layers = probe, layers
        self.heads = sorted(heads)
        self.lam = lam
        self.ln_lam = torch.tensor(math.log(lam), dtype=torch.float32).item()
        self.margins = {}
        self.phases = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.split, self.visual = split, visual
        self.split_chunks = {}
        self.split_prefill = defaultdict(lambda: defaultdict(int))

    @torch.no_grad()
    def observe(self, *, layer, phase, query, key, value, attention_mask, scaling):
        if layer not in self.layers:
            return
        query_len, width = query.shape[-2:]
        key_len = key.shape[-2]
        groups = query.shape[1] // key.shape[1]
        kv_heads = [head // groups for head in self.heads]
        basis = self.probe.basis(width, query.device)
        q_short = query[:, self.heads].double() @ basis
        k_short = key[:, kv_heads].double() @ basis
        if attention_mask is not None and attention_mask.shape[1] != 1:
            attention_mask = attention_mask[:, self.heads]
        offset = key_len - query_len
        key_positions = torch.arange(key_len, device=query.device)
        self.margins.clear()
        self.split_chunks.clear()
        split_prefill = self.split is not None and phase == "prefill"
        if split_prefill:
            visual = torch.zeros(key_len, dtype=torch.bool)
            visual[:min(len(self.visual), key_len)] = self.visual[:key_len]
            key_spans = _tile_spans(visual, self.split in ("key", "both"))
        for start in range(0, query_len, ba.TILE):
            end = min(start + ba.TILE, query_len)
            q_pos = torch.arange(start + offset, end + offset, device=query.device)
            allowed = key_positions[None, :] <= q_pos[:, None]
            mask = ba.mask_chunk(attention_mask, start, end, query_len, key_len)
            logits = (q_short[:, :, start:end] @ k_short.transpose(-1, -2))
            logits = (logits * (scaling * width / self.probe.dim)).float()
            tiles = ba.score_tiles(logits, allowed, mask)
            q_start = start + offset
            self.margins[q_start] = ba.tile_margins(tiles)[0].cpu()
            if split_prefill:
                query_spans = _tile_spans(visual[q_start:end + offset], self.split in ("query", "both"))
                split_margins = _split_margins(
                    tiles.flatten(-2)[..., :key_len], query_spans, key_spans,
                )[0].cpu()
                self.split_chunks[q_start] = (query_spans, key_spans, split_margins)

    def record(self, *, phase, head_index, q_start, tile_index, full_margin, q_visual, k_visual,
               valid_pairs, dense_weight):
        projected_margin = float(self.margins[q_start][head_index, tile_index])
        projected_skip = projected_margin < self.ln_lam
        full_skip = full_margin < self.ln_lam
        q_kind, k_kind = _tile_kind(q_visual), _tile_kind(k_visual)
        _record_mask_decision(self.phases[phase], q_kind, k_kind, full_skip, projected_skip,
                              valid_pairs, dense_weight)
        return {
            "projected_margin": projected_margin,
            "projected_skipped": projected_skip,
            "full_skipped_at_jl_lambda": full_skip,
        }

    def record_split(self, head_index, q_start, scores, probabilities, visual):
        query_spans, key_spans, projected = self.split_chunks[q_start]
        full = _split_margins(scores, query_spans, key_spans)
        valid = torch.isfinite(scores)
        measurements = []
        for qi, (qlo, qhi) in enumerate(query_spans):
            for ki, (klo, khi) in enumerate(key_spans):
                block = valid[qlo:qhi, klo:khi]
                count = int(block.sum())
                if count == 0:
                    continue
                q_kind = _tile_kind(visual[q_start + qlo:q_start + qhi][block.any(-1)])
                k_kind = _tile_kind(visual[klo:khi][block.any(-2)])
                full_margin, projected_margin = float(full[qi, ki]), float(projected[head_index, qi, ki])
                full_skip, projected_skip = full_margin < self.ln_lam, projected_margin < self.ln_lam
                weight = float(probabilities[qlo:qhi, klo:khi].sum())
                _record_mask_decision(self.split_prefill, q_kind, k_kind, full_skip, projected_skip,
                                      count, weight)
                measurements.append({
                    "q": q_start + qlo, "k": klo, "rows": qhi - qlo, "cols": khi - klo,
                    "q_kind": q_kind, "k_kind": k_kind, "margin": full_margin,
                    "mass": weight / (qhi - qlo),
                    "projected_margin": projected_margin, "projected_skipped": projected_skip,
                    "full_skipped_at_jl_lambda": full_skip,
                })
        return measurements

    def summary(self):
        summary = {
            "dimension": self.probe.dim,
            "seed": self.probe.seed,
            "lambda": self.lam,
            "phases": {phase: {name: dict(counts) for name, counts in groups.items()}
                       for phase, groups in self.phases.items()},
        }
        if self.split is not None:
            summary["split"] = self.split
            summary["split_prefill"] = {name: dict(counts) for name, counts in self.split_prefill.items()}
        return summary


class JLStudy:
    """Audit selected model inputs or replace attention in the selected layers."""

    def __init__(self, probe, visual, layers, heads, lam, tile, *, live=False):
        self.probe, self.visual = probe, visual
        self.layers, self.heads = layers, heads
        self.lam, self.tile = lam, tile
        self.live = live
        self.phases = {}

    def observe(self, *, layer, phase, query, key, value, attention_mask, scaling):
        if layer not in self.layers:
            return
        key_len = key.shape[-2]
        if self.live:
            usage = self.phases.setdefault(phase, defaultdict(int))
            heads, audit, visual = range(query.shape[1]), None, None
        else:
            audit = self.phases.setdefault(phase, JLAudit())
            heads, usage = sorted(self.heads), None
            visual = torch.zeros(key_len, device=query.device, dtype=torch.bool)
            visual[:min(key_len, len(self.visual))] = self.visual[:key_len]
        outputs = []
        groups = query.shape[1] // key.shape[1]
        for head in heads:
            kv_head = head // groups
            mask = None
            if attention_mask is not None:
                mask_head = 0 if attention_mask.shape[1] == 1 else head
                mask = attention_mask[0, mask_head, -query.shape[-2]:, :key_len]
            output = self.probe.forward(query[0, head], key[0, kv_head], value[0, kv_head],
                                        lam=self.lam, tile=self.tile, scaling=scaling, mask=mask,
                                        visual=visual, audit=audit, usage=usage)
            if self.live:
                outputs.append(output)
        if self.live:
            return torch.stack(outputs, dim=1).unsqueeze(0).contiguous()
        return None

    def summary(self):
        return {
            "dimension": self.probe.dim, "seed": self.probe.seed, "lambda": self.lam,
            "mode": "live" if self.live else "audit",
            "layers": sorted(self.layers), "heads": "all" if self.live else sorted(self.heads),
            "bound": "residual" if self.probe.epsilon is None else "jl_epsilon",
            "epsilon": self.probe.epsilon,
            "phases": {phase: dict(stats) if self.live else stats.summary()
                       for phase, stats in self.phases.items()},
        }
