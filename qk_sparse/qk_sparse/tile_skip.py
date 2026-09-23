"""Is a FlashAttention tile's score matrix determined by a few keys?

Different question from ``metrics.py``. That module asks *which entries matter
for the output*. This one asks whether the **skip decision** can be made
without computing the tile.

The setting is FlashAttention's tiling: a tile of ``P`` queries against a tile
of ``B`` keys produces ``S = Q_i K_j^T`` of shape ``[P, B]``. BLASST reduces
that to a row-wise max and skips the tile when the max sits far enough below
the row's running max -- but it has already paid the ``P x B x d`` matmul to
learn that. If a handful of the ``B`` keys predicted the row max, the decision
would cost a fraction of the matmul.

So the measurements here are:

1. **Headroom.** What fraction of tiles are skippable at a given gap? Nothing
   else matters if the answer is "almost none".
2. **Is the max concentrated?** How far above the rest of the tile does the top
   key sit, and do the ``P`` rows of the tile agree on *which* key it is?
3. **How few keys suffice?** Take the best ``r`` keys of the tile and use their
   max as a stand-in for the true row max. How much is lost, and how often does
   the skip decision change?

The subset max is always ``<=`` the true max, so an underestimate can only
cause a tile to be **skipped when it should have been kept** -- never the
reverse. That asymmetry is why the false-skip rate is the number a kernel
cares about, and why it is reported separately from raw error.

The oracle-chosen subset is the *upper bound* on what any cheap probe could
achieve: it already knows which keys are best. If even the oracle needs 32 of
64 keys, no practical estimator will do better and the idea is dead. That is
what makes this the right first step.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

_DTYPE = torch.float32


def analyse_tiles(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    query_tile: int = 128,
    key_tile: int = 64,
    gaps: Sequence[float] = (4.0, 6.0, 8.0),
    subset_sizes: Sequence[int] = (1, 2, 4, 8, 16),
    max_query_tiles: int = 8,
    max_key_tiles: int = 64,
    scale: Optional[float] = None,
    seed: int = 0,
) -> Optional[Dict[str, float]]:
    """Tile-level skip statistics for one (layer, head).

    ``gaps`` are in the same units BLASST uses: a tile is worth keeping for a
    row when ``tile_max >= row_global_max - gap``, since ``exp(-gap)`` bounds
    its possible contribution to the softmax.
    """
    n, d = q.shape
    s_len = k.shape[0]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    q = q.to(_DTYPE)
    k = k.to(_DTYPE)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Query tiles, spread across the sequence.
    starts: List[int] = []
    if causal:
        first = max(query_tile, (n // 3) // query_tile * query_tile)
        starts = list(range(first, n - query_tile + 1, query_tile))
    else:
        starts = list(range(0, n - query_tile + 1, query_tile))
    if not starts:
        return None
    if len(starts) > max_query_tiles:
        step = len(starts) / max_query_tiles
        starts = [starts[int(i * step)] for i in range(max_query_tiles)]

    acc: Dict[str, List[float]] = {}

    def add(key: str, value: float) -> None:
        acc.setdefault(key, []).append(float(value))

    for st in starts:
        n_allowed = st if causal else s_len
        if n_allowed < 4 * key_tile:
            continue
        qb = q[st : st + query_tile]
        scores = (qb @ k[:n_allowed].T) * scale  # [P, n_allowed]
        row_global_max = scores.max(-1).values  # [P]

        # Key tiles, sampled if there are many.
        n_key_tiles = n_allowed // key_tile
        tile_ids = list(range(n_key_tiles))
        if len(tile_ids) > max_key_tiles:
            step = len(tile_ids) / max_key_tiles
            tile_ids = [tile_ids[int(i * step)] for i in range(max_key_tiles)]

        for tid in tile_ids:
            tile = scores[:, tid * key_tile : (tid + 1) * key_tile]  # [P, B]
            if tile.shape[1] < key_tile:
                continue
            row_max = tile.max(-1).values  # [P]
            deficit = row_global_max - row_max  # >= 0; large = skippable

            # 1. Headroom: how often is this tile negligible for a row?
            for gap in gaps:
                add(f"skippable@{gap:g}", float((deficit > gap).to(_DTYPE).mean()))

            # 2. Is the top key dominant, and do rows agree on which it is?
            sorted_tile = tile.sort(dim=-1, descending=True).values
            spread = (sorted_tile[:, 0] - sorted_tile[:, -1]).clamp_min(1e-6)
            add("top1_margin", float(((sorted_tile[:, 0] - sorted_tile[:, 1]) / spread).mean()))
            argmax_ids = tile.argmax(-1)
            add("argmax_distinct_frac", float(argmax_ids.unique().numel() / tile.shape[0]))

            # 3. How few keys reproduce the row max?
            #    Oracle subset: the columns with the highest max over the tile's
            #    rows -- the best any probe could pick. Random subset: what an
            #    uninformed probe gets. The truth for a real estimator lies
            #    between them.
            col_rank = tile.max(0).values.argsort(descending=True)
            for r in subset_sizes:
                if r >= key_tile:
                    continue
                oracle_cols = col_rank[:r]
                rnd_cols = torch.randperm(key_tile, generator=generator)[:r]
                for name, cols in (("oracle", oracle_cols), ("random", rnd_cols)):
                    approx = tile[:, cols].max(-1).values  # <= row_max always
                    err = row_max - approx
                    add(f"{name}_r{r}/max_err", float(err.mean()))
                    add(f"{name}_r{r}/exact_frac", float((err < 1e-4).to(_DTYPE).mean()))
                    # A tile kept under the true max but skipped under the
                    # estimate is the error that silently drops attention.
                    for gap in gaps:
                        keep_true = deficit <= gap
                        keep_est = (row_global_max - approx) <= gap
                        false_skip = keep_true & ~keep_est
                        denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                        add(
                            f"{name}_r{r}/false_skip@{gap:g}",
                            float(false_skip.to(_DTYPE).sum() / denom),
                        )

            # 3b. Probes a kernel could actually build.
            #
            # The oracle above chooses keys per (query tile, key tile) pair,
            # which no kernel can do -- by the time you know which keys score
            # highest you have done the matmul. These are query-independent:
            # computed once per key tile, reused for every query tile.
            k_tile = k[:n_allowed][tid * key_tile : (tid + 1) * key_tile]
            k_mean = k_tile.mean(0)
            radius = (k_tile - k_mean).norm(dim=-1).max()
            norms = k_tile.norm(dim=-1)

            for r in subset_sizes:
                if r >= key_tile:
                    continue
                # Largest-norm keys: precomputable, query-independent. A key
                # can only score high if it is long, so norm is the obvious
                # query-free proxy for "might be the max".
                cols = norms.argsort(descending=True)[:r]
                approx = tile[:, cols].max(-1).values
                err = row_max - approx
                add(f"norm_r{r}/max_err", float(err.mean()))
                add(f"norm_r{r}/exact_frac", float((err < 1e-4).to(_DTYPE).mean()))
                for gap in gaps:
                    keep_true = deficit <= gap
                    keep_est = (row_global_max - approx) <= gap
                    denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                    add(f"norm_r{r}/false_skip@{gap:g}",
                        float((keep_true & ~keep_est).to(_DTYPE).sum() / denom))

            # Mean-key score: one dot product per query instead of B. Since
            # q.k_mean is the *average* score over the tile it never exceeds the
            # max, so like the subsets it can only cause false skips.
            mean_score = (qb @ k_mean) * scale
            err = row_max - mean_score
            add("meankey/max_err", float(err.mean()))
            for gap in gaps:
                keep_true = deficit <= gap
                keep_est = (row_global_max - mean_score) <= gap
                denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                add(f"meankey/false_skip@{gap:g}",
                    float((keep_true & ~keep_est).to(_DTYPE).sum() / denom))

            # Provable upper bound: max_b q.k_b <= q.k_mean + ||q||*radius.
            # It can never drop a tile that mattered, so the question is not
            # accuracy but whether it is tight enough to skip anything at all.
            bound = mean_score + qb.norm(dim=-1) * radius * scale
            add("bound/slack", float((bound - row_max).mean()))
            for gap in gaps:
                skip_true = deficit > gap
                skip_est = (row_global_max - bound) > gap
                add(f"bound/false_skip@{gap:g}", float(
                    (skip_est & ~skip_true).to(_DTYPE).sum()
                    / (~skip_true).to(_DTYPE).sum().clamp_min(1.0)))
                # Of the tiles genuinely skippable, how many does it catch?
                add(f"bound/skip_recall@{gap:g}", float(
                    (skip_est & skip_true).to(_DTYPE).sum()
                    / skip_true.to(_DTYPE).sum().clamp_min(1.0)))

            # 3c. Two probes that cut a different axis.
            #
            # Box bound: the ball bound above treats the key tile as a sphere
            # of radius max||k - k_mean||, which is loose because key tiles are
            # not spherical. Bounding each dimension separately is strictly
            # tighter and costs the same -- d multiply-adds per query, i.e. one
            # key's worth of work instead of 64 -- and is still a true upper
            # bound, so it can never drop a tile that mattered.
            lo = k_tile.min(0).values
            hi = k_tile.max(0).values
            box = (torch.maximum(qb * lo, qb * hi).sum(-1)) * scale
            add("box/slack", float((box - row_max).mean()))
            for gap in gaps:
                skip_true = deficit > gap
                skip_est = (row_global_max - box) > gap
                add(f"box/false_skip@{gap:g}", float(
                    (skip_est & ~skip_true).to(_DTYPE).sum()
                    / (~skip_true).to(_DTYPE).sum().clamp_min(1.0)))
                add(f"box/skip_recall@{gap:g}", float(
                    (skip_est & skip_true).to(_DTYPE).sum()
                    / skip_true.to(_DTYPE).sum().clamp_min(1.0)))

            # Channel subset: instead of few keys x all dims, use few dims x
            # all keys. The largest-magnitude channels of q dominate the dot
            # product, so scoring every key on those alone costs c/d of the
            # matmul. Unlike a key subset this sees every key, so it cannot
            # miss the max by looking in the wrong place -- only by truncating
            # dimensions. It is not a bound, so it can err either way.
            for c in (8, 16, 32):
                if c >= d:
                    continue
                chan = qb.abs().mean(0).topk(c).indices
                approx = (qb[:, chan] @ k_tile[:, chan].T).max(-1).values * scale
                add(f"chan{c}/abs_err", float((approx - row_max).abs().mean()))
                for gap in gaps:
                    keep_true = deficit <= gap
                    keep_est = (row_global_max - approx) <= gap
                    denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                    add(f"chan{c}/false_skip@{gap:g}", float(
                        (keep_true & ~keep_est).to(_DTYPE).sum() / denom))

            # 3d. Downsample the QUERY side instead of the key side.
            #
            # Key subsetting fails because which keys win depends on the query.
            # But the rows of a query tile mostly agree on their argmax, so one
            # pooled query scored against *all* keys should carry the signal --
            # and it costs 1/P of the tile matmul (1/128) rather than r/B (1/4).
            q_mean = qb.mean(0)
            poolq = (q_mean @ k_tile.T) * scale  # [B], all keys, one query
            poolq_max = poolq.max()
            for gap in gaps:
                keep_true = deficit <= gap
                keep_est = (row_global_max - poolq_max) <= gap
                denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                add(f"poolq/false_skip@{gap:g}",
                    float((keep_true & ~keep_est).to(_DTYPE).sum() / denom))
                # Wasted work rather than lost accuracy, but it eats the saving.
                denom_k = (~keep_true).to(_DTYPE).sum().clamp_min(1.0)
                add(f"poolq/false_keep@{gap:g}",
                    float(((~keep_true) & keep_est).to(_DTYPE).sum() / denom_k))
            add("poolq/max_err", float((row_max - poolq_max).abs().mean()))

            # Two-stage: the pooled query nominates the top-r keys, then those
            # r keys are scored exactly for every row. Cost 1/P + r/B. This is
            # the shape a real kernel would take.
            for r in subset_sizes:
                if r >= key_tile:
                    continue
                cols = poolq.topk(r).indices
                approx = tile[:, cols].max(-1).values
                add(f"poolq_top{r}/max_err", float((row_max - approx).mean()))
                add(f"poolq_top{r}/exact_frac",
                    float(((row_max - approx) < 1e-4).to(_DTYPE).mean()))
                for gap in gaps:
                    keep_true = deficit <= gap
                    keep_est = (row_global_max - approx) <= gap
                    denom = keep_true.to(_DTYPE).sum().clamp_min(1.0)
                    add(f"poolq_top{r}/false_skip@{gap:g}",
                        float((keep_true & ~keep_est).to(_DTYPE).sum() / denom))

            # 4. How much of the tile's softmax mass sits in its top-r columns?
            tile_soft = torch.softmax(tile, dim=-1)
            col_mass = tile_soft.mean(0)
            ordered = col_mass.sort(descending=True).values
            for r in subset_sizes:
                if r < key_tile:
                    add(f"mass_top{r}", float(ordered[:r].sum()))

    if not acc:
        return None
    return {key: sum(vals) / len(vals) for key, vals in acc.items()}
