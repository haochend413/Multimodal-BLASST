"""Metrics that test for vertical-vector sparsity in attention.

The central claim under test (VecAttention, arXiv 2603.29494) is that
neighbouring queries share their important keys, so the useful entries of the
attention map form short vertical segments: one key column spanning a
contiguous group of query rows.

Everything here is computed from ``Q`` and ``K`` alone (plus the softmax they
induce).  No model-specific code, no kernels, no imports from VecAttention.

Vocabulary
----------
block
    ``P_q`` contiguous query rows -- the unit over which a shared key set is
    chosen.  VecAttention uses 64 or 128.
allowed keys
    The key positions a block is scored against.  For bidirectional models this
    is every key.  For causal models we use only keys strictly *before* the
    block, so that every row in the block shares one allowed set and all
    patterns are measured on the same ground.  The near-diagonal region is
    excluded on purpose: it is always kept dense by real kernels, so including
    it would flatter every pattern equally.
density
    Fraction of allowed (query, key) entries a pattern is allowed to keep.  All
    patterns are compared at the *same* density, which is the only way the
    comparison means anything.
recovered mass
    Fraction of each row's softmax mass that lands inside the kept entries,
    averaged over the rows of the block.  1.0 means the pattern kept everything
    that mattered.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch

# Selection is "which keys matter", a ranking question, so it is done in float32
# regardless of the model's compute dtype -- bf16 ties would otherwise turn into
# arbitrary tie-breaks and show up as spurious disagreement between rows.
_DTYPE = torch.float32


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def topk_mask(values: torch.Tensor, m: int) -> torch.Tensor:
    """Boolean mask of the ``m`` largest entries along the last dim."""
    m = max(1, min(m, values.shape[-1]))
    idx = values.topk(m, dim=-1).indices
    mask = torch.zeros_like(values, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def jaccard(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Jaccard overlap between two boolean masks, over the last dim."""
    inter = (a & b).sum(-1).to(_DTYPE)
    union = (a | b).sum(-1).to(_DTYPE).clamp_min(1.0)
    return inter / union


def chance_jaccard(m: int, n: int) -> float:
    """Expected Jaccard of two independent uniform top-``m`` sets out of ``n``.

    Two sets of size ``m`` drawn independently overlap by ``m^2/n`` on average,
    so J = inter / (2m - inter).  This is the null hypothesis every overlap
    number below has to beat; without it a "high" overlap at large density is
    meaningless.
    """
    if n <= 0 or m <= 0:
        return 0.0
    inter = (m * m) / n
    union = 2 * m - inter
    return float(inter / union) if union > 0 else 0.0


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(a.to(_DTYPE), b.to(_DTYPE), dim=-1)


# --------------------------------------------------------------------------
# pattern families, all evaluated at one matched density
# --------------------------------------------------------------------------


def _block_means(col_values: torch.Tensor, block: int) -> torch.Tensor:
    """Mean of ``col_values`` over contiguous key blocks (last dim)."""
    n = col_values.shape[-1]
    pad = (-n) % block
    if pad:
        col_values = torch.nn.functional.pad(col_values, (0, pad))
    return col_values.unflatten(-1, (-1, block)).mean(-1)


def _expand_block_mask(block_mask: torch.Tensor, block: int, n: int) -> torch.Tensor:
    return block_mask.repeat_interleave(block, dim=-1)[..., :n]


def pattern_masses(
    attn: torch.Tensor,
    m: int,
    key_block: int,
    global_col_rank: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Recovered mass for each pattern family at a budget of ``m`` keys/row.

    ``attn`` is ``[P, S]``, already restricted to the allowed keys and
    renormalised so each row sums to 1.  Every pattern is given the same entry
    budget ``P * m``; they differ only in the *shape* they must spend it in.

    Returned keys, coarse to fine:

    ``line``
        One key set for the whole head (all blocks) -- MInference-style full
        vertical lines.  Needs ``global_col_rank``.
    ``block``
        Query block x key block, the XAttention / block-sparse unit.
    ``vertical``
        Query block x single key: the vertical vector, chosen by an oracle that
        already knows the true attention.  This is the *pattern's* ceiling, not
        what a selector achieves.
    ``horizontal``
        Single query x key block -- the transpose of ``vertical``, included as
        the control that tells you whether the vertical direction is doing the
        work or whether any fine-grained shape would do.
    ``oracle``
        Per-row top-m entries.  Unstructured upper bound.
    """
    p, s = attn.shape
    m = max(1, min(m, s))
    col_sum = attn.sum(0)  # [S]; mean over rows = col_sum / P

    out: Dict[str, float] = {}

    # Unstructured per-row ceiling.
    out["oracle"] = float(attn.topk(m, dim=-1).values.sum(-1).mean())

    # Vertical vectors: one shared key set per block, oracle-chosen.
    vert_idx = col_sum.topk(m).indices
    out["vertical"] = float(col_sum[vert_idx].sum() / p)

    # Key-blocks shared by the whole query block.
    #
    # A block-granular pattern cannot spend less than one whole key-block. When
    # the budget is smaller than that, rounding up to one block hands `block`
    # and `horizontal` several times the budget every other pattern gets -- 4.3x
    # for a 1500-key sequence at density 0.01 -- and they then "win" on stolen
    # budget. There is no honest way to shrink them, so they are omitted and the
    # report shows "-" rather than a number that is not comparable.
    nblk = m // key_block
    if nblk >= 1:
        blk_scores = _block_means(col_sum, key_block)
        blk_mask = topk_mask(blk_scores, nblk)
        col_mask = _expand_block_mask(blk_mask, key_block, s)
        out["block"] = float((col_sum * col_mask).sum() / p)

        # Per-row key-blocks (fine in the query direction, coarse in the key one).
        row_blk = _block_means(attn, key_block)  # [P, S/key_block]
        row_mask = _expand_block_mask(topk_mask(row_blk, nblk), key_block, s)
        out["horizontal"] = float((attn * row_mask).sum(-1).mean())

    # One key set for the entire head.
    if global_col_rank is not None:
        line_idx = global_col_rank[:m]
        out["line"] = float(col_sum[line_idx].sum() / p)

    return out


def output_errors(
    attn: torch.Tensor,
    value: torch.Tensor,
    masks: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Relative L2 error in the layer's *output* under each pattern.

    Recovered mass is a proxy, and a loose one: 0.73 of the mass says nothing
    about whether the 27% left behind carried the signal. What a kernel
    actually changes is ``O = A V``, so this computes what the row would emit
    if only the kept keys existed -- renormalising over them, exactly as a
    masked flash-attention kernel does:

        O_sparse[i] = sum_{j in S} A[i,j] V[j] / sum_{j in S} A[i,j]

    and reports ``||O_sparse - O_dense|| / ||O_dense||`` averaged over rows.
    Still not end-task accuracy, but it is the quantity that propagates.
    """
    dense = attn @ value  # [P, D]
    dense_norm = dense.norm(dim=-1).clamp_min(1e-9)
    out: Dict[str, float] = {}
    for name, mask in masks.items():
        kept = attn * mask
        denom = kept.sum(-1, keepdim=True).clamp_min(1e-9)
        sparse = (kept @ value) / denom
        out[name] = float(((sparse - dense).norm(dim=-1) / dense_norm).mean())
    return out


def pattern_masks(
    attn: torch.Tensor,
    m: int,
    key_block: int,
    selectors: Dict[str, torch.Tensor],
    global_col_rank: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """The boolean keep-masks behind ``pattern_masses``, for output scoring."""
    p, s = attn.shape
    m = max(1, min(m, s))
    col_sum = attn.sum(0)
    nblk = m // key_block

    masks: Dict[str, torch.Tensor] = {}
    masks["oracle"] = topk_mask(attn, m)
    shared = torch.zeros(s, dtype=torch.bool, device=attn.device)
    shared[col_sum.topk(m).indices] = True
    masks["vertical"] = shared.expand(p, s)
    if nblk >= 1:  # see pattern_masses: below one key-block they cannot be fair
        masks["block"] = _expand_block_mask(
            topk_mask(_block_means(col_sum, key_block), nblk), key_block, s
        ).expand(p, s)
        masks["horizontal"] = _expand_block_mask(
            topk_mask(_block_means(attn, key_block), nblk), key_block, s
        )
    if global_col_rank is not None:
        line = torch.zeros(s, dtype=torch.bool, device=attn.device)
        line[global_col_rank[:m]] = True
        masks["line"] = line.expand(p, s)
    for name, score in selectors.items():
        sel = torch.zeros(s, dtype=torch.bool, device=attn.device)
        sel[score.topk(max(1, min(m, score.shape[-1]))).indices] = True
        masks[name] = sel.expand(p, s)
    return masks


def selector_masses(
    attn: torch.Tensor,
    scores: torch.Tensor,
    selectors: Dict[str, torch.Tensor],
    m: int,
) -> Dict[str, float]:
    """Mass recovered when the key set is picked by a *cheap estimator*.

    ``pattern_masses['vertical']`` says how good vertical vectors could be;
    this says how much of that an implementable selector actually gets, since
    it ranks keys without having seen the true attention map.

    ``selectors`` maps a name to a ``[S]`` score vector over allowed keys.
    """
    p = attn.shape[0]
    col_sum = attn.sum(0)
    out = {}
    for name, score in selectors.items():
        idx = score.topk(max(1, min(m, score.shape[-1]))).indices
        out[name] = float(col_sum[idx].sum() / p)
    return out


def minp_density(scores: torch.Tensor, gap: float) -> float:
    """Fraction of keys a MinP rule keeps: ``s_j >= max_j s_j - gap``.

    This is VecAttention's actual selection rule (its ``threshold`` becomes
    ``gap = -log(threshold)``).  Unlike top-m it is content-adaptive: the number
    of keys kept varies per block, which is the property the fixed-budget
    comparisons above deliberately hold constant.
    """
    keep = scores >= (scores.max() - gap)
    return float(keep.to(_DTYPE).mean())


# --------------------------------------------------------------------------
# does the structure exist in Q/K geometry at all?
# --------------------------------------------------------------------------


def overlap_vs_distance(
    q_rows: torch.Tensor,
    keys: torch.Tensor,
    deltas: Sequence[int],
    densities: Sequence[float],
    scale: float,
) -> Dict[str, float]:
    """Top-key agreement between a row and the row ``delta`` positions later.

    ``q_rows`` is ``[1 + len(deltas), D]``: the anchor query followed by the
    queries at each offset.  All rows are scored against the *same* allowed
    keys, so the comparison is not contaminated by causal masking letting the
    later row see more keys.

    Returns ``jaccard@<delta>`` plus ``qcos@<delta>`` (cosine between the query
    vectors themselves).  Reading the two together separates the two possible
    causes: query similarity, versus a few keys being salient to everyone.
    """
    scores = (q_rows.to(_DTYPE) @ keys.to(_DTYPE).T) * scale  # [R, S]
    s = scores.shape[-1]
    anchor_q = q_rows[0]

    out: Dict[str, float] = {}
    # Query cosine is a property of the vectors, not of any budget.
    for i, d in enumerate(deltas, start=1):
        out[f"qcos@{d}"] = float(cosine(anchor_q, q_rows[i]))

    # Overlap depends strongly on how many keys the budget buys, so record it
    # at every density the run measures.  Fixing one density here was a bug:
    # a 43-token model reported "zero overlap" that was really "top-2 sets".
    for density in densities:
        m = max(1, int(round(density * s)))
        masks = topk_mask(scores, m)
        tag = f"d{density:g}"
        for i, d in enumerate(deltas, start=1):
            out[f"{tag}/jaccard@{d}"] = float(jaccard(masks[0], masks[i]))
        out[f"{tag}/chance_jaccard"] = chance_jaccard(m, s)
        out[f"{tag}/budget_keys"] = float(m)
    return out


def block_coherence(q_block: torch.Tensor) -> Dict[str, float]:
    """How alike are the queries inside one block?

    ``cos_to_mean`` is the quantity mean-pooling depends on: if it is near 1 the
    pooled query stands in for every row.  ``dispersion`` is the same thing
    scale-free, ``E||q - qbar||^2 / ||qbar||^2``, which is what enters the
    ranking-stability condition ``gamma > |delta . (k_a - k_b)|``.
    """
    q = q_block.to(_DTYPE)
    mean_q = q.mean(0, keepdim=True)
    return {
        "cos_to_mean": float(cosine(q, mean_q.expand_as(q)).mean()),
        "cos_adjacent": float(cosine(q[:-1], q[1:]).mean()) if q.shape[0] > 1 else 1.0,
        "dispersion": float(
            ((q - mean_q) ** 2).sum(-1).mean() / (mean_q**2).sum().clamp_min(1e-9)
        ),
    }


def concentration(attn: torch.Tensor) -> Dict[str, float]:
    """How peaked is the attention in the first place?

    ``eff_key_frac`` is exp(entropy) -- the effective number of keys a row
    actually uses -- divided by the number available.  This is the question that
    comes *before* "which pattern is best": if a head spreads its mass over half
    the sequence, no sparse pattern can help it, and a high ``vertical/oracle``
    ratio just means every selection is equally mediocre.

    PatchTST at 43 tokens is exactly that case, which is why it needs its own
    column rather than a verdict.
    """
    eps = 1e-12
    entropy = -(attn * (attn + eps).log()).sum(-1)  # [P]
    eff = entropy.exp()
    return {
        "attn_entropy": float(entropy.mean()),
        "eff_keys": float(eff.mean()),
        "eff_key_frac": float((eff / attn.shape[-1]).mean()),
    }


def margin(scores: torch.Tensor, m: int) -> float:
    """Normalised gap between the m-th and (m+1)-th ranked key.

    A block can have near-identical queries and still route unstably if many
    keys are nearly tied; this is the second half of the condition, and it is
    why ``cos_to_mean`` alone does not predict selector quality.
    """
    s = scores.shape[-1]
    m = max(1, min(m, s - 1))
    top = scores.topk(m + 1).values
    spread = (scores.max() - scores.min()).clamp_min(1e-9)
    return float((top[m - 1] - top[m]) / spread)


# --------------------------------------------------------------------------
# per-head driver
# --------------------------------------------------------------------------


def strided_group_masses(
    q: torch.Tensor,
    k: torch.Tensor,
    anchor: int,
    stride: int,
    group: int,
    densities: Sequence[float],
    scale: float,
    causal: bool = False,
) -> Dict[str, float]:
    """Retention when the query group is *strided* instead of contiguous.

    VecAttention's vertical vector requires the queries sharing a key set to be
    adjacent in sequence.  In a space-time token grid the queries that actually
    want the same keys may sit one frame apart -- 196 positions for a 14x14
    tubelet -- so a contiguous block is the wrong grouping and no amount of
    threshold tuning fixes it.

    This builds a group of the same size from rows ``anchor, anchor+stride,
    anchor+2*stride, ...`` and measures the identical shared-key retention, so
    the two groupings are directly comparable at equal cost.  A kernel cannot
    gather strided rows as cheaply as contiguous ones; the point is to find out
    whether it would be worth the trouble.
    """
    n = q.shape[0]
    rows = list(range(anchor, min(n, anchor + stride * group), stride))
    # A stride of 1560 in a 7800-token sequence yields 5 rows. Five queries
    # agree almost by default -- both groupings scored ~0.96 there, which says
    # nothing about grouping shape. Require enough rows for the comparison to
    # carry information.
    if len(rows) < 16:
        return {}

    # CRITICAL: a strided group runs out of sequence long before a contiguous
    # one does -- at stride 196 in a 1568-token video only 8 rows fit, against
    # 128 contiguous. Fewer queries trivially agree more, so comparing the two
    # directly measures group size, not grouping shape. The honest control is a
    # contiguous group of exactly the same number of rows.
    size = len(rows)
    contiguous_rows = list(range(anchor, min(n, anchor + size)))
    if len(contiguous_rows) < size:
        contiguous_rows = list(range(max(0, n - size), n))

    # Under causality both groups must be scored against the keys available to
    # their *earliest* row, or the strided group -- whose later rows see much
    # more history -- would win on access rather than on grouping shape.
    if causal:
        limit = min(rows[0], contiguous_rows[0])
        if limit < 8:
            return {}
        k = k[:limit]

    def _stats(indices: List[int]) -> Dict[str, torch.Tensor]:
        qb = q[indices]
        attn = torch.softmax((qb @ k.T) * scale, dim=-1)
        return {"attn": attn, "col_sum": attn.sum(0), "qb": qb, "p": len(indices)}

    strided = _stats(rows)
    matched = _stats(contiguous_rows)

    out: Dict[str, float] = {}
    for density in densities:
        m = max(1, int(round(density * k.shape[0])))
        tag = f"d{density:g}"
        for name, st in (("strided", strided), ("matched", matched)):
            col_sum, attn, p = st["col_sum"], st["attn"], st["p"]
            idx = col_sum.topk(min(m, col_sum.shape[-1])).indices
            out[f"{tag}/{name}_vertical"] = float(col_sum[idx].sum() / p)
            out[f"{tag}/{name}_oracle"] = float(
                attn.topk(m, dim=-1).values.sum(-1).mean()
            )
        pooled = (strided["qb"].mean(0) @ k.T) * scale
        pidx = pooled.topk(min(m, pooled.shape[-1])).indices
        out[f"{tag}/strided_pooled"] = float(
            strided["col_sum"][pidx].sum() / strided["p"]
        )
    qb = strided["qb"]
    out["strided_cos_to_mean"] = float(
        cosine(qb, qb.mean(0, keepdim=True).expand_as(qb)).mean()
    )
    out["stride"] = float(stride)
    out["group_rows"] = float(size)
    return out


def analyse_head(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    value: Optional[torch.Tensor] = None,
    causal: bool,
    query_block: int,
    key_block: int,
    densities: Sequence[float],
    deltas: Sequence[int],
    max_blocks: int,
    scale: Optional[float] = None,
    minp_gaps: Sequence[float] = (),
    group_strides: Sequence[int] = (),
) -> Optional[Dict[str, object]]:
    """Run every metric for one (layer, head).

    ``q`` is ``[N, D]`` and ``k`` is ``[S, D]``, post-RoPE, exactly as the model
    handed them to the attention kernel.

    Returns ``None`` when the sequence is too short to hold a block plus a
    non-trivial allowed key set -- short inputs cannot say anything about
    sparsity and silently averaging them in would bias every number upward.
    """
    n, d = q.shape
    s = k.shape[0]
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    q = q.to(_DTYPE)
    k = k.to(_DTYPE)
    v = value.to(_DTYPE) if value is not None else None

    # Which blocks to look at, and which keys each may attend to.
    starts: List[int] = []
    if causal:
        # Blocks need enough history in front of them for "past" to be a
        # meaningful region; start a third of the way in.
        first = max(query_block, (n // 3) // query_block * query_block)
        for st in range(first, n - query_block + 1, query_block):
            starts.append(st)
    else:
        for st in range(0, n - query_block + 1, query_block):
            starts.append(st)
    if not starts:
        return None
    if len(starts) > max_blocks:
        stride = len(starts) / max_blocks
        starts = [starts[int(i * stride)] for i in range(max_blocks)]

    per_block: List[Dict[str, object]] = []
    col_sums: List[torch.Tensor] = []
    key_counts: List[int] = []

    for st in starts:
        qb = q[st : st + query_block]  # [P, D]
        n_allowed = st if causal else s
        if n_allowed < 8 * key_block:
            continue
        kb = k[:n_allowed]

        scores = (qb @ kb.T) * scale  # [P, S0]
        attn = torch.softmax(scores, dim=-1)

        # Cheap selectors, all ranking keys without seeing `attn`.
        pooled = (qb.mean(0) @ kb.T) * scale
        centre = scores[qb.shape[0] // 2]
        stride_rows = scores[:: max(1, query_block // 4)]
        selectors = {
            "sel_pooled": pooled,  # VecAttention: mean-pooled query
            "sel_centre": centre,  # one representative row
            "sel_strided_max": stride_rows.max(0).values,  # union of 4 probes
            "sel_strided_mean": stride_rows.mean(0),
            "sel_random_row": scores[torch.randint(qb.shape[0], (1,)).item()],
        }

        entry = {"start": st, "n_allowed": n_allowed}
        entry.update(block_coherence(qb))
        entry.update(concentration(attn))

        for density in densities:
            m = max(1, int(round(density * n_allowed)))
            tag = f"d{density:g}"
            for name, val in pattern_masses(attn, m, key_block).items():
                entry[f"{tag}/{name}"] = val
            for name, val in selector_masses(attn, scores, selectors, m).items():
                entry[f"{tag}/{name}"] = val
            entry[f"{tag}/margin_pooled"] = margin(pooled, m)
            if v is not None:
                masks = pattern_masks(attn, m, key_block, selectors)
                for name, err in output_errors(attn, v[:n_allowed], masks).items():
                    entry[f"{tag}/err_{name}"] = err

        for gap in minp_gaps:
            entry[f"minp{gap:g}/density"] = minp_density(pooled, gap)

        per_block.append(entry)
        col_sums.append(attn.sum(0))
        key_counts.append(n_allowed)

    if not per_block:
        return None

    # Full vertical lines need one key ranking for the whole head, so they can
    # only be scored once every block has been seen.
    width = min(key_counts)
    stacked = torch.stack([c[:width] for c in col_sums])  # [B, width]
    global_rank = stacked.mean(0).argsort(descending=True)
    for entry, col_sum in zip(per_block, col_sums):
        for density in densities:
            m = max(1, int(round(density * width)))
            idx = global_rank[:m]
            entry[f"d{density:g}/line"] = float(col_sum[idx].sum() / query_block)

    # Overlap-vs-distance probes, anchored inside the sampled blocks.
    # Probe offsets are filtered per anchor, not globally: a large delta that
    # does not fit after a late block still fits after an early one, and
    # dropping it everywhere silently hid the frame-stride offsets that matter
    # most for video (d=196 for a 14x14 tubelet). Clamping to n-1 would be
    # worse -- several deltas would collapse onto the same row and report a
    # fake agreement of 1.0.
    probes: List[Dict[str, float]] = []
    for st in starts[: min(8, len(starts))]:
        anchor = st + query_block // 2
        fits = [d for d in deltas if anchor + d < n]
        if not fits:
            continue
        n_allowed = anchor if causal else s
        if n_allowed < 8 * key_block:
            continue
        rows = [anchor] + [anchor + d for d in fits]
        probes.append(
            overlap_vs_distance(q[rows], k[:n_allowed], fits, densities, scale)
        )

    # Contiguous vs grid-strided grouping, for models whose tokens carry a grid.
    strided: List[Dict[str, float]] = []
    if True:
        for stride in group_strides:
            for st in starts[: min(4, len(starts))]:
                res = strided_group_masses(
                    q, k, st, stride, query_block, densities, scale, causal=causal
                )
                if res:
                    strided.append(res)

    return {"blocks": per_block, "probes": probes, "strided": strided,
            "n_query": n, "n_key": s, "head_dim": d}
