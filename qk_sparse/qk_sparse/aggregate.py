"""Turn raw run records into the two tables that answer the question.

Table 1 -- pattern comparison.  At a fixed density, how much attention mass
does each pattern family recover?  The claim under test predicts

    line  <  block  <  vertical  <=  oracle

with ``vertical`` close to ``oracle`` and clearly above ``block``.  If
``vertical`` and ``block`` are within noise of each other, vertical vectors buy
nothing for that model.  ``horizontal`` is the control: it has the same
fine/coarse split as ``vertical`` but along the other axis, so
``vertical - horizontal`` isolates what is specific to sharing keys across
neighbouring queries.

Table 2 -- key-set agreement vs distance.  Top-key Jaccard between a query and
the query ``delta`` positions away, against the chance level for the same
density.  This is the mechanism: if it decays with distance and sits well above
chance at small distance, neighbouring queries really do share keys.

Everything is reported as a median across (sample, layer, head) with an
interquartile range, because head-level variation is large and a mean hides it.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PATTERNS = ["line", "block", "horizontal", "vertical", "oracle"]
SELECTORS = ["sel_pooled", "sel_centre", "sel_strided_max", "sel_strided_mean", "sel_random_row"]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def load_records(paths: Iterable[Path]) -> List[dict]:
    out = []
    for path in paths:
        if path.is_dir():
            out.extend(load_records(sorted(path.glob("*.json"))))
        else:
            out.append(json.loads(path.read_text()))
    return out


def iter_blocks(record: dict):
    """Yield (layer_index, head_index, block_dict) for every analysed block."""
    for sample in record.get("samples", []):
        for layer in sample.get("layers", []):
            for head in layer.get("heads", []):
                for block in head.get("blocks", []):
                    yield layer.get("depth", layer["call_index"]), head["head"], block


def iter_probes(record: dict):
    for sample in record.get("samples", []):
        for layer in sample.get("layers", []):
            for head in layer.get("heads", []):
                for probe in head.get("probes", []):
                    yield layer.get("depth", layer["call_index"]), head["head"], probe


def collect_pattern_stats(record: dict, density: float) -> Dict[str, Dict[str, float]]:
    tag = f"d{density:g}"
    buckets: Dict[str, List[float]] = defaultdict(list)
    for _, _, block in iter_blocks(record):
        for name in PATTERNS + SELECTORS:
            key = f"{tag}/{name}"
            if key in block:
                buckets[name].append(float(block[key]))
        for extra in ("cos_to_mean", "cos_adjacent", "dispersion",
                      "eff_key_frac", "eff_keys"):
            if extra in block:
                buckets[extra].append(float(block[extra]))
        mkey = f"{tag}/margin_pooled"
        if mkey in block:
            buckets["margin_pooled"].append(float(block[mkey]))
    return {
        name: {
            "median": median(vals),
            "p25": _quantile(vals, 0.25),
            "p75": _quantile(vals, 0.75),
            "n": len(vals),
        }
        for name, vals in buckets.items()
        if vals
    }


def collect_overlap_stats(record: dict, density: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """Overlap stats at one density.

    Probe keys are stored as ``d<density>/jaccard@<delta>``.  Result files
    written before that change stored a single untagged set, so those are read
    as-is rather than silently coming back empty.
    """
    tag = f"d{density:g}/" if density is not None else None
    buckets: Dict[str, List[float]] = defaultdict(list)
    for _, _, probe in iter_probes(record):
        for key, val in probe.items():
            if tag and key.startswith(tag):
                buckets[key[len(tag):]].append(float(val))
            elif "/" not in key:  # qcos@, or a pre-density-tag result file
                buckets[key].append(float(val))
    return {
        key: {"median": median(vals), "p25": _quantile(vals, 0.25),
              "p75": _quantile(vals, 0.75), "n": len(vals)}
        for key, vals in buckets.items()
        if vals
    }


def _fmt(stat: Optional[Dict[str, float]], digits: int = 3) -> str:
    if not stat:
        return "-"
    return f"{stat['median']:.{digits}f}"


def pattern_table(records: Sequence[dict], density: float) -> str:
    """Markdown table: recovered attention mass per pattern, per model."""
    head = (
        f"### Recovered attention mass at density {density:g}\n\n"
        "| model | modality | seq | eff keys | line | block | horizontal "
        "| **vertical** | oracle | vert-block | vert/oracle |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for rec in records:
        stats = collect_pattern_stats(rec, density)
        if not stats:
            continue
        seq = _sequence_length(rec)
        name = rec["adapter"]
        if rec.get("invalid_weights"):
            name += " ⚠️INVALID"
        if rec.get("synthetic_inputs"):
            name += " (synthetic input)"
        vert = stats.get("vertical", {}).get("median")
        blk = stats.get("block", {}).get("median")
        orc = stats.get("oracle", {}).get("median")
        gain = f"{vert - blk:+.3f}" if vert is not None and blk is not None else "-"
        ratio = f"{vert / orc:.3f}" if vert and orc else "-"
        rows.append(
            f"| {name} | {rec['modality']} | {seq} "
            f"| {_fmt(stats.get('eff_key_frac'), 2)} "
            f"| {_fmt(stats.get('line'))} | {_fmt(stats.get('block'))} "
            f"| {_fmt(stats.get('horizontal'))} | **{_fmt(stats.get('vertical'))}** "
            f"| {_fmt(stats.get('oracle'))} | {gain} | {ratio} |"
        )
    return head + "\n".join(rows) + "\n"


def selector_table(records: Sequence[dict], density: float) -> str:
    """How much of the vertical-vector ceiling a cheap selector reaches."""
    head = (
        f"### Selector quality at density {density:g} "
        "(mass recovered; `vertical` is the oracle ceiling)\n\n"
        "| model | vertical | pooled | centre row | strided max | strided mean "
        "| random row | pooled/ceiling | cos_to_mean |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for rec in records:
        s = collect_pattern_stats(rec, density)
        if not s:
            continue
        vert = s.get("vertical", {}).get("median")
        pooled = s.get("sel_pooled", {}).get("median")
        frac = f"{pooled / vert:.3f}" if vert and pooled else "-"
        rows.append(
            f"| {rec['adapter']} | {_fmt(s.get('vertical'))} | {_fmt(s.get('sel_pooled'))} "
            f"| {_fmt(s.get('sel_centre'))} | {_fmt(s.get('sel_strided_max'))} "
            f"| {_fmt(s.get('sel_strided_mean'))} | {_fmt(s.get('sel_random_row'))} "
            f"| {frac} | {_fmt(s.get('cos_to_mean'))} |"
        )
    return head + "\n".join(rows) + "\n"


def _separator(n_cols: int) -> str:
    return "|" + "---|" * n_cols + "\n"


def overlap_table(records: Sequence[dict], deltas: Sequence[int],
                  density: Optional[float] = None) -> str:
    """Top-key Jaccard vs query distance, with the chance baseline."""
    cols = " | ".join(f"d={d}" for d in deltas)
    head = (
        "### Top-key agreement vs query distance (Jaccard, median)\n\n"
        f"| model | chance | {cols} |\n" + _separator(len(deltas) + 2)
    )
    rows = []
    for rec in records:
        stats = collect_overlap_stats(rec, density)
        if not stats:
            continue
        cells = [_fmt(stats.get(f"jaccard@{d}")) for d in deltas]
        rows.append(
            f"| {rec['adapter']} | {_fmt(stats.get('chance_jaccard'))} | "
            + " | ".join(cells)
            + " |"
        )
    return head + "\n".join(rows) + "\n"


def query_similarity_table(records: Sequence[dict], deltas: Sequence[int]) -> str:
    """Cosine between query vectors vs distance -- the proposed cause."""
    cols = " | ".join(f"d={d}" for d in deltas)
    head = (
        "### Query-vector cosine vs distance (median)\n\n"
        f"| model | {cols} |\n" + _separator(len(deltas) + 1)
    )
    rows = []
    for rec in records:
        stats = collect_overlap_stats(rec)
        if not stats:
            continue
        cells = [_fmt(stats.get(f"qcos@{d}")) for d in deltas]
        rows.append(f"| {rec['adapter']} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def grouping_table(records: Sequence[dict], density: float) -> str:
    """Contiguous vs strided query grouping, where a grid makes both sensible.

    If strided retention beats contiguous, the queries that want the same keys
    are not neighbours in sequence, and a contiguous vertical vector is
    structurally the wrong unit for that modality -- regardless of how well its
    threshold is tuned.
    """
    tag = f"d{density:g}"
    rows = []
    for rec in records:
        groups: Dict[int, List[float]] = defaultdict(list)
        matched: Dict[int, List[float]] = defaultdict(list)
        sizes: Dict[int, List[float]] = defaultdict(list)
        for sample in rec.get("samples", []):
            for layer in sample.get("layers", []):
                for head in layer.get("heads", []):
                    for entry in head.get("strided", []):
                        stride = int(entry.get("stride", 0))
                        vert = entry.get(f"{tag}/strided_vertical")
                        orc = entry.get(f"{tag}/strided_oracle")
                        mv = entry.get(f"{tag}/matched_vertical")
                        mo = entry.get(f"{tag}/matched_oracle")
                        if vert is not None and orc:
                            groups[stride].append(vert / orc)
                            sizes[stride].append(entry.get("group_rows", float("nan")))
                        if mv is not None and mo:
                            matched[stride].append(mv / mo)
        if not groups:
            continue
        for stride, vals in sorted(groups.items()):
            base_vals = matched.get(stride)
            if not base_vals:
                continue
            strided_med, matched_med = median(vals), median(base_vals)
            # n is the number of (sample, layer, head) measurements behind the
            # median. Adapters that derive strides from the input -- Qwen2.5-VL
            # reads them from video_grid_thw -- produce a *different* stride per
            # clip, so a row can rest on a single video. Without n that is
            # invisible, and a +0.03 difference reads as a result.
            rows.append(
                f"| {rec['adapter']} | {int(median(sizes[stride]))} | {stride} "
                f"| {matched_med:.3f} | {strided_med:.3f} "
                f"| {strided_med - matched_med:+.3f} | {len(vals)} |"
            )
    if not rows:
        return ""
    return (
        f"### Contiguous vs strided query grouping (retention of per-row oracle, "
        f"density {density:g})\n\n"
        f"Both groupings use the *same number of query rows*, so the difference "
        f"is grouping shape rather than group size.\n\n"
        "| model | rows | stride | contiguous | strided | difference | n |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def layer_table(records: Sequence[dict], density: float) -> str:
    """Per-depth breakdown.

    Attention structure is strongly depth-dependent, and a median over sampled
    layers hides it: on Qwen2.5-VL, going from 2 to 6 sampled layers moved
    adjacent-query agreement from 0.46 to 0.15 and the global-key share from
    0.95 to 0.82. Those are different regimes, not noise, and a kernel that has
    to work at every depth needs to see the spread.
    """
    tag = f"d{density:g}"
    out = [f"### Per-layer breakdown (density {density:g})\n",
           "| model | depth | eff keys | line | block | **vertical** | oracle "
           "| vert/oracle | line/vert | n |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    any_row = False
    for rec in records:
        per_layer: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list))
        for layer_idx, _, block in iter_blocks(rec):
            for name in ("line", "block", "vertical", "oracle"):
                val = block.get(f"{tag}/{name}")
                if val is not None:
                    per_layer[layer_idx][name].append(float(val))
            if "eff_key_frac" in block:
                per_layer[layer_idx]["eff"].append(float(block["eff_key_frac"]))
        for layer_idx in sorted(per_layer):
            vals = per_layer[layer_idx]
            if not vals.get("vertical"):
                continue

            def med(key: str) -> Optional[float]:
                """Block-granular patterns are omitted below one key-block, so
                any column here can legitimately be missing."""
                return median(vals[key]) if vals.get(key) else None

            def fmt(value: Optional[float]) -> str:
                return "-" if value is None else f"{value:.3f}"

            vert = median(vals["vertical"])
            orc, line, eff = med("oracle"), med("line"), med("eff")
            any_row = True
            out.append(
                f"| {rec['adapter']} | {layer_idx} "
                f"| {'-' if eff is None else f'{eff:.2f}'} "
                f"| {fmt(line)} | {fmt(med('block'))} | **{vert:.3f}** "
                f"| {fmt(orc)} | {fmt(vert / orc if orc else None)} "
                f"| {fmt(line / vert if line is not None and vert else None)} "
                f"| {len(vals['vertical'])} |"
            )
    return "\n".join(out) + "\n" if any_row else ""


def error_table(records: Sequence[dict], density: float) -> str:
    """Relative output error, which is what a kernel actually changes.

    Mass recovered and output error are not interchangeable. A pattern can keep
    70% of the mass and still emit a nearly correct row (if the dropped mass was
    spread thinly over unimportant values) or a badly wrong one (if it was not).
    Lower is better here; 0.0 means the sparse output equals the dense one.
    """
    tag = f"d{density:g}"
    names = ["line", "block", "horizontal", "vertical", "oracle", "sel_pooled"]
    rows = []
    for rec in records:
        buckets: Dict[str, List[float]] = defaultdict(list)
        for _, _, block in iter_blocks(rec):
            for name in names:
                val = block.get(f"{tag}/err_{name}")
                if val is not None:
                    buckets[name].append(float(val))
        if not buckets:
            continue
        cells = [
            f"{median(buckets[n]):.3f}" if buckets.get(n) else "-" for n in names
        ]
        rows.append(f"| {rec['adapter']} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    return (
        f"### Relative output error at density {density:g} (lower is better)\n\n"
        "`||O_sparse - O_dense|| / ||O_dense||`, renormalised over kept keys.\n\n"
        "| model | line | block | horizontal | **vertical** | oracle | pooled |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def density_sweep_table(records: Sequence[dict], densities: Sequence[float]) -> str:
    """Retention across budgets.

    Reading one density is a choice, and the ordering between patterns need not
    survive it: a pattern that wins at 5% can lose at 1%, where every key must
    earn its place.
    """
    rows = []
    for rec in records:
        cells = []
        for density in densities:
            stats = collect_pattern_stats(rec, density)
            vert = stats.get("vertical", {}).get("median")
            orc = stats.get("oracle", {}).get("median")
            cells.append(f"{vert / orc:.2f}" if vert and orc else "-")
        if any(c != "-" for c in cells):
            rows.append(f"| {rec['adapter']} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    head = " | ".join(f"{d:g}" for d in densities)
    return (
        "### Retention (vertical/oracle) vs budget\n\n"
        f"| model | {head} |\n" + _separator(len(densities) + 1)
        + "\n".join(rows) + "\n"
    )


def minp_table(records: Sequence[dict]) -> str:
    """Density that VecAttention's own MinP rule selects.

    Their kernel does not use a fixed budget: it keeps every key within `gap`
    of the row maximum, so the density is content-adaptive. This is what that
    rule actually costs on each model -- the bridge between our fixed-budget
    comparison and their algorithm.
    """
    gaps: Dict[float, bool] = {}
    rows = []
    for rec in records:
        buckets: Dict[float, List[float]] = defaultdict(list)
        for _, _, block in iter_blocks(rec):
            for key, val in block.items():
                if key.startswith("minp") and key.endswith("/density"):
                    gap = float(key[4:].split("/")[0])
                    buckets[gap].append(float(val))
                    gaps[gap] = True
        if buckets:
            cells = [f"{median(buckets[g]):.3f}" if g in buckets else "-"
                     for g in sorted(gaps)]
            rows.append(f"| {rec['adapter']} | " + " | ".join(cells) + " |")
    if not rows:
        return ""
    head = " | ".join(f"gap={g:g}" for g in sorted(gaps))
    return (
        "### MinP selected density (VecAttention's own rule)\n\n"
        f"| model | {head} |\n" + _separator(len(gaps) + 1)
        + "\n".join(rows) + "\n"
    )


def iter_tiles(record: dict):
    for sample in record.get("samples", []):
        for layer in sample.get("layers", []):
            for head in layer.get("heads", []):
                tiles = head.get("tiles")
                if tiles:
                    yield layer.get("depth", 0), head["head"], tiles


def tile_table(records: Sequence[dict], gap: float = 4.0,
               sizes: Sequence[int] = (1, 2, 4, 8, 16)) -> str:
    """Can a FlashAttention tile's row-max be read off a few keys?

    `skippable` is the headroom: the fraction of (row, tile) pairs whose max
    sits more than `gap` below the row's global max, i.e. that BLASST could
    skip. If that is near zero there is nothing to win.

    `argmax agree` is 1 - (distinct argmax keys / rows): how much the 128 rows
    of a tile agree on *which* key is their maximum. High means one probe
    serves the whole tile.

    `oracle r=N` is the fraction of rows whose true max is recovered exactly
    from the best N keys of the tile. It is an upper bound -- the oracle
    already knows which keys to look at -- so it says whether the structure
    exists at all. `random r=N` is the same with keys chosen blindly, and the
    gap between them is what a real probe would have to close.
    """
    rows = []
    for rec in records:
        buckets: Dict[str, List[float]] = defaultdict(list)
        for _, _, tiles in iter_tiles(rec):
            for key, val in tiles.items():
                buckets[key].append(float(val))
        if not buckets:
            continue

        def med(key: str) -> str:
            return f"{median(buckets[key]):.3f}" if buckets.get(key) else "-"

        agree = (f"{1 - median(buckets['argmax_distinct_frac']):.3f}"
                 if buckets.get("argmax_distinct_frac") else "-")
        cells = [med(f"oracle_r{r}/exact_frac") for r in sizes]
        rand = med(f"random_r{sizes[len(sizes) // 2]}/exact_frac")
        rows.append(
            f"| {rec['adapter']} | {med(f'skippable@{gap:g}')} | {agree} | "
            + " | ".join(cells) + f" | {rand} |"
        )
    if not rows:
        return ""
    head = " | ".join(f"r={r}" for r in sizes)
    return (
        f"### Tile skip signal: is the row max determined by a few keys?\n\n"
        f"Fraction of rows whose tile max is recovered **exactly** from the "
        f"best `r` keys of a 64-key tile. `oracle` knows which keys to pick "
        f"(upper bound); `random` does not.\n\n"
        f"| model | skippable@{gap:g} | argmax agree | {head} | random r={sizes[len(sizes)//2]} |\n"
        + _separator(len(sizes) + 4) + "\n".join(rows) + "\n"
    )


def tile_decision_table(records: Sequence[dict], gap: float = 4.0,
                        sizes: Sequence[int] = (1, 4, 8, 16)) -> str:
    """Does probing a few keys change the *skip decision*?

    Recovering the exact max is stricter than BLASST needs. The rule is
    "keep this tile if its max is within `gap` of the row's running max", so an
    underestimate only matters when it drags the tile across that line.

    A subset max is always <= the true max, so the only possible error is a
    **false skip**: a tile that mattered, dropped. That is the number a kernel
    lives or dies by, and it is reported as a fraction of the tiles that should
    have been kept. `max err` is the average shortfall in score units, which
    says how close to the line those mistakes are.
    """
    rows = []
    for rec in records:
        buckets: Dict[str, List[float]] = defaultdict(list)
        for _, _, tiles in iter_tiles(rec):
            for key, val in tiles.items():
                buckets[key].append(float(val))
        if not buckets:
            continue

        def med(key: str) -> str:
            return f"{median(buckets[key]):.3f}" if buckets.get(key) else "-"

        cells = [med(f"oracle_r{r}/false_skip@{gap:g}") for r in sizes]
        # The realistic probe: keys chosen by norm, once per key tile, with no
        # knowledge of the queries. This is the column that decides whether the
        # idea is buildable -- the oracle only says whether the signal exists.
        cells += [med(f"norm_r{r}/false_skip@{gap:g}") for r in sizes]
        # Channel probes cut the other axis: all keys, few dimensions. They
        # cannot miss the max by looking in the wrong place, only by truncating
        # dimensions -- a different and apparently milder failure mode.
        cells += [med(f"chan{c}/false_skip@{gap:g}") for c in (8, 16, 32)]
        # Downsampling the query side: 1/P of the matmul, and the one that
        # works with the structure (rows of a tile share their argmax).
        cells += [med(f"poolq/false_skip@{gap:g}"),
                  med(f"poolq/false_keep@{gap:g}")]
        cells += [med(f"poolq_top{r}/false_skip@{gap:g}") for r in (4, 16)]
        cells += [med(f"bound/skip_recall@{gap:g}"),
                  med(f"box/skip_recall@{gap:g}")]
        rows.append(f"| {rec['adapter']} | {med(f'skippable@{gap:g}')} | "
                    + " | ".join(cells) + " |")
    if not rows:
        return ""
    oracle_h = " | ".join(f"or r={r}" for r in sizes)
    norm_h = " | ".join(f"**norm r={r}**" for r in sizes)
    return (
        f"### Tile skip decision quality at gap={gap:g}\n\n"
        f"**False-skip rate**: tiles that mattered but a probe would drop, as a "
        f"fraction of tiles that should be kept. Lower is better; a kernel "
        f"needs this near zero.\n\n"
        f"`or` = oracle, choosing the best keys per query-tile/key-tile pair -- "
        f"not buildable, shows whether the signal exists at all.\n"
        f"`norm` = the `r` largest-norm keys of the tile, chosen once per key "
        f"tile with no knowledge of the queries -- **this is the buildable "
        f"one**.\n"
        f"`meankey` = one dot product against the tile's mean key.\n"
        f"`poolQ` = mean the tile's queries into one and score it against all "
        f"64 keys: 1/128 of the matmul. `poolQ waste` is tiles it keeps "
        f"needlessly (lost saving, not lost accuracy). `pQ+N` lets the pooled "
        f"query nominate N keys which are then scored exactly per row.\n"
        f"`chN` = score every key using only the N largest-magnitude channels "
        f"of q (N of 64 dims) -- cuts the *other* axis, and cannot miss the max "
        f"by probing the wrong keys.\n"
        f"`ball rec` / `box rec` = fraction of genuinely skippable tiles caught "
        f"by the two provable upper bounds, which can *never* false-skip; "
        f"higher is better, 0 means the bound is too loose to be useful.\n\n"
        f"| model | skippable | {oracle_h} | {norm_h} | ch8 | ch16 | ch32 "
        f"| **poolQ** | poolQ waste | pQ+4 | pQ+16 | ball rec | box rec |\n"
        + _separator(2 * len(sizes) + 10) + "\n".join(rows) + "\n"
    )


def _sequence_length(rec: dict) -> str:
    for sample in rec.get("samples", []):
        for layer in sample.get("layers", []):
            return str(layer.get("n_key", "?"))
    return "?"


def verdict(records: Sequence[dict], density: float) -> str:
    """One line per model saying whether the pattern showed up.

    The decisive statistic is ``vertical / oracle``: how much of per-row
    selection a *single shared key set per query block* retains.  That is the
    claim itself -- rows in a block want the same keys -- and it is the number
    the synthetic control pins down (1.00 when queries inside a block are
    identical, 0.27 when they are independent).

    ``vertical - block`` is reported too, but it is not used to decide, because
    part of it is free: picking individual key columns beats picking key blocks
    even when queries are completely unrelated, purely from finer granularity
    along the key axis.  The independent-query control shows exactly that
    (+0.036 with no shared structure whatsoever).

    Mechanism check: ``jaccard@1`` against its chance level.  Both have to hold
    for a "yes", so a model cannot pass on granularity alone.
    """
    lines = ["### Verdict\n"]
    for rec in records:
        s = collect_pattern_stats(rec, density)
        o = collect_overlap_stats(rec, density)
        if not s:
            lines.append(f"- **{rec['adapter']}**: no usable captures.")
            continue
        # A density that buys only a handful of keys makes every pattern
        # degenerate: top-2 sets of neighbouring rows can disagree completely
        # while the underlying geometry is perfectly ordinary. Say so rather
        # than reporting a "no" that is really an artefact of the budget.
        try:
            seq = int(_sequence_length(rec))
            budget = round(density * seq)
        except ValueError:
            budget = None
        if budget is not None and budget < 4:
            lines.append(
                f"- **{rec['adapter']}** ({rec['modality']}): density {density:g} "
                f"selects only {budget} key(s) out of {seq} -- too coarse to "
                f"measure. Re-report this model at a higher density "
                f"(`--density 0.2`)."
            )
            continue

        eff = s.get("eff_key_frac", {}).get("median", float("nan"))

        vert = s.get("vertical", {}).get("median", float("nan"))
        blk = s.get("block", {}).get("median", float("nan"))
        horiz = s.get("horizontal", {}).get("median", float("nan"))
        orc = s.get("oracle", {}).get("median", float("nan"))
        j1 = o.get("jaccard@1", {}).get("median", float("nan"))
        chance = o.get("chance_jaccard", {}).get("median", float("nan"))
        retention = vert / orc if orc else float("nan")
        above_chance = j1 > 3 * chance if chance else j1 > 0.1
        if retention >= 0.8 and above_chance:
            word = "yes"
        elif retention >= 0.5 and above_chance:
            word = "partial"
        else:
            word = "no"
        if rec.get("invalid_weights"):
            lines.append(
                f"- **{rec['adapter']}** ({rec['modality']}): **INVALID** -- some "
                f"attention parameters were randomly initialised at load time, so "
                f"its Q/K are not the pretrained model's. Numbers below are not "
                f"interpretable; fix the checkpoint and re-run."
            )
            continue
        # Diffuseness does not change whether neighbouring queries agree; it
        # changes whether anyone should care. Say both rather than suppressing
        # the mechanism result.
        caveat = ""
        if eff == eff and eff > 0.4:
            caveat = (
                f" **Caveat:** rows spread their mass over {eff:.0%} of available "
                f"keys, so no sparse pattern helps much here regardless of shape."
            )
        # Two different mechanisms produce a high retention number, and they
        # have opposite engineering consequences. If one key set for the whole
        # head (`line`) already recovers most of what a per-block set does, the
        # structure is globally salient keys (attention sinks), not neighbouring
        # queries agreeing -- and then a vertical *vector* buys almost nothing
        # over a vertical *line*.
        line = s.get("line", {}).get("median", float("nan"))
        # Only meaningful when the shared key set works at all: with
        # independent queries `line` and `vertical` are both near-useless and
        # their ratio is ~0.9, which fired the sink warning on a control that
        # has no sinks by construction.
        if vert and line == line and retention >= 0.5:
            share = line / vert
            if share > 0.85:
                caveat += (
                    f" **Mechanism:** a single global key set already recovers "
                    f"{share:.0%} of what per-block sets do, so this is key "
                    f"salience (sinks), not neighbouring-query agreement."
                )
        lines.append(
            f"- **{rec['adapter']}** ({rec['modality']}): _{word}_ -- a shared key "
            f"set per {rec['query_block']}-query block retains {retention:.2f} of "
            f"per-row oracle mass; adjacent-query key overlap {j1:.3f} vs chance "
            f"{chance:.3f} ({j1 / chance:.1f}x); practical gain over block-sparsity "
            f"{vert - blk:+.3f}, over horizontal {vert - horiz:+.3f}.{caveat}"
        )
    return "\n".join(lines) + "\n"


def report(records: Sequence[dict], density: float, deltas: Sequence[int],
           by_layer: bool = False, gap: float = 4.0) -> str:
    parts = [
        "# Q/K vertical-vector sparsity across modalities\n",
        verdict(records, density),
        "",
        pattern_table(records, density),
        "",
        selector_table(records, density),
        "",
        grouping_table(records, density),
        "",
        overlap_table(records, deltas, density),
        "",
        query_similarity_table(records, deltas),
        "",
        error_table(records, density),
        "",
        density_sweep_table(records, (0.01, 0.02, 0.05, 0.1, 0.2)),
        "",
        minp_table(records),
        "",
        tile_table(records, gap),
        "",
        tile_decision_table(records, gap),
    ]
    if by_layer:
        parts += ["", layer_table(records, density)]
    warnings = []
    for rec in records:
        for w in rec.get("warnings", []):
            warnings.append(f"- {rec['adapter']}: {w}")
    if warnings:
        parts += ["", "### Warnings\n", *warnings]
    return "\n".join(parts)
