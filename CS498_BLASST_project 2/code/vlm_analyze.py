"""
Step 1 of the VLM region study, CPU only: % of tiles skipped per attention region, from the margins that
vlm_regions.py recorded (one dense prefill per prompt, Qwen2.5-VL-7B-Instruct).

Regions. Every 128-token block of the prompt is "text" (no video tokens), "video" (all video tokens) or
"mixed". A tile = (query block, key block), and its region is:
  V->V  video rows looking at video keys       T->V  text rows (the question) looking at video keys
  T->T  text rows looking at text keys         V->T  video rows looking back at the transcript before it
  mixed either block straddles a text/video boundary

For each prompt we pick the ONE lambda that skips exactly 50% (and 75%) of all causal tiles (all 28 layers
x 28 heads), the way BLASST uses one lambda everywhere, and then ask how those skips are spread over the
regions. A tile is skipped iff margin < ln(lambda).

Writes results/vlm_regions/: prompts.csv, region_table.csv, per_layer.csv, curves.csv,
region_curves.png, per_layer.png
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

W = "/work/nvme/beqg/zyang21/blasst"
OUT = f"{W}/results/vlm_regions"
TILE = 128
REGIONS = ["V->V", "T->V", "T->T", "V->T", "mixed"]           # order = color order (key pair first)
# reference palette, categorical slots 1-5 in fixed order (validated: adjacent pairs pass for lines)
COLORS = dict(zip(REGIONS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LAMS = np.logspace(-8, 0, 161)
TARGETS = (0.50, 0.75)
DIST_BUCKETS = [(0, 0), (1, 3), (4, 15), (16, 47), (48, 10**9)]   # query tile minus key tile


def block_kinds(is_vision):
    """0 = text, 1 = mixed, 2 = video, for each 128-token block (the last block may be shorter)."""
    n = math.ceil(is_vision.numel() / TILE)
    frac = torch.stack([is_vision[i * TILE:(i + 1) * TILE].float().mean() for i in range(n)])
    return torch.where(frac == 0, 0, torch.where(frac == 1, 2, 1))


def region_map(kind):
    """[n, n] region index into REGIONS for every (query block, key block)."""
    q, k = kind[:, None], kind[None, :]
    r = torch.full((len(kind), len(kind)), REGIONS.index("mixed"))
    r[(q == 2) & (k == 2)] = REGIONS.index("V->V")
    r[(q == 0) & (k == 2)] = REGIONS.index("T->V")
    r[(q == 0) & (k == 0)] = REGIONS.index("T->T")
    r[(q == 2) & (k == 0)] = REGIONS.index("V->T")
    return r


def threshold_for(sorted_vals, frac):
    """ln(lambda) that skips exactly round(frac * N) of the sorted margins (midpoint between neighbours)."""
    k = int(round(frac * sorted_vals.numel()))
    return float((sorted_vals[k - 1] + sorted_vals[k]) / 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for ln in open(f"{W}/vlm_margins/runs.jsonl"):
        r = json.loads(ln)
        runs[(r["videoID"], r["length_name"])] = r                       # last run wins
    prompts, table, layer_rows, counts, dist_rows, qsplit = [], [], [], {}, [], []
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    for path in sorted(glob.glob(f"{W}/vlm_margins/*.pt")):
        d = torch.load(path)
        m, meta = d["margins"].float(), d["meta"]                         # [28, 28, n, n]
        Ln = meta["length_name"]
        S = d["is_vision"].numel()
        kind = block_kinds(d["is_vision"])
        causal = torch.isfinite(m[0, 0])
        assert torch.equal(causal, torch.tril(torch.ones_like(causal))), "causal tiles should be the lower triangle"
        reg = region_map(kind)[causal]                                    # [n causal]
        qi, ki = torch.nonzero(causal, as_tuple=True)                     # tile coordinates, same order
        dist = qi - ki                                                    # distance from the diagonal, in tiles
        vals = m[:, :, causal]                                            # [28, 28, n causal]
        ln_lam = {t: threshold_for(vals.flatten().sort().values, t) for t in TARGETS}
        n_all = vals.numel()
        run = runs.get((meta["videoID"], Ln), {})
        p = {**{k: meta[k] for k in ("videoID", "length_name", "domain", "video_seconds", "nframes", "H", "W",
                                      "video_tokens", "text_tokens", "L")},
             "q_blocks_text": int((kind == 0).sum()), "q_blocks_video": int((kind == 2).sum()),
             "q_blocks_mixed": int((kind == 1).sum()),
             "lambda_50": math.exp(ln_lam[0.5]), "a_50": math.exp(ln_lam[0.5]) * S,
             "lambda_75": math.exp(ln_lam[0.75]), "a_75": math.exp(ln_lam[0.75]) * S,
             "answer_top1": run.get("answer_top1"), "answer_true": meta["answer"],
             "prefill_seconds": run.get("seconds_ours_recording")}
        # T->V split: blocks holding the question (and answer position) vs transcript-only blocks after the video
        ids = d["input_ids"]
        q_start = next(i for i in range(len(ids) - 1, -1, -1) if tok.decode(ids[i:i + 2]).startswith("Question"))
        tv = reg == REGIONS.index("T->V")
        for part, sel in (("question blocks", tv & (qi >= q_start // TILE)), ("transcript blocks", tv & (qi < q_start // TILE))):
            v = vals[:, :, sel]
            qsplit.append({"prompt": f"{meta['videoID']}_{Ln}", "length_name": Ln, "part": part,
                           "tiles": v.numel(), "skipped": int((v < ln_lam[0.5]).sum())})
        n_check = 0
        for ri, rname in enumerate(REGIONS):
            v = vals[:, :, reg == ri]                                     # [28, 28, n region tiles]
            n = v.numel()
            n_check += n
            # distance control: same region, split by how far the key tile is behind the query tile
            for b, (lo, hi) in enumerate(DIST_BUCKETS):
                sel = (reg == ri) & (dist >= lo) & (dist <= hi)
                vb = vals[:, :, sel]
                if vb.numel():
                    dist_rows.append({"length_name": Ln, "region": rname, "bucket": b, "tiles": vb.numel(),
                                      "skipped": int((vb < ln_lam[0.5]).sum())})
            row = {"scope": f"{meta['videoID']}_{Ln}", "length_name": Ln, "region": rname, "tiles": n,
                   "share_pct": 100 * n / n_all}
            for t in TARGETS:
                row[f"skipped_{int(t*100)}"] = int((v < ln_lam[t]).sum())
                row[f"pct_skipped_at_overall_{int(t*100)}"] = 100 * row[f"skipped_{int(t*100)}"] / max(n, 1)
            if n:
                sv = v.flatten().sort().values                            # all margins of the region, sorted
                pct = lambda f: sv[int(f * (n - 1))].item()               # exact percentile from the sorted list
                row.update({"margin_p10": pct(0.1), "margin_median": pct(0.5), "margin_p90": pct(0.9),
                            "pct_margin_zero": 100 * (v == 0).float().mean().item()})
                # per layer at overall 50%
                sk = (v < ln_lam[0.5]).float().mean(dim=(1, 2))           # [28]
                for layer in range(v.shape[0]):
                    layer_rows.append({"scope": row["scope"], "length_name": Ln, "region": rname, "layer": layer,
                                       "tiles": v[layer].numel(), "skipped": float(sk[layer]) * v[layer].numel()})
                # curve: skipped counts at each lambda (summed over prompts later)
                c = torch.searchsorted(sv, torch.tensor(np.log(LAMS), dtype=sv.dtype)).numpy()
                acc = counts.setdefault((Ln, rname), [np.zeros(len(LAMS)), 0])
                acc[0] += c
                acc[1] += n
            table.append(row)
        assert n_check == n_all, "region tile counts must add up to all causal tiles"
        p["overall_pct_skipped_50"] = 100 * float((vals < ln_lam[0.5]).float().mean())
        p["overall_pct_skipped_75"] = 100 * float((vals < ln_lam[0.75]).float().mean())
        prompts.append(p)
        print(f"{p['videoID']} {Ln}: L={S} lambda50={p['lambda_50']:.3g} (a={p['a_50']:.0f}) "
              f"overall {p['overall_pct_skipped_50']:.2f}%  " +
              "  ".join(f"{r['region']} {r['pct_skipped_at_overall_50']:.1f}%" for r in table[-5:]), flush=True)

    qs = pd.DataFrame(qsplit)
    qs = pd.concat([qs, qs.assign(prompt="pooled_" + qs.length_name).groupby(["prompt", "length_name", "part"])[["tiles", "skipped"]].sum().reset_index(),
                    qs.assign(prompt="pooled_all", length_name="all").groupby(["prompt", "length_name", "part"])[["tiles", "skipped"]].sum().reset_index()])
    qs["pct_skipped_at_overall_50"] = 100 * qs.skipped / qs.tiles
    qs.to_csv(f"{OUT}/t2v_question_split.csv", index=False, float_format="%.4g")
    print(qs[qs.prompt.str.startswith("pooled")].round(2).to_string(index=False))
    dt = pd.DataFrame(dist_rows).groupby(["length_name", "region", "bucket"])[["tiles", "skipped"]].sum().reset_index()
    dt["distance_tiles"] = dt.bucket.map(lambda b: f"{DIST_BUCKETS[b][0]}-{DIST_BUCKETS[b][1]}" if DIST_BUCKETS[b][1] < 10**6
                                         else f">={DIST_BUCKETS[b][0]}")
    dt["pct_skipped_at_overall_50"] = 100 * dt.skipped / dt.tiles
    dt.to_csv(f"{OUT}/distance_table.csv", index=False, float_format="%.4g")
    piv = dt[dt.region.isin(["V->V", "T->V"])].pivot_table(index=["length_name", "distance_tiles"], columns="region",
                                                             values="pct_skipped_at_overall_50")
    print("distance control (% skipped at overall 50%):\n", piv.round(1).to_string())
    tab = pd.DataFrame(table)
    # pooled rows: all prompts of a length (and all 10), each prompt at its own 50% / 75% lambda
    pooled = []
    for scope, sub in [("pooled_8k", tab[tab.length_name == "8k"]), ("pooled_16k", tab[tab.length_name == "16k"]),
                       ("pooled_all", tab)]:
        tot = sub.groupby("region").tiles.sum().sum()
        for rname in REGIONS:
            s = sub[sub.region == rname]
            row = {"scope": scope, "length_name": scope.split("_")[1], "region": rname, "tiles": int(s.tiles.sum()),
                   "share_pct": 100 * s.tiles.sum() / tot}
            for t in TARGETS:
                k = int(t * 100)
                row[f"skipped_{k}"] = int(s[f"skipped_{k}"].sum())
                row[f"pct_skipped_at_overall_{k}"] = 100 * s[f"skipped_{k}"].sum() / max(s.tiles.sum(), 1)
            # margin percentiles pooled: tile-weighted mean of per-prompt percentiles (approximate)
            for c in ("margin_p10", "margin_median", "margin_p90", "pct_margin_zero"):
                row[c] = float(np.average(s[c], weights=s.tiles)) if s.tiles.sum() else float("nan")
            row["prompt_min_pct_50"] = s.pct_skipped_at_overall_50.min()
            row["prompt_max_pct_50"] = s.pct_skipped_at_overall_50.max()
            pooled.append(row)
    tab = pd.concat([pd.DataFrame(pooled), tab], ignore_index=True)
    tab.to_csv(f"{OUT}/region_table.csv", index=False, float_format="%.4g")
    pd.DataFrame(prompts).to_csv(f"{OUT}/prompts.csv", index=False, float_format="%.4g")

    lay = pd.DataFrame(layer_rows).groupby(["length_name", "region", "layer"])[["tiles", "skipped"]].sum().reset_index()
    lay["pct_skipped_at_overall_50"] = 100 * lay.skipped / lay.tiles
    lay.to_csv(f"{OUT}/per_layer.csv", index=False, float_format="%.4g")

    cur = [{"length_name": Ln, "region": rname, "lam": lam, "pct_skipped": 100 * c / n}
           for (Ln, rname), (cs, n) in counts.items() for lam, c in zip(LAMS, cs)]
    cur = pd.DataFrame(cur)
    cur.to_csv(f"{OUT}/curves.csv", index=False, float_format="%.5g")

    # ------------------------------------------------------------------ figures
    lengths = [L for L in ("8k", "16k") if L in set(cur.length_name)]
    pr = pd.DataFrame(prompts)

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK2)
        ax.tick_params(colors=INK2, labelsize=8.5)

    def direct_labels(ax, ends, x, gap):
        placed = []
        for y, name in sorted(ends, reverse=True):
            y_lab = min(y, placed[-1] - gap) if placed else y
            placed.append(y_lab)
            ax.annotate(name, (x, y_lab), xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK,
                        va="center", annotation_clip=False)

    # Figure 1: % skipped vs lambda per region
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.3 * len(lengths), 3.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, Ln in zip(np.atleast_1d(axes), lengths):
        style(ax)
        lam50 = pr[pr.length_name == Ln].lambda_50
        ax.axvspan(lam50.min(), lam50.max(), color=GRID, alpha=0.7, linewidth=0)
        ax.annotate("one lambda that skips\n50% of all tiles\n(range over 5 prompts)", (lam50.min(), 97),
                    xytext=(-4, 0), textcoords="offset points", ha="right", va="top", fontsize=7, color=INK2)
        ends = []
        for rname in REGIONS:
            d = cur[(cur.length_name == Ln) & (cur.region == rname)]
            if d.empty:
                continue
            ax.plot(d.lam, d.pct_skipped, "--" if rname == "mixed" else "-", color=COLORS[rname],
                    linewidth=1.3 if rname == "mixed" else 1.9, label=rname)
            ends.append((float(d.pct_skipped.iloc[-1]), rname))
        direct_labels(ax, ends, 1.0, 5)
        ax.set_xscale("log")
        ax.set_xlim(1e-6, 1)
        ax.set_ylim(0, 100)
        n = len(pr[pr.length_name == Ln])
        ax.set_title(f"{Ln.upper()}  ({n} prompts, {int(pr[pr.length_name == Ln].L.mean()):,} tokens on average)",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("threshold lambda (log scale)", fontsize=9, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("% of causal tiles skipped in the region", fontsize=9, color=INK)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=5, bbox_to_anchor=(0.01, 0.9))
    fig.suptitle("Qwen2.5-VL-7B, Video-MME prompts: % of tiles skipped vs lambda, per attention region",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.85))
    fig.savefig(f"{OUT}/region_curves.png", facecolor=SURFACE)

    # Figure 2: per layer at the lambda that skips 50% overall
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.3 * len(lengths), 3.9), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, Ln in zip(np.atleast_1d(axes), lengths):
        style(ax)
        ends = []
        for rname in REGIONS:
            d = lay[(lay.length_name == Ln) & (lay.region == rname)]
            if d.empty:
                continue
            ax.plot(d.layer, d.pct_skipped_at_overall_50, "--" if rname == "mixed" else "-", color=COLORS[rname],
                    linewidth=1.3 if rname == "mixed" else 1.9, marker=None if rname == "mixed" else "o", ms=3.2,
                    mfc=COLORS[rname], mec=SURFACE, mew=0.6, label=rname)
            ends.append((float(d.pct_skipped_at_overall_50.iloc[-1]), rname))
        direct_labels(ax, ends, 27, 5)
        ax.axhline(50, color=INK2, linewidth=0.8)
        ax.set_xlim(-0.5, 27.5)
        ax.set_ylim(0, 100)
        ax.set_title(f"{Ln.upper()}: at the lambda that skips 50% of all tiles", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("layer", fontsize=9, color=INK)
    np.atleast_1d(axes)[0].set_ylabel("% of causal tiles skipped in the region", fontsize=9, color=INK)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=5, bbox_to_anchor=(0.01, 0.9))
    fig.suptitle("Per layer: where one lambda's skips land (gray line = 50%, the overall rate)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.85))
    fig.savefig(f"{OUT}/per_layer.png", facecolor=SURFACE)

    pd.set_option("display.width", 220)
    cols = ["scope", "region", "tiles", "share_pct", "pct_skipped_at_overall_50", "pct_skipped_at_overall_75",
            "margin_p10", "margin_median", "margin_p90", "pct_margin_zero", "prompt_min_pct_50", "prompt_max_pct_50"]
    print(tab[tab.scope.str.startswith("pooled")][cols].round(2).to_string(index=False))
    print(pd.DataFrame(prompts)[["videoID", "length_name", "L", "video_tokens", "lambda_50", "a_50", "lambda_75",
                                 "overall_pct_skipped_50", "answer_top1", "answer_true"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
