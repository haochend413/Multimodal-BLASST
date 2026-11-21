"""
VLM study, second batch, CPU only: tables and figures for VLM_REGIONS_2.md from vlm2_tensors/ and
results/vlm_regions/v2_pilot.jsonl. Writes only new files (prefix v2_) into results/vlm_regions/.

Weight lost (items 1 and 2). For one (layer, head, query row), the dense softmax weights sum to 1; the
"weight lost" at lambda is the part of that sum sitting on tiles that lambda skips. Rows are grouped by their
128-token block: text before the video, video, text after the video, the question's blocks, and the answer
position (the last token, whose output picks the letter). Keys are split by token: video tokens vs the rest.
Averages are over layers, heads and rows, pooled over the prompts of a set and length.

Region % skipped (item 2) uses the tile regions of step 1 (VLM_REGIONS.md), at each prompt's own lambda that
skips exactly 50% (75%) of all causal tiles.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from vlm_analyze import REGIONS, block_kinds, region_map

W = "/work/nvme/beqg/zyang21/blasst"
TENS = f"{W}/vlm2_tensors"
OUT = f"{W}/results/vlm_regions"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TILE = 128


def question_start(ids, tok, layout):
    """Position of the first token of the question: "Question:" in layout A, "Select the best answer" in B/C."""
    words = tok.convert_ids_to_tokens(ids.tolist())
    marker = "Question" if layout == "A" else "Select"
    return max(i for i, w in enumerate(words) if w == marker)


def row_groups(ids, is_vision, q_start):
    """Boolean masks over query positions (by token) for each row group."""
    S = ids.numel()
    pos = torch.arange(S)
    groups = {"all rows": torch.ones(S, dtype=torch.bool)}
    if is_vision.any():
        first, last = int(is_vision.nonzero().min()), int(is_vision.nonzero().max())
        groups["text before video"] = ~is_vision & (pos < first)
        groups["video"] = is_vision
        groups["text after video"] = ~is_vision & (pos > last)
        groups["question"] = pos >= q_start
    groups["answer position"] = pos == S - 1
    return groups


def weight_rows(d, groups, layout_set, length, prompt):
    """Per row group x key kind x lambda: dense weight %, weight lost %, and per-row loss distribution."""
    rows, dist, layers = [], [], []
    dv = d["dense_vid"].float()                                    # [L, H, S]
    lv, lo = d["lost_vid"].float(), d["lost_other"].float()        # [n_lams, L, H, S]
    for g, mask in groups.items():
        if not mask.any():
            continue
        n = int(mask.sum()) * dv.shape[0] * dv.shape[1]
        dense = {"video keys": dv[..., mask], "other keys": 1 - dv[..., mask]}
        for j, lam_name in enumerate(d["lams"]):
            lost = {"video keys": lv[j][..., mask], "other keys": lo[j][..., mask]}
            lost["all keys"] = lost["video keys"] + lost["other keys"]
            for kk in ("video keys", "other keys", "all keys"):
                dw = 100.0 if kk == "all keys" else 100 * dense[kk].mean().item()
                lw = 100 * lost[kk].mean().item()
                rows.append({"set": layout_set, "length_name": length, "prompt": prompt, "row_group": g, "keys": kk,
                             "lambda_name": lam_name, "lambda": d["lams"][lam_name], "values": n,
                             "dense_weight_sum": dw * n / 100, "lost_weight_sum": lw * n / 100})
            tot = lost["all keys"].flatten()
            q = torch.quantile(tot[torch.randperm(tot.numel())[:2_000_000]] if tot.numel() > 2_000_000 else tot,
                               torch.tensor([0.5, 0.9, 0.99]))
            dist.append({"set": layout_set, "length_name": length, "prompt": prompt, "row_group": g,
                         "lambda_name": lam_name, "values": n, "p50": 100 * q[0].item(), "p90": 100 * q[1].item(),
                         "p99": 100 * q[2].item(), "max": 100 * tot.max().item(),
                         "n_gt_1pct": int((tot > 0.01).sum()), "n_gt_5pct": int((tot > 0.05).sum()),
                         "n_gt_10pct": int((tot > 0.10).sum())})
            if lam_name in ("exact50", "cal50"):
                per_layer_lost = (lost["all keys"].mean(dim=(1, 2)), lost["video keys"].mean(dim=(1, 2)))
                per_layer_dense = dense["video keys"].mean(dim=(1, 2))
                for layer in range(dv.shape[0]):
                    layers.append({"set": layout_set, "length_name": length, "prompt": prompt, "row_group": g,
                                   "lambda_name": lam_name, "layer": layer, "values": n // dv.shape[0],
                                   "lost_all_sum": per_layer_lost[0][layer].item() * n / dv.shape[0],
                                   "lost_vid_sum": per_layer_lost[1][layer].item() * n / dv.shape[0],
                                   "dense_vid_sum": per_layer_dense[layer].item() * n / dv.shape[0]})
    return rows, dist, layers


def region_skips(d, set_name, length, prompt, q_start):
    """Step 1's tile regions, plus "question->video": query block holds a question token, key block holds a
    video token (the question has no block of its own when it is short, e.g. layout C)."""
    m = d["margins"].float()
    causal = torch.isfinite(m[0, 0])
    reg = region_map(block_kinds(d["is_vision"]))[causal]
    vals = m[:, :, causal]
    n = m.shape[-1]
    vis = torch.nn.functional.pad(d["is_vision"], (0, n * TILE - d["is_vision"].numel())).view(n, TILE).any(1)
    q_blk = torch.arange(n) >= q_start // TILE
    qv = (q_blk[:, None] & vis[None, :])[causal]
    out = []
    for ri, rname in [(i, r) for i, r in enumerate(REGIONS)] + [(None, "question->video")]:
        v = vals[:, :, qv] if ri is None else vals[:, :, reg == ri]
        r = {"set": set_name, "length_name": length, "prompt": prompt, "region": rname, "tiles": v.numel()}
        for name in ("exact50", "exact75"):
            r[f"skipped_{name}"] = int((v < np.log(d["lams"][name])).sum())
        out.append(r)
    return out


def weight_figure(pooled):
    """Dot chart: mean weight lost per row group at the 50% lambda, one marker per (set, length)."""
    show = pooled[(pooled["keys"] == "all keys") & (pooled.lambda_name == "exact50")]
    groups = ["text before video", "video", "text after video", "question", "answer position", "all rows"]
    sets = [("qwen_A", "Qwen, layout A: [text][video][text][question]", "#2a78d6"),
            ("qwen_B", "Qwen, layout B: [video][subtitles][question]", "#eb6834"),
            ("qwen_C", "Qwen, layout C: [video][question]", "#1baf7a"),
            ("llama", "Llama-3.1-8B, RULER text", "#eda100")]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for gi, g in enumerate(groups):
        y0 = -gi
        for si, (sname, label, col) in enumerate(sets):
            if g == "text before video" and sname in ("qwen_B", "qwen_C"):
                continue                                   # only the system prompt comes before the video there
            for L, mk, dy in (("8k", "o", -0.07), ("16k", "s", 0.07)):
                x = show[(show.set.str.startswith(sname)) & (show.length_name == L) & (show.row_group == g)]
                if x.empty:
                    continue
                ax.plot(x.lost_weight_pct.iloc[0], y0 + 0.3 - 0.2 * si + dy, mk, color=col, ms=6, mec=SURFACE, mew=0.8,
                        label=label if (gi == len(groups) - 1 and L == "8k") else None)
    ax.set_yticks([-i for i in range(len(groups))]); ax.set_yticklabels(groups, color=INK, fontsize=8.5)
    ax.set_xlim(0, max(8, show.lost_weight_pct.max() * 1.1))
    ax.set_xlabel("weight lost: mean % of each row's dense attention that sits on skipped tiles", color=INK, fontsize=9)
    ax.set_title("Weight on skipped tiles, at the lambda that skips 50% of all tiles (circle 8K, square 16K)",
                 color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/v2_weight_lost.png", facecolor=SURFACE)


def main():
    import sys
    if "--figures-only" in sys.argv:
        weight_figure(pd.read_csv(f"{OUT}/v2_weight_regions.csv"))
        return
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    rows, dist, layers, regs = [], [], [], []
    for path in sorted(glob.glob(f"{TENS}/*/*.pt")):
        d = torch.load(path)
        set_name, prompt = d["set"], d["name"]
        length = d["meta"]["length_name"]
        layout = set_name.split("_")[1] if set_name.startswith("qwen") else "llama"
        q_start = question_start(d["input_ids"], tok, layout) if layout != "llama" else d["input_ids"].numel()
        groups = row_groups(d["input_ids"], d["is_vision"], q_start)
        r, ds, ly = weight_rows(d, groups, set_name, length, prompt)
        rows += r; dist += ds; layers += ly
        if layout != "llama":
            regs += region_skips(d, set_name, length, prompt, q_start)
        print("read", set_name, prompt, flush=True)

    # ---------------- pooled tables (sums over prompts, then ratios)
    wr = pd.DataFrame(rows)
    pooled = wr.groupby(["set", "length_name", "row_group", "keys", "lambda_name"])[
        ["values", "dense_weight_sum", "lost_weight_sum"]].sum().reset_index()
    pooled["dense_weight_pct"] = 100 * pooled.dense_weight_sum / pooled["values"]
    pooled["lost_weight_pct"] = 100 * pooled.lost_weight_sum / pooled["values"]
    pooled["pct_of_region_weight_removed"] = 100 * pooled.lost_weight_sum / pooled.dense_weight_sum
    pooled.to_csv(f"{OUT}/v2_weight_regions.csv", index=False, float_format="%.4g")
    wr.to_csv(f"{OUT}/v2_weight_regions_per_prompt.csv", index=False, float_format="%.5g")

    dd = pd.DataFrame(dist)
    dp = dd.groupby(["set", "length_name", "row_group", "lambda_name"]).agg(
        values=("values", "sum"), p50_median_over_prompts=("p50", "median"), p90_median_over_prompts=("p90", "median"),
        p99_median_over_prompts=("p99", "median"), max=("max", "max"), n_gt_1pct=("n_gt_1pct", "sum"),
        n_gt_5pct=("n_gt_5pct", "sum"), n_gt_10pct=("n_gt_10pct", "sum")).reset_index()
    for t in ("1pct", "5pct", "10pct"):
        dp[f"share_rows_gt_{t}"] = 100 * dp[f"n_gt_{t}"] / dp["values"]
    dp.to_csv(f"{OUT}/v2_weight_distribution.csv", index=False, float_format="%.4g")

    ly = pd.DataFrame(layers).groupby(["set", "length_name", "row_group", "lambda_name", "layer"])[
        ["values", "lost_all_sum", "lost_vid_sum", "dense_vid_sum"]].sum().reset_index()
    for c in ("lost_all", "lost_vid", "dense_vid"):
        ly[f"{c}_pct"] = 100 * ly[f"{c}_sum"] / ly["values"]
    ly.to_csv(f"{OUT}/v2_weight_layers.csv", index=False, float_format="%.4g")

    rg = pd.DataFrame(regs)
    rp = rg.groupby(["set", "length_name", "region"])[["tiles", "skipped_exact50", "skipped_exact75"]].sum().reset_index()
    tot = rp[rp.region != "question->video"].groupby(["set", "length_name"]).tiles.sum()
    tot = rp.set_index(["set", "length_name"]).index.map(tot)
    rp["share_pct"] = 100 * rp.tiles / tot
    rp["pct_skipped_at_overall_50"] = 100 * rp.skipped_exact50 / rp.tiles.clip(lower=1)
    rp["pct_skipped_at_overall_75"] = 100 * rp.skipped_exact75 / rp.tiles.clip(lower=1)
    rp.to_csv(f"{OUT}/v2_layout_regions.csv", index=False, float_format="%.4g")

    weight_figure(pooled)

    # ---------------- pilot summary
    if os.path.exists(f"{OUT}/v2_pilot.jsonl"):
        pr = pd.DataFrame([json.loads(l) for l in open(f"{OUT}/v2_pilot.jsonl")])
        pr = pr.drop_duplicates(["question_id", "setting"], keep="last")
        full = pr.groupby("question_id").setting.nunique()
        pr["all_three"] = pr.question_id.map(full) == 3
        summ = []
        for scope, sub in [("all", pr), ("short", pr[pr.duration == "short"]), ("medium", pr[pr.duration == "medium"])]:
            for s in ("dense", "one50", "protect"):
                x = sub[(sub.setting == s) & sub.all_three]
                if x.empty:
                    continue
                summ.append({"videos": scope, "setting": s, "correct": int(x.correct.sum()), "N": len(x),
                             "pct_skipped_live_mean": x.get("pct_skipped_live", pd.Series(dtype=float)).mean(),
                             "pct_skipped_TV_live_mean": x.get("pct_skipped_TV_live", pd.Series(dtype=float)).mean(),
                             "pct_skipped_VV_live_mean": x.get("pct_skipped_VV_live", pd.Series(dtype=float)).mean()})
        pd.DataFrame(summ).to_csv(f"{OUT}/v2_pilot_summary.csv", index=False, float_format="%.3g")
        print(pd.DataFrame(summ).round(2).to_string(index=False))

    pd.set_option("display.width", 250)
    print(pooled[(pooled.lambda_name.isin(["exact50", "cal50"])) & (pooled["keys"] != "other keys")]
          [["set", "length_name", "row_group", "keys", "lambda_name", "dense_weight_pct", "lost_weight_pct",
            "pct_of_region_weight_removed"]].round(3).to_string(index=False))
    print(rp.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
