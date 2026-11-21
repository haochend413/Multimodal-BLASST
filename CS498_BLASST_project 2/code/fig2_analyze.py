"""
Fig. 2 analysis (CPU only). Reads results/fig2_ruler_*.csv and produces:

  results/fig2_table.csv         per length x config: score per task, average over tasks, achieved % skipped,
                                 change of the average vs dense with a 95% resampling range
  results/fig2_accuracy_vs_sparsity.png
                                 RULER score relative to dense (100 = same as dense) vs achieved % skipped,
                                 one line per task, solid = 8K, dashed = 16K (paper Fig. 2 left)
  results/fig2_flips.csv         per length x config x task: samples that got worse / better than dense
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "/work/nvme/beqg/zyang21/blasst/results"
PAPER = {"8k": {"dense": 95.07, "blasst~50%": 94.70}, "16k": {"dense": 94.80, "blasst~50%": 94.61}}
PAPER_SPARSITY = {"8k": 49.70, "16k": 52.20}      # paper Table 6, calibrated 50% target (x position of the paper point)
TASK_COLORS = {"niah_multikey_1": "#2a78d6", "vt": "#eb6834", "fwe": "#1baf7a"}   # reference palette slots 1-3
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f"{R}/fig2_ruler_*.csv"))], ignore_index=True)
df = df.drop_duplicates(["length", "task", "row", "config"], keep="last")
order = sorted(df.config.unique(), key=lambda c: -1 if c == "dense" else int(c[3:]))
lengths = [L for L in ["4k", "8k", "16k", "32k"] if L in set(df.length)]

g = df.groupby(["length", "config", "task"]).agg(n=("score", "size"), score=("score", "mean"),
                                                  pct_skipped=("pct_skipped", "mean"),
                                                  seconds=("seconds", "mean")).reset_index()
rows = []
for L in lengths:
    for c in order:
        d = g[(g.length == L) & (g.config == c)]
        if d.empty:
            continue
        r = {"length": L, "config": c, "n_per_task": int(d.n.min()),
             "avg_score": d.score.mean(), "avg_pct_skipped": d.pct_skipped.mean(), "avg_seconds": d.seconds.mean()}
        for _, x in d.iterrows():
            r[f"score_{x.task}"] = x.score
            r[f"pct_skipped_{x.task}"] = x.pct_skipped
        rows.append(r)
table = pd.DataFrame(rows)

# Change in the 3-task average vs dense, with a 95% range from resampling ("paired bootstrap"):
# draw 50 of the 50 samples with replacement, 10,000 times, and see how much the change moves.
# Each sample keeps its own dense and BLASST scores together (paired), so only real changes count.
rng = np.random.default_rng(0)
delta, lo, hi = [], [], []
for _, r in table.iterrows():
    if r.config == "dense":
        delta.append(0.0), lo.append(0.0), hi.append(0.0)
        continue
    boot = np.zeros(10000)
    tasks = sorted(df[(df.length == r.length) & (df.config == r.config)].task.unique())
    for t in tasks:
        a = df[(df.length == r.length) & (df.task == t)].pivot_table(index="row", columns="config", values="score")
        diff = (a[r.config] - a["dense"]).dropna().to_numpy()            # one number per sample
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        boot += diff[idx].mean(1) / len(tasks)
    delta.append(r.avg_score - table[(table.length == r.length) & (table.config == "dense")].avg_score.iloc[0])
    lo.append(np.percentile(boot, 2.5)), hi.append(np.percentile(boot, 97.5))
table.insert(4, "change_vs_dense", delta)
table.insert(5, "change_95pct_low", lo)
table.insert(6, "change_95pct_high", hi)
table.to_csv(f"{R}/fig2_table.csv", index=False, float_format="%.2f")

# per-sample changes vs dense
flips = []
dense = df[df.config == "dense"].set_index(["length", "task", "row"]).score
for (L, c, t), d in df[df.config != "dense"].groupby(["length", "config", "task"]):
    d = d.set_index(["length", "task", "row"]).score
    common = d.index.intersection(dense.index)
    diff = d.loc[common] - dense.loc[common]
    flips.append({"length": L, "config": c, "task": t, "n": len(common),
                  "worse": int((diff < 0).sum()), "better": int((diff > 0).sum())})
pd.DataFrame(flips).to_csv(f"{R}/fig2_flips.csv", index=False)

# ----------------------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(7.8, 3.9), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK2)
ax.tick_params(colors=INK2, labelsize=9)
styles = {"8k": ("-", "o"), "16k": ("--", "s"), "4k": (":", "^"), "32k": ("-.", "D")}
for t, col in TASK_COLORS.items():
    for L in lengths:
        d = g[(g.length == L) & (g.task == t)].copy()
        if d.empty or "dense" not in set(d.config):
            continue
        base = d[d.config == "dense"].score.iloc[0]
        d["rel"] = 100 * d.score / base if base > 0 else float("nan")
        d["k"] = d.config.map(lambda c: -1 if c == "dense" else int(c[3:]))
        d = d.sort_values("k")
        ls, mk = styles[L]
        ax.plot(d.pct_skipped, d.rel, ls, color=col, linewidth=1.2, marker=mk, ms=5, mfc=col, mec=SURFACE, mew=1.0,
                alpha=0.75, label=f"{t} {L.upper()}")
# the 3-task average (the number compared with the paper), drawn on top in black
for L in lengths:
    d = table[table.length == L]
    base = d[d.config == "dense"].avg_score.iloc[0]
    ls, mk = styles[L]
    ax.plot(d.avg_pct_skipped, 100 * d.avg_score / base, ls, color=INK, linewidth=2.0, marker=mk, ms=6.5,
            mfc=INK, mec=SURFACE, mew=1.2, zorder=6, label=f"average of 3 tasks {L.upper()}")
    if L in PAPER_SPARSITY:                                     # paper Table 3 at about 50% skipped
        ax.plot(PAPER_SPARSITY[L], 100 * PAPER[L]["blasst~50%"] / PAPER[L]["dense"], mk, ms=8, mfc=SURFACE,
                mec=INK, mew=1.6, zorder=7)
ax.plot([], [], "o", ms=8, mfc=SURFACE, mec=INK, mew=1.6, label="paper Table 3 (13-task avg),\ncircle 8K, square 16K")
ax.axhline(100, color=INK2, linewidth=0.8)
ax.set_xlim(-2, 100)
ax.set_xlabel("achieved % of causal tiles skipped (prefill)", color=INK, fontsize=10)
ax.set_ylabel("RULER score relative to dense (%)", color=INK, fontsize=10)
ax.set_title("RULER accuracy with skipping on, Llama-3.1-8B-Instruct, 50 samples per task\n"
             "points: dense, then calibrated 50 / 75 / 90% targets; solid = 8K, dashed = 16K",
             color=INK, fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.01, 1.0))
fig.tight_layout()
fig.savefig(f"{R}/fig2_accuracy_vs_sparsity.png", facecolor=SURFACE)

pd.set_option("display.width", 220)
print(table.round(2).to_string(index=False))
print("\npaper Table 3 (13-task RULER average):", PAPER)
print(pd.DataFrame(flips).to_string(index=False))
