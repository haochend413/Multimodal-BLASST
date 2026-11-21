"""
Fig. 1 analysis (CPU only). Reads what fig1_sweep.py wrote and produces:

  results/fig1_sparsity_vs_lambda.png   % skipped vs lambda, one line per length, paper Table 6 as hollow markers
  results/fig1_collapse_lambdaL.png     % skipped vs lambda * L, with the fitted calibration curve
  results/fig1_table.csv                per length: fixed lambda 1e-3 / 3e-3 and calibrated 50% / 75%, vs paper
  results/fig1_per_task.csv             per length and task: % skipped at 1e-3 and 3e-3
  results/calibration.json              fitted alpha, beta and the a for each target (used by Fig. 2)

Calibration = BLASST Algorithm 2 as NVIDIA implements it (modelopt calibrator):
  points (lambda * L, s) from every sample and lambda, keep 0.10 <= s <= 0.90,
  fit lambda * L = alpha * exp(beta * s) by least squares, then a(target) = alpha * exp(beta * target)
  and lambda = a / L at run time. All % skipped numbers below are recomputed exactly from the
  saved margins (tile skipped iff margin < ln lambda).
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
from scipy.optimize import curve_fit

R = "/work/nvme/beqg/zyang21/blasst/results"
LENGTHS = ["4k", "8k", "16k", "32k"]
PAPER_FIXED = {1e-3: {"4k": 23.09, "8k": 37.92, "16k": 52.38, "32k": 65.72},
               3e-3: {"4k": 42.35, "8k": 57.54, "16k": 69.83, "32k": 79.36}}
PAPER_CAL = {0.5: {"4k": 54.20, "8k": 49.70, "16k": 52.20, "32k": 46.96},
             0.7: {"4k": 67.99, "8k": 74.65, "16k": 73.64, "32k": 72.54}}
TARGETS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

# reference palette (dataviz skill, light mode): categorical slots 1-4, text and grid tokens
COLORS = {"4k": "#2a78d6", "8k": "#eb6834", "16k": "#1baf7a", "32k": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


# ----------------------------------------------------------------------------- load
curve = pd.read_csv(f"{R}/fig1_sparsity_vs_lambda.csv")
curve = curve.drop_duplicates(["length", "task", "row", "lam"], keep="last")
lengths = [L for L in LENGTHS if L in set(curve["length"])]

# margins: one entry per sample -> sorted causal-tile margins (enough to get % skipped at any lambda)
samples = []
for L in lengths:
    for path in sorted(glob.glob(f"{R}/margins/{L}/*.pt")):
        d = torch.load(path)
        m = d["margins"].float()
        vals = m[torch.isfinite(m)].flatten().sort().values      # causal tiles only (non-causal are -inf)
        samples.append({"length": L, "task": d["task"], "row": d.get("row"), "n_tokens": d["n_tokens"], "vals": vals})


def pct_skipped(vals, lam):
    return 100.0 * torch.searchsorted(vals, math.log(lam)).item() / vals.numel()


# ----------------------------------------------------------------------------- calibration fit
pts = curve[(curve.pct_skipped >= 10) & (curve.pct_skipped <= 90)]
s_frac = pts.pct_skipped.values / 100
lamL = pts.lam.values * pts.n_tokens.values
f = lambda s, a, b: a * np.exp(b * s)
(alpha, beta), _ = curve_fit(f, s_frac, lamL, p0=[1.0, 5.0], bounds=([0, 0], [np.inf, 20]), maxfev=20000)
resid = lamL - f(s_frac, alpha, beta)
r2 = 1 - (resid ** 2).sum() / ((lamL - lamL.mean()) ** 2).sum()
# the same model fitted in log space (less dominated by the largest lambda * L values)
b_log, ln_a_log = np.polyfit(s_frac, np.log(lamL), 1)
alpha_log, beta_log = math.exp(ln_a_log), b_log
cal = {"model": "lambda * L = alpha * exp(beta * s)", "n_points": int(len(pts)),
       "linear_fit": {"alpha": alpha, "beta": beta, "r2": r2,
                      "a": {str(t): alpha * math.exp(beta * t) for t in TARGETS}},
       "log_fit": {"alpha": alpha_log, "beta": beta_log,
                   "a": {str(t): alpha_log * math.exp(beta_log * t) for t in TARGETS}}}

# ----------------------------------------------------------------------------- tables
rows = []
for L in lengths:
    S = [s for s in samples if s["length"] == L]
    r = {"length": L, "n_samples": len(S), "mean_tokens": round(float(np.mean([s["n_tokens"] for s in S])))}
    for lam in (1e-3, 3e-3):
        v = [pct_skipped(s["vals"], lam) for s in S]
        r[f"fixed_{lam:g}_mean"], r[f"fixed_{lam:g}_std"] = np.mean(v), np.std(v)
        r[f"fixed_{lam:g}_paper"] = PAPER_FIXED[lam].get(L)
    for fit in ("linear_fit", "log_fit"):
        for t in (0.5, 0.75):
            a = cal[fit]["a"][str(t)]
            v = [pct_skipped(s["vals"], min(a / s["n_tokens"], 1.0)) for s in S]
            r[f"cal{int(t*100)}_{fit}_mean"], r[f"cal{int(t*100)}_{fit}_std"] = np.mean(v), np.std(v)
    r["cal50_paper"] = PAPER_CAL[0.5].get(L)
    rows.append(r)
table = pd.DataFrame(rows)
table.to_csv(f"{R}/fig1_table.csv", index=False, float_format="%.2f")

pt = []
for s in samples:
    pt.append({"length": s["length"], "task": s["task"], "row": s["row"], "n_tokens": s["n_tokens"],
               "pct_1e-3": pct_skipped(s["vals"], 1e-3), "pct_3e-3": pct_skipped(s["vals"], 3e-3)})
per_task = pd.DataFrame(pt).groupby(["length", "task"], sort=False).agg(
    n=("row", "size"), pct_1e3=("pct_1e-3", "mean"), pct_3e3=("pct_3e-3", "mean")).reset_index()
per_task.to_csv(f"{R}/fig1_per_task.csv", index=False, float_format="%.2f")

# calibrated achieved sparsity, also store it in the json for the report
for fit in ("linear_fit", "log_fit"):
    cal[fit]["achieved_pct"] = {f"{int(t*100)}": {r["length"]: round(r[f"cal{int(t*100)}_{fit}_mean"], 2) for r in rows}
                                for t in (0.5, 0.75)}
with open(f"{R}/calibration.json", "w") as fh:
    json.dump(cal, fh, indent=2)

# ----------------------------------------------------------------------------- figure 1a
mean = curve.groupby(["length", "lam"]).pct_skipped.agg(["mean", "min", "max"]).reset_index()
fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
style(ax)
for L in lengths:
    d = mean[mean.length == L]
    ax.fill_between(d.lam, d["min"], d["max"], color=COLORS[L], alpha=0.12, linewidth=0)
    ax.plot(d.lam, d["mean"], color=COLORS[L], linewidth=1.6, label=f"{L} (ours)")
    for lam in (1e-3, 3e-3):
        if L in PAPER_FIXED[lam]:
            ax.plot(lam, PAPER_FIXED[lam][L], "o", ms=6.5, mfc=SURFACE, mec=COLORS[L], mew=1.6, zorder=5)
    # direct label at the right end of each line
# direct labels at the right end, nudged apart so they never overlap (min gap 4.5 points)
ends = sorted(((mean[mean.length == L]["mean"].iloc[-1], L) for L in lengths), reverse=True)
placed = []
for y, L in ends:
    y_lab = min(y, placed[-1] - 4.5) if placed else y
    placed.append(y_lab)
    ax.annotate(L, (1.0, y_lab), xytext=(5, 0), textcoords="offset points", color=INK, fontsize=8.5,
                va="center", annotation_clip=False)
ax.plot([], [], "o", ms=6.5, mfc=SURFACE, mec=INK2, mew=1.6, label="paper, Table 6")
ax.set_xscale("log")
ax.set_xlim(1e-6, 1)
ax.set_ylim(0, 100)
ax.set_xlabel("threshold lambda (log scale)", color=INK, fontsize=10)
ax.set_ylabel("% of causal tiles skipped", color=INK, fontsize=10)
ax.set_title("Llama-3.1-8B-Instruct, RULER prefill: tiles skipped vs lambda\n"
             "line = mean over samples, band = min to max", color=INK, fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
fig.tight_layout()
fig.savefig(f"{R}/fig1_sparsity_vs_lambda.png", facecolor=SURFACE)

# ----------------------------------------------------------------------------- figure 1b
fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=200)
fig.patch.set_facecolor(SURFACE)
style(ax)
tok = curve.groupby("length").n_tokens.mean()
for L in lengths:
    d = mean[mean.length == L]
    ax.plot(d.lam * tok[L], d["mean"], color=COLORS[L], linewidth=1.6, label=L)
s_grid = np.linspace(0.05, 0.95, 100)
ax.plot(alpha_log * np.exp(beta_log * s_grid), 100 * s_grid, "--", color=INK, linewidth=1.3,
        label=f"fit used (log space): lambda*L = {alpha_log:.3g} exp({beta_log:.2f} s)")
ax.plot(alpha * np.exp(beta * s_grid), 100 * s_grid, ":", color=INK2, linewidth=1.1,
        label=f"NVIDIA default fit (linear): {alpha:.3g} exp({beta:.2f} s)")
ax.set_xscale("log")
ax.set_ylim(0, 100)
ax.set_xlim(1e-3, 1e5)
ax.set_xlabel("lambda x L  (L = tokens in the sample; log scale)", color=INK, fontsize=10)
ax.set_ylabel("% of causal tiles skipped", color=INK, fontsize=10)
ax.set_title("Does lambda = a / L hold?  Overlapping curves = yes\n"
             "(mean over samples per length; fit on points with 10-90% skipped)", color=INK, fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left")
fig.tight_layout()
fig.savefig(f"{R}/fig1_collapse_lambdaL.png", facecolor=SURFACE)

# ----------------------------------------------------------------------------- print
pd.set_option("display.width", 200)
print(f"{len(samples)} samples; calibration points {len(pts)}")
print(f"linear fit: alpha={alpha:.4g} beta={beta:.3f} R2={r2:.3f}; a(50%)={cal['linear_fit']['a']['0.5']:.4g} "
      f"a(75%)={cal['linear_fit']['a']['0.75']:.4g}")
print(f"log fit:    alpha={alpha_log:.4g} beta={beta_log:.3f}; a(50%)={cal['log_fit']['a']['0.5']:.4g} "
      f"a(75%)={cal['log_fit']['a']['0.75']:.4g}")
print(table.round(2).T.to_string())
print(per_task.round(2).to_string(index=False))
