"""
Step 5 / Fig. 1: % of tiles skipped vs lambda at several context lengths.

One prefill per sample with skipping OFF records the margin of every tile; % skipped
for any lambda is then computed offline (tile skipped iff margin < ln lambda).
Same measurement as NVIDIA's calibration: sparsity on dense activations.

Outputs (under results/):
  fig1_sparsity_vs_lambda.csv   one row per (sample, lambda): % of causal tiles skipped
  fig1_per_layer.csv            one row per (sample, layer, lambda in {1e-3, 3e-3})
  margins/<length>/<task>_row<r>.pt   fp16 margins [layers, heads, q tiles, k tiles] (hand-off to B, C)
  qk/<length>/<task>_row<r>.pt        Q (post-RoPE) and K (pre-GQA-repeat), bf16, all layers (hand-off to B)

  python fig1_sweep.py --lengths 4k 8k 16k --per_task 6
Re-running skips samples whose margin file already exists.
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch

import blasst_attn as ba
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--lengths", nargs="+", default=["4k", "8k", "16k"])
ap.add_argument("--tasks", nargs="+", default=["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "vt", "fwe"])
ap.add_argument("--per_task", type=int, default=6)
ap.add_argument("--per_task_32k", type=int, default=3)
ap.add_argument("--row_offset", type=int, default=100,
                help="first dataset row; rows 0-49 are kept for the Fig. 2 accuracy test")
ap.add_argument("--qk", nargs="*", default=["8k:1", "16k:1"],
                help="'<length>:<rows per task>' for which Q/K are dumped; 16k only for the first two tasks")
args = ap.parse_args()

LAMS = np.logspace(-6, 0, 121).tolist()          # 1e-6 .. 1, 20 points per decade
qk_rows = {s.split(":")[0]: int(s.split(":")[1]) for s in args.qk}

os.makedirs(C.RESULTS, exist_ok=True)
curve_csv = f"{C.RESULTS}/fig1_sparsity_vs_lambda.csv"
layer_csv = f"{C.RESULTS}/fig1_per_layer.csv"
new_curve = not os.path.exists(curve_csv)
new_layer = not os.path.exists(layer_csv)
fc, fl = open(curve_csv, "a", newline=""), open(layer_csv, "a", newline="")
wc, wl = csv.writer(fc), csv.writer(fl)
if new_curve:
    wc.writerow(["length", "task", "row", "index", "n_tokens", "lam", "pct_skipped"])
if new_layer:
    wl.writerow(["length", "task", "row", "n_tokens", "layer", "lam", "pct_skipped"])

tok, model = C.load_model("blasst")
ba.set_threshold(None)                           # skipping OFF: we only record margins
ba.STATE["collect_stats"] = False
C.timed_prefill(model, torch.randint(0, 1000, (1, 512), device="cuda"))   # warm-up

for length in args.lengths:
    n = args.per_task_32k if length == "32k" else args.per_task
    for t_i, task in enumerate(args.tasks):
        for i, sample in enumerate(C.load_ruler(task, length, n, start=args.row_offset)):
            row = sample["row"]
            mpath = f"{C.RESULTS}/margins/{length}/{task}_row{row}.pt"
            if os.path.exists(mpath):
                continue
            dump_qk = i < qk_rows.get(length, 0) and (length != "16k" or t_i < 2)
            ba.reset_records(record=True, dump_qk_layers=list(range(32)) if dump_qk else [])
            ids = C.encode(tok, sample["input"])
            S = ids.shape[1]
            _, secs = C.timed_prefill(model, ids)
            margins, causal = C.stack_margins()

            curve = C.sparsity_curve(margins, causal, LAMS)
            for lam, sp in zip(LAMS, curve):
                wc.writerow([length, task, row, sample["index"], S, f"{lam:.6g}", f"{100*sp:.4f}"])
            for lam in (1e-3, 3e-3):
                for layer, sp in enumerate(C.per_layer_sparsity(margins, causal, lam)):
                    wl.writerow([length, task, row, S, layer, lam, f"{100*sp:.4f}"])
            fc.flush(); fl.flush()

            C.save_margins(mpath, margins, sample, S)
            if dump_qk:
                qpath = f"{C.RESULTS}/qk/{length}/{task}_row{row}.pt"
                os.makedirs(os.path.dirname(qpath), exist_ok=True)
                torch.save({"qk": ba.STATE["qk"], "task": task, "index": sample["index"], "n_tokens": S,
                            "note": "per layer: (q [1, 32, S, 128] post-RoPE, k [1, 8, S, 128] post-RoPE, "
                                    "before GQA repeat; query head h uses KV head h // 4)"}, qpath)
            i1 = LAMS.index(min(LAMS, key=lambda x: abs(math.log(x / 1e-3))))
            print(f"{length:>4s} {task:16s} row {row}: {S:6d} tokens, prefill {secs:6.2f} s, "
                  f"% skipped at 1e-3 = {100*curve[i1]:5.2f}{'  (Q/K saved)' if dump_qk else ''}", flush=True)
            ba.reset_records(record=False, dump_qk_layers=[])

fc.close(); fl.close()
print("done")
