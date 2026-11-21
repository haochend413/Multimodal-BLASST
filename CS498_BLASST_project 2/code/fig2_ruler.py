"""
Step 7 / Fig. 2: RULER accuracy, dense vs calibrated BLASST, skipping actually ON during prefill.

  config 'dense'  : attn_implementation 'sdpa' (PyTorch standard attention), no skipping
  config 'cal50'  : our function, lambda = a / L with a fitted for a 50% target (results/calibration.json)
  config 'cal75'  : same, 75% target          (any 'calNN' with NN in calibration.json works)
Decode always uses standard attention (paper Table 3 is prefill-only sparsity).

Per sample: greedy answer, RULER score (string_match_all), % of causal tiles skipped in that
prefill (measured live), seconds. Rows 0..n-1 of each task (calibration used rows >= 100).

  python fig2_ruler.py --length 8k --tasks niah_multikey_1 vt fwe --n 50 --configs dense cal50 cal75
Appends to results/fig2_ruler.csv; re-running skips (length, task, row, config) already done.
"""
import argparse
import csv
import json
import os
import time

import torch

import blasst_attn as ba
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--length", required=True)
ap.add_argument("--tasks", nargs="+", default=["niah_multikey_1", "vt", "fwe"])
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--configs", nargs="+", default=["dense", "cal50", "cal75"])
ap.add_argument("--fit", default="linear_fit", help="which fit in calibration.json")
ap.add_argument("--out", default=f"{C.RESULTS}/fig2_ruler.csv")
args = ap.parse_args()

cal = None
if any(c != "dense" for c in args.configs):        # dense runs do not need the calibration
    cal = json.load(open(f"{C.RESULTS}/calibration.json"))[args.fit]["a"]

FIELDS = ["length", "task", "row", "index", "n_tokens", "config", "a", "lambda", "pct_skipped",
          "score", "seconds", "prediction", "answers"]
done = set()
if os.path.exists(args.out):
    with open(args.out) as fh:
        for r in csv.DictReader(fh):
            done.add((r["length"], r["task"], int(r["row"]), r["config"]))
new = not os.path.exists(args.out)
fh = open(args.out, "a", newline="")
w = csv.DictWriter(fh, fieldnames=FIELDS)
if new:
    w.writeheader()

tok, model = C.load_model("sdpa")
data = {t: C.load_ruler(t, args.length, args.n) for t in args.tasks}

for config in args.configs:
    if config == "dense":
        model.set_attn_implementation("sdpa")
        a = None
    else:
        a = cal[str(int(config[3:]) / 100)]
        ba.use_blasst(model)
        ba.set_threshold(a=a)
    for task in args.tasks:
        n_gen = C.GEN_TOKENS[C.family(task)]
        scores, sps = [], []
        for s in data[task]:
            if (args.length, task, s["row"], config) in done:
                continue
            ids = C.encode(tok, s["input"])
            S = ids.shape[1]
            ba.reset_records(record=False)
            t0 = time.perf_counter()
            pred = C.generate_answer(tok, model, ids, n_gen)
            secs = time.perf_counter() - t0
            sp = 100 * ba.sparsity_from_stats()["overall"] if a is not None else 0.0
            score = C.string_match_all(pred, s["answers"])
            w.writerow({"length": args.length, "task": task, "row": s["row"], "index": s["index"], "n_tokens": S,
                        "config": config, "a": a if a is not None else "", "lambda": min(a / S, 1.0) if a else "",
                        "pct_skipped": f"{sp:.3f}", "score": score, "seconds": f"{secs:.2f}",
                        "prediction": pred, "answers": json.dumps(s["answers"])})
            fh.flush()
            scores.append(score)
            sps.append(sp)
        if scores:
            print(f"{args.length:>4s} {config:6s} {task:16s} n={len(scores):3d}  score {sum(scores)/len(scores):6.2f}  "
                  f"% skipped {sum(sps)/len(sps):5.1f}", flush=True)
    ba.set_threshold(None)
fh.close()
print("done")
