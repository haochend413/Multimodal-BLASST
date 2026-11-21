"""
Step 4 smoke test on one RULER sample (default: niah_multikey_2 at 4K, row 0).

  1. dense reference (attn 'sdpa'): greedy answer, prefill time
  2. ours, skipping OFF, recording margins: last-token logits vs sdpa; % skipped at several lambdas
     from the margins (paper Table 6 at 4K: 23.09% at 1e-3, 42.35% at 3e-3)
  3. ours, skipping OFF: greedy answer, should equal sdpa's
  4. ours, skipping ON at lambda = 1e-3: % skipped measured live, and the answer

  python smoke_test.py [--task vt] [--length 8k] [--row 3]
Writes results/smoke_<task>_<length>_row<row>.json
"""
import argparse
import json
import os

import torch

import blasst_attn as ba
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="niah_multikey_2")
ap.add_argument("--length", default="4k")
ap.add_argument("--row", type=int, default=0)
args = ap.parse_args()

tok, model = C.load_model("sdpa")
sample = C.load_ruler(args.task, args.length)[args.row]
ids = C.encode(tok, sample["input"])
S = ids.shape[1]
n_gen = C.GEN_TOKENS[C.family(args.task)]
print(f"sample {args.task}_{args.length} row {args.row}: {S} tokens (dataset says {sample['length']}), "
      f"answers {sample['answers']}")
print("prompt ends with:", repr(sample["input"][-200:]))

# warm-up: the first CUDA calls pay one-time setup costs that should not count as prefill time
C.timed_prefill(model, ids[:, :512])

# 1. dense reference
logits_sdpa, t_sdpa = C.timed_prefill(model, ids)
ans_sdpa = C.generate_answer(tok, model, ids, n_gen)

# 2. ours, skipping off, record margins
ba.use_blasst(model)
ba.set_threshold(None)
ba.reset_records(record=False)
C.timed_prefill(model, ids[:, :512])                  # warm-up for our function too
_, t_ours = C.timed_prefill(model, ids)               # time without recording
ba.reset_records(record=True)
logits_ours, t_ours_rec = C.timed_prefill(model, ids)
margins, causal = C.stack_margins()                   # [32 layers, 32 heads, q tiles, k tiles]
lams = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
curve = C.sparsity_curve(margins, causal, lams)
per_layer = C.per_layer_sparsity(margins, causal, 1e-3)
logit_err = (logits_ours - logits_sdpa).abs().max().item()
same_top1 = int(logits_ours.argmax()) == int(logits_sdpa.argmax())

# 3. ours, skipping off: answer
ba.reset_records(record=False)
ans_ours = C.generate_answer(tok, model, ids, n_gen)

# 4. ours, skipping on at lambda = 1e-3
ba.set_threshold(1e-3)
ba.reset_records(record=False)
ans_skip = C.generate_answer(tok, model, ids, n_gen)
online = ba.sparsity_from_stats()["overall"]          # decode steps use SDPA and add nothing here
ba.set_threshold(None)

res = {
    "task": args.task, "length": args.length, "row": args.row, "n_tokens": S,
    "margins_shape": list(margins.shape), "causal_tiles_per_head": int(causal.sum()),
    "prefill_seconds": {"sdpa": t_sdpa, "blasst_no_record": t_ours, "blasst_record": t_ours_rec},
    "last_logits_max_abs_diff_vs_sdpa": logit_err, "same_top1_token": same_top1,
    "pct_skipped_offline": {f"{l:g}": 100 * c for l, c in zip(lams, curve)},
    "pct_skipped_online_lambda_1e-3": 100 * online,
    "per_layer_pct_skipped_1e-3": [round(100 * x, 2) for x in per_layer],
    "answers": {"reference": sample["answers"], "sdpa": ans_sdpa, "blasst_no_skip": ans_ours,
                "blasst_lambda_1e-3": ans_skip},
    "score": {"sdpa": C.string_match_all(ans_sdpa, sample["answers"]),
              "blasst_no_skip": C.string_match_all(ans_ours, sample["answers"]),
              "blasst_lambda_1e-3": C.string_match_all(ans_skip, sample["answers"])},
    "answer_identical_no_skip_vs_sdpa": ans_ours == ans_sdpa,
    "peak_gpu_memory_GB": torch.cuda.max_memory_allocated() / 1e9,
}
os.makedirs(C.RESULTS, exist_ok=True)
out = f"{C.RESULTS}/smoke_{args.task}_{args.length}_row{args.row}.json"
with open(out, "w") as f:
    json.dump(res, f, indent=2)

print(f"\nprefill time: sdpa {t_sdpa:.3f} s | blasst {t_ours:.3f} s | blasst + recording {t_ours_rec:.3f} s")
print(f"last-token logits: max |blasst - sdpa| = {logit_err:.3f}, same top-1 token: {same_top1}")
print("% skipped (offline, from margins):", {k: round(v, 2) for k, v in res["pct_skipped_offline"].items()})
print(f"% skipped online at lambda=1e-3: {100*online:.2f}  (paper Table 6, 4K: 23.09)")
print("per-layer % skipped at 1e-3:", res["per_layer_pct_skipped_1e-3"])
for k, v in res["answers"].items():
    print(f"  {k:20s} {v!r}")
print("scores:", res["score"], "| no-skip answer identical to sdpa:", res["answer_identical_no_skip_vs_sdpa"])
print(f"peak GPU memory {res['peak_gpu_memory_GB']:.1f} GB; wrote {out}")
