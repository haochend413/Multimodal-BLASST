"""
Step 6: cross-check our % skipped against NVIDIA's reference (nvidia-modelopt 0.47.0,
method flash_skip_softmax, PyTorch backend) on the SAME RULER samples.

For each sample:
  ours     : one prefill with skipping off, margins -> % skipped at each lambda
             (a) our rule, (b) NVIDIA's padding convention: the last query tile, if it is
             partial, is never skipped (their padded dummy rows always vote "keep")
  modelopt : eager attention with their softmax patch; the first threshold is 1e-30 so the
             applied mask skips nothing (dense forward, same as ours); the others are measured.

  python crosscheck_modelopt.py niah_multikey_2:4k:0 fwe:4k:0 vt:8k:0
Writes results/crosscheck_modelopt.json
"""
import json
import math
import sys

import torch

import blasst_attn as ba
import common as C

LAMS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
specs = sys.argv[1:] or ["niah_multikey_2:4k:0"]
samples = []
for s in specs:
    task, length, row = s.split(":")
    samples.append(C.load_ruler(task, length, int(row) + 1)[int(row)])

tok, model = C.load_model("blasst")
ba.set_threshold(None)
ba.STATE["collect_stats"] = False
results = []
for smp in samples:
    ids = C.encode(tok, smp["input"])
    ba.reset_records(record=True)
    C.timed_prefill(model, ids)
    margins, causal = C.stack_margins()
    ours = C.sparsity_curve(margins, causal, LAMS)
    # NVIDIA convention: if the last query tile is partial, all of its tiles are kept
    S = ids.shape[1]
    m2 = margins.clone()
    if S % ba.TILE:
        m2[:, :, -1, :] = 0.0
    ours_pad = C.sparsity_curve(m2, causal, LAMS)
    results.append({"task": smp["task"], "length": smp["length_name"], "index": smp["index"], "n_tokens": S,
                    "ours": [100 * x for x in ours], "ours_nvidia_padding": [100 * x for x in ours_pad]})
    ba.reset_records(record=False)
    print(f"ours {smp['task']} {smp['length_name']} ({S} tokens):", " ".join(f"{100*x:6.2f}" for x in ours),
          "| nvidia padding:", " ".join(f"{100*x:6.2f}" for x in ours_pad), flush=True)

# ---- NVIDIA's reference on the same model and inputs
import modelopt.torch.sparsity.attention_sparsity as mtsa
from modelopt.torch.sparsity.attention_sparsity.utils import get_named_sparse_attention_modules

cfg = {"sparse_cfg": {
    "*attn*": {"method": "flash_skip_softmax", "thresholds": {"prefill": [1e-30] + LAMS, "decode": [1e-30] + LAMS},
               "br": 128, "bc": 128, "backend": "pytorch", "collect_stats": True, "enable": True},
    "default": {"enable": False}}}
model = mtsa.sparsify(model, cfg)
print("attn implementation after sparsify:", model.config._attn_implementation)
mods = get_named_sparse_attention_modules(model)
print("sparse attention modules:", len(mods))

for smp, res in zip(samples, results):
    for _, m in mods:
        m._stats_manager.reset()
    ids = C.encode(tok, smp["input"])
    with torch.no_grad():
        model(ids, logits_to_keep=1, use_cache=False)
    per_layer = [m.get_stats()["average_sparsity"] for _, m in mods]     # [layers][thresholds]
    avg = [100 * sum(pl[i] for pl in per_layer) / len(per_layer) for i in range(len(LAMS) + 1)]
    res["modelopt"] = avg[1:]
    res["modelopt_applied_threshold_1e-30"] = avg[0]

with open(f"{C.RESULTS}/crosscheck_modelopt.json", "w") as f:
    json.dump({"lambdas": LAMS, "samples": results}, f, indent=2)

print("\n% of causal tiles skipped (dense activations). lambdas:", LAMS)
for r in results:
    print(f"\n{r['task']} {r['length']} ({r['n_tokens']} tokens)")
    for k in ("ours", "ours_nvidia_padding", "modelopt"):
        print(f"  {k:22s}", " ".join(f"{x:6.2f}" for x in r[k]))
    print(f"  modelopt applied mask at 1e-30 skipped {r['modelopt_applied_threshold_1e-30']:.3f}% (should be 0)")
