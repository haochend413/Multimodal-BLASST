"""
VLM study, second batch, items (1) and (2), GPU: how much attention WEIGHT the skipped tiles carry.

For every prompt, two dense passes with blasst_regions (nothing is skipped in either):
  pass 1  record the tile margins -> the prompt's own lambda that skips exactly 50% (and 75%) of all causal
          tiles (the same choice as step 1)
  pass 2  probe at those lambdas: for every (layer, head, query row), the share of the row's softmax weight
          on the tiles that lambda would skip, split into video keys and other keys, plus the row's dense
          weight on video keys
Qwen2.5-VL-7B: layout A (step 1's prompts, vlm_inputs/), B and C (vlm2_inputs/layout_B|C/).
Llama-3.1-8B-Instruct: RULER row 100 of niah_multikey_1/2/3, vt, fwe at 8K and 16K; also probed at the
calibrated lambda = a / L of the baseline (a = 106.5 for 50%, 846 for 75%).

  python vlm2_weights.py --model qwen [--limit 1]
  python vlm2_weights.py --model llama [--limit 1]
Writes vlm2_tensors/<set>/<prompt>.pt (margins fp16 + probe arrays fp16) and vlm2_tensors/runs_<model>.jsonl.
"""
import argparse
import glob
import json
import math
import os
import time

import torch

import blasst_regions as br

W = "/work/nvme/beqg/zyang21/blasst"
TENS = f"{W}/vlm2_tensors"
VIDEO_PAD = 151656
CAL_A = {"cal50": 106.5, "cal75": 846.0}            # results/calibration.json, log_fit (Llama baseline)


def exact_lambda(margins, frac):
    """lambda that skips exactly round(frac * N) of the causal tiles (midpoint between neighbours)."""
    causal = torch.isfinite(margins[0, 0])
    v = margins[:, :, causal].flatten().float().sort().values
    k = int(round(frac * v.numel()))
    return math.exp(min(float((v[k - 1] + v[k]) / 2), 0.0))


@torch.inference_mode()
def two_passes(model, inputs, is_vision, extra_lams=None):
    br.set_kinds(is_vision)
    br.set_table(None)
    br.set_probe()
    br.reset(record=True)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model(**inputs, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    margins = br.get_margins()
    lams = {"exact50": exact_lambda(margins, 0.5), "exact75": exact_lambda(margins, 0.75), **(extra_lams or {})}
    br.set_probe(list(lams.values()))
    br.reset(record=False)
    model(**inputs, use_cache=False, logits_to_keep=1)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    pr = br.STATE["probe"]
    layers = sorted(pr)
    out = {"margins": margins.half(), "lams": lams, "is_vision": is_vision.cpu(),
           "dense_vid": torch.stack([pr[l]["dense_vid"] for l in layers]),          # [L, H, S]
           "lost_vid": torch.stack([pr[l]["lost_vid"] for l in layers], dim=1),     # [n_lams, L, H, S]
           "lost_other": torch.stack([pr[l]["lost_other"] for l in layers], dim=1)}
    br.set_probe()
    br.reset()
    return out, {"seconds_margins": round(t1 - t0, 2), "seconds_probe": round(t2 - t1, 2)}


def qwen_prompts():
    for layout, folder in (("A", f"{W}/vlm_inputs"), ("B", f"{W}/vlm2_inputs/layout_B"), ("C", f"{W}/vlm2_inputs/layout_C")):
        for p in sorted(glob.glob(f"{folder}/*_8k.pt")) + sorted(glob.glob(f"{folder}/*_16k.pt")):
            yield f"qwen_{layout}", os.path.basename(p)[:-3], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "llama"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="only the first N prompts (debugging)")
    args = ap.parse_args()
    runs = f"{TENS}/runs_{args.model}.jsonl"
    os.makedirs(TENS, exist_ok=True)
    done = {json.loads(l)["name"] for l in open(runs)} if os.path.exists(runs) else set()

    if args.model == "qwen":
        from transformers import Qwen2_5_VLForConditionalGeneration
        from vlm_prepare import MODEL, make_inputs, processor
        proc = processor()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation={"text_config": "blasst_regions", "vision_config": "sdpa"}).eval()
        assert model.model.language_model.config._attn_implementation == "blasst_regions"
        assert model.model.visual.config._attn_implementation == "sdpa"
        items = list(qwen_prompts())
    else:
        import common as C
        tok, model = C.load_model("blasst_regions")
        items = [(f"llama_{L}", f"{t}_row100_{L}", (t, L)) for L in ("8k", "16k")
                 for t in ("niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "vt", "fwe")]
    if args.limit:
        items = items[:args.limit]

    for set_name, name, src in items:
        key = f"{set_name}/{name}"
        if key in done:
            continue
        if args.model == "qwen":
            d = torch.load(src)
            inputs = make_inputs(proc, d["text"], d["frames"], d["fps"]).to("cuda")
            ids = inputs.input_ids[0]
            is_vision = ids == VIDEO_PAD
            meta = d["meta"]
            extra = None
        else:
            s = C.load_ruler(src[0], src[1], n=1, start=100)[0]
            ids = C.encode(tok, s["input"])
            inputs = {"input_ids": ids}
            ids = ids[0]
            is_vision = torch.zeros_like(ids, dtype=torch.bool)
            meta = {"task": src[0], "length_name": src[1], "row": 100}
            extra = {k: min(a / ids.numel(), 1.0) for k, a in CAL_A.items()}
        res, timing = two_passes(model, inputs, is_vision, extra)
        res.update({"meta": meta, "input_ids": ids.cpu(), "set": set_name, "name": name})
        os.makedirs(f"{TENS}/{set_name}", exist_ok=True)
        torch.save(res, f"{TENS}/{set_name}/{name}.pt")
        lost = (res["lost_vid"].float() + res["lost_other"].float()).mean(dim=(1, 2, 3))     # per lambda
        rec = {"name": key, "L": int(ids.numel()), "video_tokens": int(is_vision.sum()), **timing,
               "lams": res["lams"], "mean_weight_lost_pct": dict(zip(res["lams"], [round(100 * x, 3) for x in lost.tolist()])),
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        with open(runs, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        del inputs, res
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
