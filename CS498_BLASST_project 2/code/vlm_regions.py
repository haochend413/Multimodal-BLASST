"""
Step 1 of the VLM region study, GPU part: one dense prefill per video prompt with tile margins recorded.

Model: Qwen2.5-VL-7B-Instruct, bf16. Our attention function ("blasst") runs ONLY in the language model;
the vision encoder stays on PyTorch's standard attention (sdpa). Skipping is off, so the run is dense;
the margins give the skip decision for every lambda afterwards (same trick as Fig. 1).

  python vlm_regions.py            # all prompts in vlm_inputs/, correctness check on the first one
Writes vlm_margins/<videoID>_<8k|16k>.pt (fp16 margins [28 layers, 28 heads, q tiles, k tiles], the
video-token flag per position, prompt facts) and appends one line per prompt to vlm_margins/runs.jsonl.
"""
import glob
import json
import os
import time

import torch

import blasst_attn as ba                     # registers attn_implementation "blasst"
import common as C
from vlm_prepare import MODEL, OUT as INPUTS, VIDEO_PAD, make_inputs, processor

MARGINS = "/work/nvme/beqg/zyang21/blasst/vlm_margins"
TEXT_ONLY_OURS = {"text_config": "blasst", "vision_config": "sdpa"}
ALL_SDPA = {"text_config": "sdpa", "vision_config": "sdpa"}


def attn_names(model):
    return (model.model.language_model.config._attn_implementation,
            model.model.visual.config._attn_implementation)


@torch.inference_mode()
def last_logits(model, inputs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    torch.cuda.synchronize()
    return logits, time.perf_counter() - t0


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration
    os.makedirs(MARGINS, exist_ok=True)
    proc = processor()
    tok = proc.tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation=TEXT_ONLY_OURS).eval()
    print("attention (language model, vision encoder):", attn_names(model), flush=True)
    assert attn_names(model) == ("blasst", "sdpa")
    ba.STATE["collect_stats"] = False            # we only need the margins
    ba.set_threshold(None)                       # skipping off: dense

    paths = sorted(glob.glob(f"{INPUTS}/*_8k.pt")) + sorted(glob.glob(f"{INPUTS}/*_16k.pt"))
    for i, path in enumerate(paths):
        d = torch.load(path)
        meta = d["meta"]
        inputs = make_inputs(proc, d["text"], d["frames"], d["fps"]).to("cuda")
        ids = inputs.input_ids[0]
        is_vision = ids == VIDEO_PAD
        assert int(is_vision.sum()) == meta["video_tokens"] and ids.numel() == meta["L"]
        rec = {"videoID": meta["videoID"], "length_name": meta["length_name"], "L": meta["L"]}

        if i == 0:   # correctness: skipping off must equal standard attention
            ba.reset_records(record=False)
            ours, _ = last_logits(model, inputs)
            model.set_attn_implementation(ALL_SDPA)
            assert attn_names(model) == ("sdpa", "sdpa")
            ref, t_sdpa = last_logits(model, inputs)
            model.set_attn_implementation(TEXT_ONLY_OURS)
            assert attn_names(model) == ("blasst", "sdpa")
            rec["check"] = {"max_abs_logit_diff": (ours - ref).abs().max().item(),
                            "top1_ours": tok.decode(ours.argmax()), "top1_sdpa": tok.decode(ref.argmax()),
                            "same_top1": bool(ours.argmax() == ref.argmax()), "seconds_sdpa": round(t_sdpa, 2)}
            print("correctness check:", rec["check"], flush=True)

        ba.reset_records(record=True)
        logits, secs = last_logits(model, inputs)
        margins, causal = C.stack_margins()                               # [28, 28, nq, nk], [nq, nk]
        ba.reset_records(record=False)
        nq = (meta["L"] + ba.TILE - 1) // ba.TILE
        assert margins.shape == (28, 28, nq, nq), margins.shape
        answer = tok.decode(logits.argmax()).strip()
        rec.update({"seconds_ours_recording": round(secs, 2), "answer_top1": answer, "answer_true": meta["answer"],
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)})
        torch.save({"margins": margins.half(), "is_vision": is_vision.cpu(), "input_ids": ids.cpu(),
                    "meta": meta, "tile": ba.TILE,
                    "note": "skip tile at lambda iff margin < ln(lambda); non-causal tiles are -inf; "
                            "is_vision = input_ids == 151656 (<|video_pad|>)"},
                   f"{MARGINS}/{meta['videoID']}_{meta['length_name']}.pt")
        with open(f"{MARGINS}/runs.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
        del inputs, logits, margins
        torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
