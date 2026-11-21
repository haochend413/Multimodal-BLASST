"""
Correctness check for Qwen2.5-VL: our attention function with skipping OFF vs PyTorch's standard attention
(sdpa), language model only (the vision encoder is sdpa in both).

For every prompt: the logits of the four answer letters A-D at the last position, under both.
For one 8K prompt: the top-1 token at each of the last 512 positions under both, and the largest logit
difference there. A real bug changes many positions; bf16 rounding only flips near-ties.

  python vlm_check.py      -> results/vlm_regions/check.json
"""
import glob
import json

import torch

import blasst_attn as ba
from vlm_prepare import MODEL, OUT as INPUTS, make_inputs, processor
from vlm_regions import ALL_SDPA, TEXT_ONLY_OURS, attn_names

OUT = "/work/nvme/beqg/zyang21/blasst/results/vlm_regions/check.json"


@torch.inference_mode()
def logits(model, inputs, keep):
    return model(**inputs, use_cache=False, logits_to_keep=keep).logits[0].float()


def main():
    from transformers import Qwen2_5_VLForConditionalGeneration
    proc = processor()
    tok = proc.tokenizer
    letters = [tok.convert_tokens_to_ids(c) for c in "ABCD"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation=TEXT_ONLY_OURS).eval()
    ba.STATE["collect_stats"] = False
    ba.set_threshold(None)
    ba.reset_records(record=False)
    paths = sorted(glob.glob(f"{INPUTS}/*_8k.pt")) + sorted(glob.glob(f"{INPUTS}/*_16k.pt"))
    report = {"prompts": []}
    for i, path in enumerate(paths):
        d = torch.load(path)
        inputs = make_inputs(proc, d["text"], d["frames"], d["fps"]).to("cuda")
        keep = 512 if i == 0 else 1
        model.set_attn_implementation(TEXT_ONLY_OURS)
        assert attn_names(model) == ("blasst", "sdpa")
        ours = logits(model, inputs, keep)
        model.set_attn_implementation(ALL_SDPA)
        assert attn_names(model) == ("sdpa", "sdpa")
        ref = logits(model, inputs, keep)
        r = {"prompt": path.split("/")[-1][:-3],
             "letters_ours": [round(x, 3) for x in ours[-1, letters].tolist()],
             "letters_sdpa": [round(x, 3) for x in ref[-1, letters].tolist()],
             "max_abs_diff_last": (ours[-1] - ref[-1]).abs().max().item()}
        if keep > 1:
            same = (ours.argmax(-1) == ref.argmax(-1))
            top2 = ref.topk(2, dim=-1).values
            r["positions_compared"] = keep
            r["top1_agree"] = int(same.sum())
            r["max_abs_diff_any_position"] = (ours - ref).abs().max().item()
            r["sdpa_top1_minus_top2_where_disagree"] = [round(x, 3) for x in (top2[~same, 0] - top2[~same, 1]).tolist()]
        report["prompts"].append(r)
        print(r, flush=True)
    json.dump(report, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
