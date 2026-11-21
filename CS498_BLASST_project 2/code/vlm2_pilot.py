"""
VLM study, second batch, item (3), GPU: Video-MME accuracy pilot with skipping ON during prefill.

Prompts: vlm2_inputs/pilot/ (30 short + medium videos, 3 questions each, [video][question], about 16K tokens).
Answer = the letter A-D with the highest logit at the last prompt position (the model is asked to reply with
the letter only), so no text generation is needed.

Settings, all with our attention function (blasst_regions) in the language model:
  dense    nothing skipped; this pass also records the margins that set the two lambdas below
  one50    one lambda for every tile, the prompt's own lambda that skips exactly 50% of all causal tiles
           (computed from its dense margins, as in step 1)
  protect  the question's view of the video is never skipped: every tile whose query block holds any text
           after the video (the question; ~100 tokens, so it shares a block with the last video tokens) and
           whose key block holds any video token. One shared lambda for every other tile, set from the dense
           margins so the total is still 50% of all causal tiles
% skipped is also measured live in the skipping runs (it can differ a little from the dense estimate).

Order: (a) dense on the CHECK set = question 1 of each of the 30 videos; stop unless at least 15 of 30 are
right (chance is 7.5); (b) all 90 questions, all three settings.

  python vlm2_pilot.py [--limit N]
Appends to results/vlm_regions/v2_pilot.jsonl (small; one line per question and setting; resumable).
"""
import argparse
import glob
import json
import math
import os
import time

import torch

import blasst_regions as br
from vlm2_weights import exact_lambda
from vlm_prepare import MODEL, VIDEO_PAD, make_inputs, processor

W = "/work/nvme/beqg/zyang21/blasst"
OUT = f"{W}/results/vlm_regions/v2_pilot.jsonl"
CHECK_MIN = 15


def question_view_kinds(is_vision):
    """Query side: a block holding any text that comes after the video is "text" (the question).
    Key side: a block holding any video token is "video". Tiles (text, video) = the question's view of the video."""
    kinds = br.block_kinds(is_vision)
    S, n = is_vision.numel(), len(kinds)
    after = torch.zeros(n * br.TILE, dtype=torch.bool)
    after[int(is_vision.nonzero().max()) + 1:S] = True
    has_vid = torch.nn.functional.pad(is_vision, (0, n * br.TILE - S)).view(n, br.TILE).any(1)
    q_kinds = torch.where(after.view(n, br.TILE).any(1), br.TEXT, kinds)
    k_kinds = torch.where(has_vid, br.VIDEO, br.TEXT)
    return q_kinds, k_kinds


def protect_lambda(margins, q_kinds, k_kinds, frac=0.5):
    """Never skip (text query, video key) tiles; one lambda for all other tiles so that frac of ALL causal tiles
    are skipped. Returns (lambda, reached, protected share of causal tiles in %)."""
    causal = torch.isfinite(margins[0, 0])
    tv = (q_kinds[:, None] == br.TEXT) & (k_kinds[None, :] == br.VIDEO)
    other = margins[:, :, causal & ~tv].flatten().float().sort().values
    k = int(round(frac * causal.sum().item() * margins.shape[0] * margins.shape[1]))
    n_neg = int((other < 0).sum())
    share = 100 * (causal & tv).sum().item() / causal.sum().item()
    if k > n_neg:                                   # cannot reach frac without lambda >= 1
        return 1.0, False, share
    return math.exp(min(float((other[k - 1] + other[k]) / 2), 0.0)), True, share


@torch.inference_mode()
def answer(model, inputs, letters):
    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    lg = logits[letters]
    return "ABCD"[int(lg.argmax())], [round(x, 3) for x in lg.tolist()], int(logits.argmax())


def region_pct(counts):
    sk, ca = counts["skipped"].double(), counts["causal"].double()
    tv = (100 * sk[br.TEXT, br.VIDEO] / ca[br.TEXT, br.VIDEO]).item() if ca[br.TEXT, br.VIDEO] > 0 else None
    vv = (100 * sk[br.VIDEO, br.VIDEO] / ca[br.VIDEO, br.VIDEO]).item()
    return 100 * (sk.sum() / ca.sum()).item(), tv, vv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N videos (debugging)")
    args = ap.parse_args()
    from transformers import Qwen2_5_VLForConditionalGeneration
    proc = processor()
    tok = proc.tokenizer
    letters = [tok.convert_tokens_to_ids(c) for c in "ABCD"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda",
        attn_implementation={"text_config": "blasst_regions", "vision_config": "sdpa"}).eval()
    assert model.model.language_model.config._attn_implementation == "blasst_regions"
    files = sorted(glob.glob(f"{W}/vlm2_inputs/pilot/*.pt"))[: args.limit]
    videos = {os.path.basename(f)[:-3]: torch.load(f) for f in files}
    done = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l)
            done[(r["question_id"], r["setting"])] = r

    def log(rec):
        done[(rec["question_id"], rec["setting"])] = rec
        with open(OUT, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(rec, flush=True)

    def run_question(vid, qi, settings):
        v, q = videos[vid], videos[vid]["questions"][qi]
        need = [s for s in settings if (q["question_id"], s) not in done]
        if not need:
            return
        inputs = make_inputs(proc, q["text"], v["frames"], v["fps"]).to("cuda")
        is_vision = inputs.input_ids[0] == VIDEO_PAD
        assert inputs.input_ids.shape[1] == q["L"]
        q_kinds, k_kinds = question_view_kinds(is_vision.cpu())
        br.set_kinds(is_vision, q_kinds=q_kinds, k_kinds=k_kinds)
        base = {"videoID": vid, "duration": v["meta"]["duration"], "question_id": q["question_id"],
                "task_type": q["task_type"], "truth": q["answer"], "L": q["L"]}
        dense = done.get((q["question_id"], "dense"))
        if dense is None:
            # dense pass: answer + margins; the margins set the lambdas of the two skipping settings,
            # which are stored in the dense record, so this pass runs once per question
            br.set_table(None); br.set_probe(); br.reset(record=True)
            t0 = time.perf_counter()
            pred, lg, top = answer(model, inputs, letters)
            secs = time.perf_counter() - t0
            m = br.get_margins()
            lam_p, reached, share = protect_lambda(m, q_kinds, k_kinds)
            dense = {**base, "setting": "dense", "pred": pred, "correct": pred == q["answer"], "letter_logits": lg,
                     "top1_is_letter": top in letters, "lambda_one50": exact_lambda(m, 0.5), "lambda_protect": lam_p,
                     "protect_reaches_50": reached, "protected_share_pct": round(share, 3), "seconds": round(secs, 2)}
            log(dense)
            br.reset()
        for s in need:
            if s == "dense":
                continue
            if s == "one50":
                br.set_table(default=dense["lambda_one50"])
            else:
                br.set_table({(br.TEXT, br.VIDEO): None}, default=dense["lambda_protect"])
            br.reset()
            t0 = time.perf_counter()
            pred, lg, top = answer(model, inputs, letters)
            secs = time.perf_counter() - t0
            overall, tv, vv = region_pct(br.region_counts())
            log({**base, "setting": s, "pred": pred, "correct": pred == q["answer"], "letter_logits": lg,
                 "top1_is_letter": top in letters, "pct_skipped_live": round(overall, 3),
                 "pct_skipped_TV_live": None if tv is None else round(tv, 3), "pct_skipped_VV_live": round(vv, 3),
                 "seconds": round(secs, 2)})
        br.set_table(None); br.reset()
        del inputs
        torch.cuda.empty_cache()

    # (a) check set: question 1 of every video, dense only
    for vid in videos:
        run_question(vid, 0, ["dense"])
    n_ok = sum(done[(videos[v]["questions"][0]["question_id"], "dense")]["correct"] for v in videos)
    status = f"check: dense {n_ok} of {len(videos)} correct (need {CHECK_MIN} of 30)"
    print(status, flush=True)
    if len(videos) == 30 and n_ok < CHECK_MIN:
        print("STOP: dense accuracy is not clearly above chance in this setting", flush=True)
        return
    # (b) all questions, all settings
    for vid in videos:
        for qi in range(len(videos[vid]["questions"])):
            run_question(vid, qi, ["dense", "one50", "protect"])
    print("done")


if __name__ == "__main__":
    main()
