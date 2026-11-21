"""
VLM study, second batch (VLM_REGIONS_2.md), CPU only: build the prompts for the layout check and the
accuracy pilot. Reuses the helpers of vlm_prepare.py (frame decoding with PyAV, Qwen's resize rule).

Layout check: the same 5 videos and questions as step 1 (vlm_inputs/), at about 8K and 16K tokens, in
  layout B  [video] [subtitles] [question]   Video-MME's standard "with subtitles" order; the subtitle text
                                              is the same transcript text step 1 split around the video
  layout C  [video] [question]               no subtitles; more frames fill the length
  -> vlm2_inputs/layout_B/<video>_<8k|16k>.pt, vlm2_inputs/layout_C/<video>_<8k|16k>.pt

Accuracy pilot: 30 short and medium Video-MME videos (15 + 15, first in ID order among chunks 1-3), all 3
questions each (90 questions), layout C at about 16K tokens.
  -> vlm2_inputs/pilot/<video>.pt (frames once per video, the 3 question prompts), vlm2_inputs/pilot.csv

Question wording for B, C and the pilot is lmms-eval's Video-MME prompt.
"""
import glob
import os

import pandas as pd
import torch

from vlm_prepare import (MAX_PIXELS, PART1, PART2_MAX, TARGETS, VIDEO_PAD, VM, W, make_inputs, processor,
                         read_frames, smart_resize, srt_text, video_info)

OUT = f"{W}/blasst/vlm2_inputs"
STEP1 = f"{W}/blasst/vlm_inputs"
PILOT_PER_DURATION = 15


def question_text(q):
    return ("Select the best answer to the following multiple-choice question based on the video. "
            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
            + q.question + "\n" + "\n".join(q.options) + "\nThe best answer is:")


def build_text(proc, q, subtitles=None):
    body = question_text(q)
    if subtitles is not None:
        body = "This video's subtitles are listed below:\n" + subtitles + "\n" + body
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": body}]}]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def frames_for(proc, path, texts, target):
    """Decode as many evenly spaced frames as fit with the longest of `texts` under `target` tokens."""
    dur, h, w = video_info(path)
    H, Wd = smart_resize(h, w)
    tpg = (H // 28) * (Wd // 28)
    n_text = max(len(proc.tokenizer(t).input_ids) - 1 for t in texts)
    nframes = 2 * ((target - n_text) // tpg)
    frames, dur = read_frames(path, nframes, H, Wd)
    return frames, dur, {"nframes": nframes, "H": H, "W": Wd, "video_seconds": round(dur, 1), "source_hw": f"{h}x{w}"}


def check(proc, text, frames, fps, target):
    inp = make_inputs(proc, text, frames, fps)
    L, n_vid = inp.input_ids.shape[1], int((inp.input_ids == VIDEO_PAD).sum())
    assert n_vid == int(inp.video_grid_thw.prod(-1).sum()) // 4 and L <= target, (L, n_vid)
    return {"L": L, "video_tokens": n_vid, "text_tokens": L - n_vid}


def main():
    proc = processor()
    tok = proc.tokenizer
    qa = pd.read_parquet(f"{VM}/videomme/test-00000-of-00001.parquet")
    videos = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{VM}/**/*.mp4", recursive=True)}

    # ---------------- layout check: same 5 videos and first questions as step 1
    step1 = pd.read_csv(f"{STEP1}/prompts.csv")
    rows = []
    for vid in sorted(step1.videoID.unique()):
        q = qa[qa.videoID == vid].sort_values("question_id").iloc[0]
        ids = tok(srt_text(f"{VM}/subtitle/{vid}.srt")).input_ids
        subs = tok.decode(ids[:PART1 + PART2_MAX])                  # the transcript text step 1 used
        for layout, text in (("B", build_text(proc, q, subs)), ("C", build_text(proc, q))):
            os.makedirs(f"{OUT}/layout_{layout}", exist_ok=True)
            for name, target in TARGETS.items():
                frames, dur, meta = frames_for(proc, videos[vid], [text], target)
                fps = meta["nframes"] / dur
                meta.update(check(proc, text, frames, fps, target))
                meta.update({"videoID": vid, "layout": layout, "length_name": name, "question_id": q.question_id,
                             "answer": q.answer, "fps": round(fps, 4)})
                torch.save({"frames": frames, "text": text, "fps": fps, "meta": meta},
                           f"{OUT}/layout_{layout}/{vid}_{name}.pt")
                rows.append(meta)
                print(meta, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/layouts.csv", index=False)

    # ---------------- pilot: 15 short + 15 medium videos, all 3 questions, layout C at 16K
    os.makedirs(f"{OUT}/pilot", exist_ok=True)
    picked = []
    for duration in ("short", "medium"):
        ids_ = sorted(v for v in qa[qa.duration == duration].videoID.unique() if v in videos)
        picked += [(v, duration) for v in ids_[:PILOT_PER_DURATION]]
    rows = []
    for vid, duration in picked:
        qs = qa[qa.videoID == vid].sort_values("question_id")
        texts = [build_text(proc, q) for _, q in qs.iterrows()]
        frames, dur, meta = frames_for(proc, videos[vid], texts, TARGETS["16k"])
        fps = meta["nframes"] / dur
        items = []
        for (_, q), text in zip(qs.iterrows(), texts):
            m = check(proc, text, frames, fps, TARGETS["16k"])
            items.append({"question_id": q.question_id, "answer": q.answer, "task_type": q.task_type, "text": text, **m})
            rows.append({"videoID": vid, "duration": duration, "domain": q.domain, "question_id": q.question_id,
                         "task_type": q.task_type, "answer": q.answer, **{k: meta[k] for k in meta}, **m})
        torch.save({"frames": frames, "fps": fps, "meta": {**meta, "videoID": vid, "duration": duration},
                    "questions": items}, f"{OUT}/pilot/{vid}.pt")
        print(vid, duration, meta, [i["L"] for i in items], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/pilot.csv", index=False)
    print("wrote", f"{OUT}/layouts.csv", f"{OUT}/pilot.csv")


if __name__ == "__main__":
    main()
