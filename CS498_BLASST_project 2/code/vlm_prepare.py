"""
Step 1 of the VLM region study, CPU only: build the 10 video prompts (5 videos x about 8K and 16K tokens).

For each prompt we save the decoded video frames and the text pieces, so the GPU job only has to run the
model. Prompt layout (Qwen2.5-VL chat template; the default system message is added by the template):

  [system] [user: transcript part 1, about 1,024 tokens] [video] [transcript part 2, 512-1,024 tokens]
  [question + 4 options] [assistant]

Text before the video gives video->text tiles (video rows looking back at text); text after it gives
text->video tiles (the question looking back at the video). Part 1 is trimmed so the first video token sits
at a multiple of 128, so the text/video boundary falls between two tiles instead of inside one.

Video length is set by the number of frames: every frame is resized to at most 256 x 28 x 28 pixels
(336 x 588 for 16:9), which gives 252 video tokens per pair of frames (Qwen merges 2 frames into one token
group). nframes is chosen so the whole prompt is just under 8,192 or 16,384 tokens.

  python vlm_prepare.py                 # writes vlm_inputs/<videoID>_<8k|16k>.pt and vlm_inputs/prompts.csv
"""
import glob
import math
import os
import re
import zipfile

import av
import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor

W = "/work/nvme/beqg/zyang21"
VM = f"{W}/data/videomme"
OUT = f"{W}/blasst/vlm_inputs"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VIDEO_PAD = 151656                     # <|video_pad|>: one per video token
MAX_PIXELS = 256 * 28 * 28             # per-frame cap -> 336 x 588 for 16:9, 252 tokens per 2 frames
MIN_PIXELS = 32 * 28 * 28
TARGETS = {"8k": 8192, "16k": 16384}
PART1, PART2_MIN, PART2_MAX = 1024, 512, 1024   # transcript tokens before / after the video
N_VIDEOS = 5
TILE = 128


def processor():
    p = AutoProcessor.from_pretrained(MODEL)
    # the HF video processor resizes frames again with its own limits; make them match ours
    p.video_processor.size = {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS}
    return p


def srt_text(path):
    """Subtitle file -> plain text: drop cue numbers, timestamps, HTML tags, repeated lines."""
    lines, last = [], None
    for ln in open(path, encoding="utf-8", errors="ignore"):
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if not ln or ln.isdigit() or "-->" in ln or ln == last:
            continue
        lines.append(ln)
        last = ln
    return " ".join(lines)


def smart_resize(h, w, factor=28, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Qwen's resize rule: sides become multiples of 28, area between min_pixels and max_pixels."""
    h_bar, w_bar = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar, w_bar = math.ceil(h * beta / factor) * factor, math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def video_info(path):
    with av.open(path) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
        return dur, s.codec_context.height, s.codec_context.width


def read_frames(path, nframes, H, W):
    """nframes evenly spaced frames (centre of each of nframes equal time slices), resized to H x W,
    as a uint8 tensor [nframes, 3, H, W]. Seeks to each time instead of decoding the whole file."""
    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        dur = float(s.duration * s.time_base) if s.duration else c.duration / 1e6
        for i in range(nframes):
            t = (i + 0.5) * dur / nframes
            c.seek(int(t / s.time_base), stream=s, backward=True, any_frame=False)
            frame = None
            for f in c.decode(s):
                frame = f
                if f.time is not None and f.time >= t:
                    break
            img = frame.to_ndarray(width=W, height=H, format="rgb24")      # [H, W, 3]
            out.append(torch.from_numpy(img).permute(2, 0, 1))
    return torch.stack(out), dur


def build_text(proc, part1, part2, question, options):
    """Chat-template text with one video placeholder between the two transcript parts."""
    q = ("Question: " + question + "\nOptions:\n" + "\n".join(options) +
         "\nAnswer with the option's letter from the given choices directly.")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Here is the first part of the video's transcript:\n" + part1 + "\n\nHere is the video:\n"},
        {"type": "video"},
        {"type": "text", "text": "\n\nHere is the rest of the transcript:\n" + part2 + "\n\n" + q},
    ]}]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_inputs(proc, text, frames, fps):
    return proc(text=[text], videos=[frames], fps=[fps], return_tensors="pt")


def main():
    os.makedirs(OUT, exist_ok=True)
    proc = processor()
    tok = proc.tokenizer
    qa = pd.read_parquet(f"{VM}/videomme/test-00000-of-00001.parquet")

    # unzip the downloaded video chunk(s) once
    for z in sorted(glob.glob(f"{VM}/videos_chunked_*.zip")):
        flag = z + ".unzipped"
        if not os.path.exists(flag):
            zipfile.ZipFile(z).extractall(VM)
            open(flag, "w").close()
    videos = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{VM}/**/*.mp4", recursive=True)}
    print(f"{len(videos)} videos on disk")

    # candidates: on disk, have subtitles with enough text, medium or long; one per domain, long first
    cands = []
    for vid, g in qa.groupby("videoID"):
        sub = f"{VM}/subtitle/{vid}.srt"
        if vid not in videos or not os.path.exists(sub) or g.duration.iloc[0] == "short":
            continue
        n_sub = len(tok(srt_text(sub)).input_ids)
        if n_sub >= PART1 + PART2_MIN + 64:
            cands.append({"videoID": vid, "duration": g.duration.iloc[0], "domain": g.domain.iloc[0], "sub_tokens": n_sub})
    cands = pd.DataFrame(cands).sort_values(["duration", "videoID"])          # 'long' < 'medium' alphabetically
    print(f"{len(cands)} candidates:\n{cands.groupby(['duration', 'domain']).size()}")
    picked, domains = [], set()
    for _, r in cands.iterrows():                                              # one per domain first
        if r.domain not in domains and len(picked) < N_VIDEOS:
            picked.append(r)
            domains.add(r.domain)
    for _, r in cands.iterrows():                                              # then fill up
        if len(picked) < N_VIDEOS and r.videoID not in [p.videoID for p in picked]:
            picked.append(r)

    rows = []
    for r in picked:
        vid = r.videoID
        q = qa[qa.videoID == vid].sort_values("question_id").iloc[0]
        ids = tok(srt_text(f"{VM}/subtitle/{vid}.srt")).input_ids
        dur, h, w = video_info(videos[vid])
        H, Wd = smart_resize(h, w)
        tpg = (H // 28) * (Wd // 28)                                           # tokens per 2-frame group
        for name, target in TARGETS.items():
            # 1) trim part 1 so the first video token is at a multiple of 128
            part2 = tok.decode(ids[PART1:PART1 + PART2_MAX])
            for cut in range(PART1, PART1 - TILE, -1):
                part1 = tok.decode(ids[:cut])
                pre = build_text(proc, part1, part2, q.question, list(q.options)).split("<|video_pad|>")[0]
                if len(tok(pre).input_ids) % TILE == 0:
                    break
            text = build_text(proc, part1, part2, q.question, list(q.options))
            n_text = len(tok(text).input_ids) - 1                              # minus the single <|video_pad|>
            # 2) frames: as many 2-frame groups as fit under the target length
            groups = (target - n_text) // tpg
            nframes = 2 * groups
            frames, _ = read_frames(videos[vid], nframes, H, Wd)
            fps = nframes / dur                                                # sampled frames per second
            inp = make_inputs(proc, text, frames, fps)
            L = inp.input_ids.shape[1]
            n_vid = int((inp.input_ids == VIDEO_PAD).sum())
            first_vid = int((inp.input_ids[0] == VIDEO_PAD).nonzero()[0])
            assert n_vid == int(inp.video_grid_thw.prod(-1).sum()) // 4 == groups * tpg, (n_vid, inp.video_grid_thw)
            assert L <= target, (L, target)
            meta = {"videoID": vid, "length_name": name, "duration": r.duration, "domain": r.domain,
                    "question_id": q.question_id, "answer": q.answer, "video_seconds": round(dur, 1),
                    "source_hw": f"{h}x{w}", "nframes": nframes, "H": H, "W": Wd,
                    "grid_thw": inp.video_grid_thw[0].tolist(), "video_tokens": n_vid, "text_tokens": L - n_vid,
                    "L": L, "first_video_pos": first_vid, "fps": round(fps, 4)}
            torch.save({"frames": frames, "text": text, "fps": fps, "meta": meta}, f"{OUT}/{vid}_{name}.pt")
            rows.append(meta)
            print(meta, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/prompts.csv", index=False)
    print("wrote", f"{OUT}/prompts.csv")


if __name__ == "__main__":
    main()
