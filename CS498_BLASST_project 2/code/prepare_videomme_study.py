# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare reproducible Video-MME attention samples using benchmark videos and questions.

Download only requested video archives, or reuse an existing local Video-MME copy.
No model inference is performed here. Media and annotations are never uploaded to the site.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import zipfile
from pathlib import Path

REPO = "lmms-eval/Video-MME"
REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
ANNOTATIONS = "videomme/test-00000-of-00001.parquet"
DURATIONS = ("short", "medium", "long")


def _unpack(archive: Path, target: Path, suffix: str) -> None:
    """Copy matching files by basename without trusting archive paths."""
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = Path(entry.filename).name
            if entry.is_dir() or not name.endswith(suffix):
                continue
            destination = target / name
            if destination.exists():
                if destination.stat().st_size != entry.file_size:
                    raise ValueError(f"Existing file differs from archive: {destination}")
                continue
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with bundle.open(entry) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            temporary.replace(destination)


def _subtitle(path: Path) -> str:
    """Return readable SRT text, excluding timestamps and cue numbers."""
    output = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = re.sub(r"<[^>]+>", "", line).strip()
        if line and not line.isdigit() and "-->" not in line:
            if not output or output[-1] != line:
                output.append(line)
    return " ".join(output)


def prepare_samples(rows: list[dict], media: dict[str, Path], subtitles: dict[str, Path],
                    *, layout: str, durations: list[str], videos_per_duration: int,
                    questions_per_video: int, seed: int, subtitle_chars: int) -> list[dict]:
    """Sample video IDs before questions; never place ground-truth answers in model inputs."""
    samples = []
    for duration in durations:
        ids = sorted({str(row["videoID"]) for row in rows
                      if row["duration"] == duration and str(row["videoID"]) in media
                      and (layout == "C" or str(row["videoID"]) in subtitles)})
        random.Random(f"{seed}:{duration}").shuffle(ids)
        if len(ids) < videos_per_duration:
            raise ValueError(f"Need {videos_per_duration} available {duration} videos, found {len(ids)}. "
                             "Download more chunks or choose fewer/different durations.")
        for video_id in ids[:videos_per_duration]:
            questions = sorted((row for row in rows if str(row["videoID"]) == video_id),
                               key=lambda row: row["question_id"])[:questions_per_video]
            for row in questions:
                options = list(row["options"])
                if len(options) != 4 or row["answer"] not in ("A", "B", "C", "D"):
                    raise ValueError(f"Invalid multiple-choice annotation: {row['question_id']}")
                question = ("Select the best answer to the following multiple-choice question based on the video. "
                            "Respond with only the letter (A, B, C, or D) of the correct option.\n"
                            + str(row["question"]) + "\n" + "\n".join(options) + "\nThe best answer is:")
                before = ""
                if layout != "C":
                    subtitle = _subtitle(subtitles[video_id])[:subtitle_chars]
                    if layout == "A":
                        split = len(subtitle) // 2
                        before = "Transcript, first part:\n" + subtitle[:split] + "\nVideo:\n"
                        question = "Transcript, remaining part:\n" + subtitle[split:] + "\n" + question
                    else:
                        question = "This video's subtitles are listed below:\n" + subtitle + "\n" + question
                samples.append({"id": f"videomme_{row['question_id']}_{layout}",
                                "video": str(media[video_id].resolve()), "text_before": before,
                                "question": question, "benchmark": {
                                    "name": "Video-MME", "split": "test", "repo": REPO,
                                    "revision": REVISION, "video_id": video_id,
                                    "question_id": str(row["question_id"]),
                                    "duration": duration, "domain": str(row["domain"]),
                                    "task_type": str(row["task_type"]), "ground_truth": row["answer"],
                                    "layout": layout, "seed": seed,
                                    "protocol": ("video-only question; diagnostic frame budget" if layout == "C"
                                                 else "diagnostic truncated-subtitle layout; not official accuracy protocol")}})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="download annotations and requested media archives")
    parser.add_argument("--chunks", default="01", help="comma-separated video archive numbers, e.g. 01,02")
    parser.add_argument("--layout", choices=("A", "B", "C"), default="C")
    parser.add_argument("--durations", default="short,medium,long")
    parser.add_argument("--videos-per-duration", type=int, default=1)
    parser.add_argument("--questions-per-video", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--subtitle-chars", type=int, default=4000)
    args = parser.parse_args()
    durations = args.durations.split(",")
    if not durations or any(d not in DURATIONS for d in durations) or len(set(durations)) != len(durations):
        parser.error("durations must be a unique comma-separated subset of short,medium,long")
    if args.videos_per_duration < 1 or args.subtitle_chars < 1:
        parser.error("videos-per-duration and subtitle-chars must be positive")
    chunks = args.chunks.split(",")
    if any(not c.isdigit() or not 1 <= int(c) <= 20 for c in chunks):
        parser.error("chunk numbers must be between 01 and 20")
    root = args.data_root.expanduser().resolve()
    if args.download:
        from huggingface_hub import hf_hub_download
        names = [ANNOTATIONS] + [f"videos_chunked_{int(c):02d}.zip" for c in chunks]
        if args.layout != "C":
            names.append("subtitle.zip")
        for name in names:
            print(f"Downloading {REPO}@{REVISION}: {name}", flush=True)
            path = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=root))
            if name.startswith("videos_chunked_"):
                _unpack(path, root / "videos", ".mp4")
            elif name == "subtitle.zip":
                _unpack(path, root / "subtitle", ".srt")
    annotation_file = root / ANNOTATIONS
    if not annotation_file.is_file():
        parser.error(f"Missing {annotation_file}; use --download or an existing dataset directory")
    import pandas as pd
    rows = pd.read_parquet(annotation_file).to_dict("records")
    required = {"videoID", "question_id", "question", "options", "answer", "duration", "domain", "task_type"}
    if not rows or not required.issubset(rows[0]):
        parser.error("annotation file is missing required Video-MME fields")
    media: dict[str, Path] = {}
    for path in root.rglob("*.mp4"):
        if path.stem in media:
            raise ValueError(f"Duplicate video ID: {path.stem}")
        media[path.stem] = path
    subtitles = {path.stem: path for path in root.rglob("*.srt")}
    samples = prepare_samples(rows, media, subtitles, layout=args.layout, durations=durations,
                              videos_per_duration=args.videos_per_duration,
                              questions_per_video=args.questions_per_video,
                              seed=args.seed, subtitle_chars=args.subtitle_chars)
    payload = {"schema": "blasst-benchmark-manifest-v1", "benchmark": "Video-MME",
               "selection_scope": "available local videos in selected chunks; not a full benchmark score",
               "annotation_revision": REVISION if args.download else "local_unverified",
               "samples": samples}
    # A local copy may have come from another revision; do not claim it was verified.
    if not args.download:
        for sample in samples:
            sample["benchmark"]["revision"] = "local_unverified"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Prepared {len(samples)} real benchmark question/video entries: {args.output}")


if __name__ == "__main__":
    main()
