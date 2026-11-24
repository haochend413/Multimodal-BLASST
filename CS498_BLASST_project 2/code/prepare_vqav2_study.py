# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare VQA v2 validation image/question samples for the existing attention explorer.

Downloads the official question/answer ZIPs and only the selected COCO images.
Reference answers are retained in metadata and never included in model prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

QUESTIONS_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
ANSWERS_URL = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
QUESTIONS_FILE = "v2_OpenEnded_mscoco_val2014_questions.json"
ANSWERS_FILE = "v2_mscoco_val2014_annotations.json"
IMAGE_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org/val2014"


def _download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output)
    partial.replace(destination)


def _load_json(root: Path, filename: str, archive_name: str) -> tuple[dict, Path]:
    direct = root / filename
    if direct.is_file():
        return json.loads(direct.read_text()), direct
    archive = root / "downloads" / archive_name
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {direct} or {archive}; use --download")
    with zipfile.ZipFile(archive) as bundle, bundle.open(filename) as source:
        return json.load(source), archive


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_samples(questions: list[dict], annotations: list[dict], root: Path,
                   *, images: int, questions_per_image: int, seed: int,
                   image_id: int | None = None, all_validation: bool = False) -> list[dict]:
    """Select benchmark questions deterministically, independent of source row ordering."""
    grouped: dict[int, list[dict]] = {}
    for question in questions:
        grouped.setdefault(int(question["image_id"]), []).append(question)
    candidates = sorted(grouped)
    random.Random(seed).shuffle(candidates)
    if image_id is not None:
        if image_id not in grouped:
            raise ValueError(f"Image {image_id} is not in VQA v2 validation")
        if images != 1:
            raise ValueError("--image-id requires --images 1")
        candidates = [image_id]
    if not all_validation and images > len(candidates):
        raise ValueError(f"Requested {images} images, only {len(candidates)} available")
    selected_questions = []
    for selected_id in (candidates if all_validation else candidates[:images]):
        items = sorted(grouped[selected_id], key=lambda item: int(item["question_id"]))
        if not all_validation and len(items) < questions_per_image:
            raise ValueError(f"Image {selected_id} has only {len(items)} questions")
        selected_questions.extend(items if all_validation else items[:questions_per_image])
    wanted = {int(q["question_id"]) for q in selected_questions}
    answers = {int(a["question_id"]): a for a in annotations if int(a["question_id"]) in wanted}
    samples = []
    for question in selected_questions:
        question_id = int(question["question_id"])
        selected_id = int(question["image_id"])
        annotation = answers[question_id]
        if int(annotation["image_id"]) != selected_id:
            raise ValueError(f"Image/question mismatch for {question_id}")
        reference_answers = [str(answer["answer"]) for answer in annotation["answers"]]
        if len(reference_answers) != 10:
            raise ValueError(f"Question {question_id} must have ten reference answers")
        filename = f"COCO_val2014_{selected_id:012d}.jpg"
        samples.append({
            "id": f"vqav2_val_{question_id}", "image": str((root / "val2014" / filename).resolve()),
            "text_before": "", "question": str(question["question"]) + "\nAnswer briefly.",
            "benchmark": {"name": "VQA v2", "split": "validation", "image_id": selected_id,
                          "question_id": question_id, "question_type": annotation["question_type"],
                          "answer_type": annotation["answer_type"], "reference_answers": reference_answers,
                          "ground_truth": annotation["multiple_choice_answer"], "seed": seed,
                          "question_text": str(question["question"]),
                          "image_url": f"{IMAGE_BASE_URL}/{filename}",
                          "layout": "image_then_question", "protocol": "attention diagnostic; not official VQA scoring"}})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--questions-per-image", type=int, default=1)
    parser.add_argument("--image-id", type=int, help="optional exact COCO validation image ID")
    parser.add_argument("--all-validation", action="store_true", help="use every VQA v2 validation question")
    parser.add_argument("--seed", type=int, default=98)
    args = parser.parse_args()
    if args.images < 1 or args.questions_per_image < 1:
        parser.error("images and questions-per-image must be positive")
    if args.all_validation and (args.image_id is not None or args.images != 1 or args.questions_per_image != 1):
        parser.error("--all-validation selects every image and question")
    root = args.data_root.expanduser().resolve()
    if args.download:
        _download(QUESTIONS_URL, root / "downloads" / "v2_Questions_Val_mscoco.zip")
        _download(ANSWERS_URL, root / "downloads" / "v2_Annotations_Val_mscoco.zip")
    questions, question_source = _load_json(root, QUESTIONS_FILE, "v2_Questions_Val_mscoco.zip")
    annotations, answer_source = _load_json(root, ANSWERS_FILE, "v2_Annotations_Val_mscoco.zip")
    samples = select_samples(questions["questions"], annotations["annotations"], root,
                             images=args.images, questions_per_image=args.questions_per_image,
                             seed=args.seed, image_id=args.image_id, all_validation=args.all_validation)
    from PIL import Image
    hashes = {}
    for sample in samples:
        path = sample["image"]
        if path not in hashes:
            image_path = Path(path)
            if args.download and not args.all_validation:
                _download(sample["benchmark"]["image_url"], image_path)
            if not image_path.is_file():
                if args.all_validation:
                    parser.error(f"Missing {path}; extract the COCO val2014 image archive into {root}")
                parser.error(f"Missing image {path}; use --download or supply the local COCO image")
            if not args.all_validation:
                with Image.open(image_path) as image:
                    image.verify()
                hashes[path] = _sha256(image_path)
            else:
                hashes[path] = None
        if hashes[path] is not None:
            sample["benchmark"]["image_sha256"] = hashes[path]
    payload = {"schema": "blasst-benchmark-manifest-v1", "benchmark": "VQA v2",
               "selection_scope": ("full validation split" if args.all_validation
                                   else "seeded validation image subset; not a full benchmark score"),
               "sources": {"questions_url": QUESTIONS_URL, "annotations_url": ANSWERS_URL,
                           "questions_sha256": _sha256(question_source),
                           "annotations_sha256": _sha256(answer_source)}, "samples": samples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(payload, stream, indent=None if args.all_validation else 2, ensure_ascii=False)
        stream.write("\n")
    print(f"Prepared {len(samples)} VQA v2 image/question samples: {args.output}")


if __name__ == "__main__":
    main()
