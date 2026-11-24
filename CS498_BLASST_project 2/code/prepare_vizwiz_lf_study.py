"""Prepare all 600 VizWiz-LF questions with their released human expert answers."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from prepare_vqav2_study import _download, _sha256

ANNOTATIONS = "https://raw.githubusercontent.com/UT-CS-HCI/lfvqa/a8b8e5ea2e2ff75290ac575e590e7ded1cc23f5c/data/expert.json"
ANNOTATIONS_SHA256 = "be3ee44681e30e307977468e046fbd108f2283a3ed63d40ba0a688b9043d2330"


def build_samples(annotations, root):
    if set(annotations) != {str(index) for index in range(1, 601)}:
        raise ValueError("Expected expert annotations for every VizWiz-LF question ID from 1 to 600")
    samples = []
    for key in sorted(annotations, key=int):
        row = annotations[key]
        if row["model"] != "Expert" or not row["answer_paragraph"].strip() or not row["question"].strip():
            raise ValueError(f"Invalid expert annotation for question {key}")
        url = urlparse(row["image_url"])
        image = Path(url.path).name
        if url.scheme != "https" or url.netloc != "vizwiz.cs.colorado.edu" or not image.endswith(".jpg"):
            raise ValueError(f"Unexpected VizWiz image URL for question {key}")
        samples.append(dict(
            id=f"vizwiz_lf_{key}", image=str(root / "images" / image), text_before="", question=row["question"],
            benchmark=dict(name="VizWiz-LF", split="released", question_id=int(key),
                           question_text=row["question"], question_type=row["question_type"],
                           answerability=row["answerability"], image_url=row["image_url"],
                           reference_answers=[row["answer_paragraph"]]),
        ))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="retrieve the expert JSON and any missing images")
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    annotation_path = root / "expert.json"
    try:
        if args.download:
            _download(ANNOTATIONS, annotation_path)
        if _sha256(annotation_path) != ANNOTATIONS_SHA256:
            raise ValueError(f"{annotation_path} differs from the pinned VizWiz-LF expert release")
        samples = build_samples(json.loads(annotation_path.read_text()), root)

        def prepare_image(sample):
            image_path = Path(sample["image"])
            if args.download:
                _download(sample["benchmark"]["image_url"], image_path)
            with Image.open(image_path) as image:
                image.verify()
            sample["benchmark"]["image_sha256"] = _sha256(image_path)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(prepare_image, samples):
                pass
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"VizWiz-LF preparation failed: {error}\n")
    payload = dict(schema="blasst-benchmark-manifest-v1", benchmark="VizWiz-LF",
                   selection_scope="all 600 released questions", samples=samples,
                   sources=dict(expert_annotations_url=ANNOTATIONS, expert_annotations_sha256=ANNOTATIONS_SHA256))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(f"Prepared {len(samples)} VizWiz-LF samples: {args.output}")


if __name__ == "__main__":
    main()
