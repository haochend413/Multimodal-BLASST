#!/usr/bin/env python
"""Pre-download every model and input this suite needs.

Compute nodes are often firewalled, and a job that dies 40 minutes in because a
dataset id moved is a wasted allocation.  Run this once on a login node, then
run the suite with ``HF_HOME`` pointing at the same cache (and optionally
``HF_HUB_OFFLINE=1`` to prove nothing else reaches out).

    python scripts/fetch_assets.py --out assets/
    HF_HUB_OFFLINE=1 ./scripts/run_suite.sh

Anything that fails is reported and skipped; the suite will then say clearly
which modality it could not run instead of failing mid-sweep.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qk_sparse.models import REGISTRY  # noqa: E402


def fetch_models(names) -> list:
    failures = []
    for name in names:
        adapter = REGISTRY[name]()
        if adapter.spec.model_id == "none":
            continue
        print(f"[fetch] model {name}: {adapter.spec.model_id}", flush=True)
        try:
            adapter.load()
            del adapter
        except Exception:
            traceback.print_exc()
            failures.append(f"model:{name}")
    return failures


def fetch_inputs(out: Path, count: int) -> list:
    """Materialise a few inputs of each kind into ``out`` for offline runs."""
    from qk_sparse import data

    out.mkdir(parents=True, exist_ok=True)
    failures = []

    try:
        print("[fetch] text", flush=True)
        for i, text in enumerate(data.long_text_chunks(count, 4096, None)):
            (out / f"text_{i}.txt").write_text(text)
    except Exception:
        traceback.print_exc()
        failures.append("text")

    try:
        print("[fetch] images", flush=True)
        for i, image in enumerate(data.natural_images(count, None)):
            image.save(out / f"image_{i}.jpg")
    except Exception:
        traceback.print_exc()
        failures.append("image")

    try:
        print("[fetch] video", flush=True)
        import shutil

        from huggingface_hub import hf_hub_download

        for repo, filename, kind in [
            ("nielsr/video-demo", "eating_spaghetti.mp4", "dataset"),
            ("hf-internal-testing/fixtures_videos", "archery.mp4", "dataset"),
        ]:
            try:
                path = hf_hub_download(repo_id=repo, filename=filename, repo_type=kind)
                shutil.copy(path, out / Path(filename).name)
            except Exception as exc:
                print(f"  {repo}/{filename}: {type(exc).__name__}")
    except Exception:
        traceback.print_exc()
        failures.append("video")

    try:
        print("[fetch] audio", flush=True)
        import soundfile as sf

        for i, (audio, rate) in enumerate(data.speech_clips(count, None)):
            sf.write(out / f"audio_{i}.wav", audio, rate)
    except Exception:
        traceback.print_exc()
        failures.append("audio")

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("assets"))
    ap.add_argument("--count", type=int, default=4, help="inputs per modality")
    ap.add_argument("--adapters", nargs="+", default=sorted(REGISTRY))
    ap.add_argument("--models-only", action="store_true")
    ap.add_argument("--inputs-only", action="store_true")
    args = ap.parse_args()

    failures = []
    if not args.inputs_only:
        failures += fetch_models(args.adapters)
    if not args.models_only:
        failures += fetch_inputs(args.out, args.count)

    if failures:
        print(f"\n[fetch] could not fetch: {', '.join(failures)}", file=sys.stderr)
        print("[fetch] the suite will skip those modalities with a warning",
              file=sys.stderr)
        return 1
    print(f"\n[fetch] all assets ready; run with --data-dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
