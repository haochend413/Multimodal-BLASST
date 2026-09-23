"""Inputs for each modality.

Two rules here:

1. Real data by default.  Attention structure is a property of the input as
   much as the weights; synthetic noise would answer a different question.
2. Every loader takes a local ``data_dir`` override, so a cluster node with no
   outbound network can still run once the assets are staged (see
   ``scripts/fetch_assets.py``).

Loaders raise ``DataUnavailable`` with an actionable message rather than
silently falling back to something unrepresentative -- except for time series,
where a clearly-labelled synthetic signal is offered, because the public ETT
CSV moves around and a synthetic AR process is a defensible stand-in for a
sanity run.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import torch


class DataUnavailable(RuntimeError):
    """Raised when inputs cannot be found and no override was given."""


def _stream_first_available(candidates: List[tuple], split: str, what: str):
    """Stream the first dataset id that resolves.

    Hub ids get re-homed under an org over time ("food101" ->
    "ethz/food101", "wikitext" -> "Salesforce/wikitext"), and which spelling
    works depends on the `datasets` version and whether a local cache already
    holds it.  Trying a list beats pinning one id that breaks in six months.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataUnavailable(
            f"install `datasets` to fetch {what}, or pass --data-dir"
        ) from exc

    errors = []
    for repo, config in candidates:
        try:
            if config:
                return load_dataset(repo, config, split=split, streaming=True)
            return load_dataset(repo, split=split, streaming=True)
        except Exception as exc:  # hub layout, offline cache, gating
            errors.append(f"{repo}{'/' + config if config else ''}: {type(exc).__name__}")
    raise DataUnavailable(
        f"could not stream {what} (" + "; ".join(errors) + "); pass --data-dir"
    )


# ---------------------------------------------------------------------------
# text
# ---------------------------------------------------------------------------


def long_text_chunks(count: int, seq_len: int, data_dir: Optional[str]) -> Iterator[str]:
    """Long prose passages.

    ``data_dir`` may hold ``.txt`` files; otherwise we stream WikiText-103 and
    glue articles together until each chunk is long enough to fill the context.
    Chunks must be genuinely long: a 4k-token context padded from 200-token
    articles would mostly measure padding.
    """
    if data_dir:
        files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if files:
            for i, path in enumerate(files[:count]):
                yield Path(path).read_text(errors="ignore")
            return

    ds = _stream_first_available(
        [
            ("Salesforce/wikitext", "wikitext-103-raw-v1"),
            ("wikitext", "wikitext-103-raw-v1"),
            ("Salesforce/wikitext", "wikitext-2-raw-v1"),
            ("wikitext", "wikitext-2-raw-v1"),
        ],
        split="train",
        what="long text",
    )

    # ~5 characters per token is a safe over-estimate for English.
    target_chars = seq_len * 6
    buffer: List[str] = []
    size = 0
    produced = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        buffer.append(text)
        size += len(text)
        if size >= target_chars:
            yield "\n".join(buffer)
            produced += 1
            buffer, size = [], 0
            if produced >= count:
                return


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------


def natural_images(count: int, data_dir: Optional[str]):
    """Natural photographs (PIL images)."""
    from PIL import Image

    if data_dir:
        paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(os.path.join(data_dir, ext)))
        if paths:
            for path in sorted(paths)[:count]:
                yield Image.open(path).convert("RGB")
            return

    ds = _stream_first_available(
        [("ethz/food101", None), ("food101", None), ("detection-datasets/coco", None)],
        split="validation",
        what="natural images",
    )
    for i, row in enumerate(ds):
        if i >= count:
            return
        image = row.get("image") or row.get("img")
        if image is None:
            raise DataUnavailable(f"image dataset row has no image column: {list(row)}")
        yield image.convert("RGB")


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def _decode_video(path: str, num_frames: int):
    """Uniformly sample ``num_frames`` RGB frames from a video file."""
    import numpy as np

    try:
        import av
    except ImportError as exc:
        raise DataUnavailable(
            "video decoding needs PyAV: pip install av"
        ) from exc

    with av.open(path) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        raise DataUnavailable(f"no frames decoded from {path}")
    idx = np.linspace(0, len(frames) - 1, num_frames).round().astype(int)
    return [frames[i] for i in idx]


def video_clips(count: int, num_frames: int, data_dir: Optional[str]):
    if data_dir:
        paths: List[str] = []
        for ext in ("*.mp4", "*.avi", "*.webm", "*.mkv"):
            paths.extend(glob.glob(os.path.join(data_dir, ext)))
        if paths:
            for path in sorted(paths)[:count]:
                yield _decode_video(path, num_frames)
            return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise DataUnavailable("pass --data-dir with .mp4 files") from exc

    # A couple of small public demo clips; both are used in HF's own docs.
    candidates = [
        ("nielsr/video-demo", "eating_spaghetti.mp4", "dataset"),
        ("hf-internal-testing/fixtures_videos", "archery.mp4", "dataset"),
    ]
    got = 0
    for repo_id, filename, repo_type in candidates:
        if got >= count:
            return
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        except Exception:
            continue
        yield _decode_video(path, num_frames)
        got += 1
    if got == 0:
        raise DataUnavailable(
            "no demo video could be downloaded; pass --data-dir with .mp4 files"
        )


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def speech_clips(count: int, data_dir: Optional[str]) -> Iterator[Tuple[object, int]]:
    if data_dir:
        paths = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
        if paths:
            import soundfile as sf

            for path in paths[:count]:
                audio, rate = sf.read(path)
                yield audio, rate
            return

    candidates = [
        ("hf-internal-testing/librispeech_asr_dummy", "clean"),
        ("distil-whisper/librispeech_long", "clean"),
    ]
    ds = _stream_first_available(candidates, split="validation", what="speech clips")

    try:
        for i, row in enumerate(ds):
            if i >= count:
                return
            yield row["audio"]["array"], row["audio"]["sampling_rate"]
        return
    except ImportError:
        # `datasets` >= 4 decodes audio through torchcodec, which needs a
        # matching FFmpeg and is easy to miss on a cluster. The bytes are
        # already there, so decode them ourselves instead of demanding it.
        pass

    try:
        import io

        import soundfile as sf
        from datasets import Audio
    except ImportError as exc:
        raise DataUnavailable(
            "audio decoding needs either `torchcodec` (for datasets) or "
            "`soundfile`; install one, or pass --data-dir with .wav files"
        ) from exc

    ds = _stream_first_available(candidates, split="validation", what="speech clips")
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, row in enumerate(ds):
        if i >= count:
            return
        raw = row["audio"]
        if isinstance(raw, dict) and raw.get("bytes"):
            audio, rate = sf.read(io.BytesIO(raw["bytes"]))
        else:
            audio, rate = sf.read(raw["path"] if isinstance(raw, dict) else raw)
        yield audio, rate


# ---------------------------------------------------------------------------
# time series
# ---------------------------------------------------------------------------


def time_series_windows(
    count: int, context_length: int, num_channels: int, data_dir: Optional[str]
) -> Iterator[torch.Tensor]:
    """Windows of shape ``[context_length, num_channels]``.

    Prefers a CSV in ``data_dir`` (ETTh1 and friends: a date column plus
    numeric channels).  Falls back to a synthetic multi-seasonal AR signal,
    which is labelled as such in the output so it can never be mistaken for a
    measurement on real data.
    """
    if data_dir:
        csvs = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if csvs:
            import csv as _csv

            rows: List[List[float]] = []
            with open(csvs[0]) as fh:
                for row in _csv.DictReader(fh):
                    numeric = []
                    for key, val in row.items():
                        try:
                            numeric.append(float(val))
                        except (TypeError, ValueError):
                            continue  # date column
                    if numeric:
                        rows.append(numeric)
            series = torch.tensor(rows, dtype=torch.float32)
            series = series[:, :num_channels]
            if series.shape[1] < num_channels:
                series = series.repeat(1, (num_channels // series.shape[1]) + 1)[
                    :, :num_channels
                ]
            step = max(1, (series.shape[0] - context_length) // max(1, count))
            for i in range(count):
                start = i * step
                if start + context_length > series.shape[0]:
                    return
                window = series[start : start + context_length]
                mu, sigma = window.mean(0, keepdim=True), window.std(0, keepdim=True)
                yield (window - mu) / sigma.clamp_min(1e-6)
            return

    for i in range(count):
        g = torch.Generator().manual_seed(7 + i)
        t = torch.arange(context_length, dtype=torch.float32)
        out = []
        for c in range(num_channels):
            period = 24 * (c + 1)
            seasonal = torch.sin(2 * torch.pi * t / period)
            trend = 0.001 * t * (1 if c % 2 == 0 else -1)
            noise = torch.randn(context_length, generator=g) * 0.1
            ar = torch.zeros(context_length)
            for j in range(1, context_length):
                ar[j] = 0.9 * ar[j - 1] + noise[j]
            out.append(seasonal + trend + ar)
        yield torch.stack(out, dim=-1)


SYNTHETIC_TIMESERIES_NOTE = (
    "time-series windows were synthetic (seasonal + AR); pass --data-dir with an "
    "ETTh1-style CSV for real data"
)
