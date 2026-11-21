#!/usr/bin/env python
"""Export per-row QK^T structure for visualisation.

Every other measurement in this repo is an aggregate: medians over blocks,
heads and layers. They answer "how concentrated is attention on average" but
never show what a single row of `QK^T` actually looks like -- which key columns
carry it, how they sit relative to FlashAttention's tile boundaries, and how
similar one row is to the row below it.

This writes a compact JSON for a viewer: for a handful of query rows from a real
forward pass, the full contribution profile plus the per-tile maxima needed to
replay the skip decision at any threshold.

Size discipline: a full row is up to 32k keys, so rows are stored as
  * `bins`  -- max attention weight per bin, enough to draw the row
  * `top`   -- the exact top-K keys, enough to read off specifics
  * `tile_max` -- one score per 64-key tile, enough to recompute skipping live

Usage:
    python scripts/export_rows.py --adapter text-small --out viz/rows.json
    python scripts/export_rows.py --adapter image --device cuda --out viz/image.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qk_sparse.capture import CaptureConfig, SDPACapture, pick_layers  # noqa: E402
from qk_sparse.models import build  # noqa: E402


def export_head(q, k, *, causal, row_start, n_rows, key_tile, n_bins, top_k, scale):
    """Per-row structure for `n_rows` consecutive queries of one head."""
    d = q.shape[-1]
    scale = scale or 1.0 / math.sqrt(d)
    q = q.float()
    k = k.float()

    rows = []
    for i in range(row_start, min(row_start + n_rows, q.shape[0])):
        # Causal rows see only their past; keep the same convention as the
        # metrics code so the picture matches the numbers.
        n_allowed = i if causal else k.shape[0]
        if n_allowed < key_tile * 4:
            continue
        scores = (q[i] @ k[:n_allowed].T) * scale
        attn = torch.softmax(scores, dim=-1)

        # Binned profile: max within each bin, so a single dominant key is never
        # averaged away into invisibility.
        pad = (-n_allowed) % n_bins
        padded = torch.nn.functional.pad(attn, (0, pad))
        bins = padded.view(n_bins, -1).max(-1).values

        top = attn.topk(min(top_k, n_allowed))
        n_tiles = n_allowed // key_tile
        tile_max = scores[: n_tiles * key_tile].view(n_tiles, key_tile).max(-1).values

        entropy = float(-(attn * (attn + 1e-12).log()).sum())
        rows.append({
            "i": i,
            "n_allowed": n_allowed,
            "global_max": round(float(scores.max()), 4),
            "eff_keys": round(math.exp(entropy), 2),
            "eff_key_frac": round(math.exp(entropy) / n_allowed, 4),
            "bins": [round(float(x), 6) for x in bins],
            "top_idx": [int(x) for x in top.indices],
            "top_w": [round(float(x), 6) for x in top.values],
            "tile_max": [round(float(x), 3) for x in tile_max],
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="text-small")
    ap.add_argument("--out", type=Path, default=Path("viz/rows.json"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--layers", type=int, default=3, help="depths to export")
    ap.add_argument("--heads", type=int, default=2, help="heads per depth")
    ap.add_argument("--rows", type=int, default=48, help="consecutive query rows")
    ap.add_argument("--row-start", type=int, default=None,
                    help="first query row; default is 2/3 through the sequence")
    ap.add_argument("--bins", type=int, default=384)
    ap.add_argument("--top-k", type=int, default=32)
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    adapter = build(args.adapter, device=args.device, dtype=dtype)
    spec = adapter.spec
    print(f"[export] loading {spec.model_id}", flush=True)
    adapter.load()

    sample = next(iter(adapter.samples(1, args.data_dir)))

    # Same two-pass approach as the runner: count calls, then capture a spread.
    with SDPACapture(CaptureConfig(layers=None, min_seq_len=spec.min_seq_len)) as probe:
        sample.run()
    self_calls = probe.self_attention_calls()
    if not self_calls:
        print("[export] no long self-attention reached SDPA", file=sys.stderr)
        return 1
    chosen = pick_layers(self_calls, args.layers)

    with SDPACapture(CaptureConfig(layers=set(chosen), min_seq_len=spec.min_seq_len,
                                   max_heads=args.heads, store_device=args.device)) as cap:
        sample.run()

    out = {
        "model": spec.model_id,
        "adapter": args.adapter,
        "modality": spec.modality,
        "causal": spec.causal,
        "key_tile": spec.key_block,
        "query_tile": spec.query_block,
        "n_bins": args.bins,
        "heads": [],
    }

    for depth, call in enumerate(cap.calls):
        n = call.query.shape[1]
        row_start = args.row_start if args.row_start is not None else (2 * n) // 3
        for h in range(call.query.shape[0]):
            rows = export_head(
                call.query[h], call.key[h],
                causal=call.is_causal or spec.causal,
                row_start=row_start, n_rows=args.rows,
                key_tile=spec.key_block, n_bins=args.bins,
                top_k=args.top_k, scale=call.scale,
            )
            if rows:
                out["heads"].append({
                    "depth": depth,
                    "call_index": call.index,
                    "head": int(call.heads[h]),
                    "seq_len": int(n),
                    "rows": rows,
                })
                print(f"[export] depth {depth} head {call.heads[h]}: "
                      f"{len(rows)} rows of {rows[0]['n_allowed']} keys", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    size = args.out.stat().st_size / 1e6
    print(f"[export] wrote {args.out} ({size:.1f} MB, {len(out['heads'])} heads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
