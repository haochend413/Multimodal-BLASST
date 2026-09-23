"""Command line entry points.

    python -m qk_sparse run --adapters synthetic-high synthetic-low --out results/
    python -m qk_sparse report results/ --density 0.05
    python -m qk_sparse list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from .aggregate import load_records, report
from .models import DEFAULT_SUITE, REGISTRY
from .runner import RunConfig, run_adapter


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--adapters", nargs="+", default=DEFAULT_SUITE,
                   help=f"adapters to run (default: the full suite). Options: {sorted(REGISTRY)}")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--samples", type=int, default=2, help="inputs per model")
    p.add_argument("--layers", type=int, default=4, help="depths sampled per model")
    p.add_argument("--heads", type=int, default=4, help="heads analysed per layer")
    p.add_argument("--max-blocks", type=int, default=16,
                   help="query blocks analysed per head")
    p.add_argument("--densities", type=float, nargs="+",
                   default=[0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--deltas", type=int, nargs="+", default=None,
                   help="query offsets to probe; default is per-adapter "
                        "(grid strides for image/video)")
    p.add_argument("--device", default="cuda" if _cuda() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--data-dir", default=None,
                   help="local inputs (.txt/.jpg/.mp4/.wav/.csv) instead of downloads")
    p.add_argument("--query-block", type=int, default=None,
                   help="override the adapter's query block size (P_q)")
    p.add_argument("--key-block", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-going", action="store_true",
                   help="continue to the next adapter if one fails")


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qk_sparse", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="capture Q/K and compute metrics")
    _add_run_args(run_p)

    rep_p = sub.add_parser("report", help="build markdown tables from results")
    rep_p.add_argument("paths", nargs="+", type=Path)
    rep_p.add_argument("--density", type=float, default=0.05)
    rep_p.add_argument("--deltas", type=int, nargs="+",
                       default=[1, 4, 16, 64, 256])
    rep_p.add_argument("--out", type=Path, default=None)
    rep_p.add_argument("--gap", type=float, default=4.0,
                       help="skip threshold in e-folds below the row max. Not "
                            "free: gap = -ln(threshold) = ln(L/tsf), so "
                            "BLASST's aggressive settings are gap 0.6-3.4")
    rep_p.add_argument("--by-layer", action="store_true",
                       help="add a per-depth breakdown; attention structure "
                            "varies strongly with layer index")

    sub.add_parser("list", help="list available adapters")

    args = parser.parse_args(argv)

    if args.command == "list":
        for name, factory in sorted(REGISTRY.items()):
            try:
                spec = factory().spec
                print(f"{name:20s} {spec.modality:11s} {spec.model_id}")
            except Exception as exc:  # pragma: no cover
                print(f"{name:20s} <error: {exc}>")
        return 0

    if args.command == "run":
        failures = []
        for name in args.adapters:
            cfg = RunConfig(
                adapter=name,
                samples=args.samples,
                layers=args.layers,
                heads=args.heads,
                densities=tuple(args.densities),
                deltas=tuple(args.deltas) if args.deltas else None,
                max_blocks=args.max_blocks,
                device=args.device,
                dtype=args.dtype,
                data_dir=args.data_dir,
                query_block=args.query_block,
                key_block=args.key_block,
                seed=args.seed,
            )
            print(f"[qk_sparse] running {name} on {args.device} ...", flush=True)
            try:
                rec = run_adapter(cfg, args.out)
            except Exception as exc:
                failures.append((name, exc))
                print(f"[qk_sparse] {name} FAILED: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                if not args.keep_going:
                    raise
                continue
            n_samples = len(rec.get("samples", []))
            print(f"[qk_sparse] {name}: {n_samples} sample(s) -> {rec['_path']}",
                  flush=True)
            for w in rec.get("warnings", []):
                print(f"[qk_sparse]   warning: {w}", flush=True)
        if failures:
            print(f"[qk_sparse] {len(failures)} adapter(s) failed: "
                  f"{', '.join(n for n, _ in failures)}", file=sys.stderr)
            return 1
        return 0

    if args.command == "report":
        records = load_records(args.paths)
        if not records:
            print("no result files found", file=sys.stderr)
            return 1
        text = report(records, args.density, args.deltas, args.by_layer, args.gap)
        if args.out:
            args.out.write_text(text)
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
