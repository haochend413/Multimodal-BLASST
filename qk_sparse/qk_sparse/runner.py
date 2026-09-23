"""Run one adapter end to end and write a JSON record per model.

Flow per sample:

1. Dry forward pass with capture counting only -- this is how we learn how many
   attention calls the model makes and which of them are long self-attentions,
   without hard-coding anything per architecture.
2. Pick a spread of depths from that list.
3. Second forward pass storing Q/K for those layers only.
4. Metrics per (layer, head), appended to the record.

The dry pass costs one extra forward.  On models this small that is cheap, and
it buys architecture independence.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from . import metrics as M
from . import tile_skip as T
from .capture import CaptureConfig, SDPACapture, pick_layers, summarise_shapes
from .models import DEFAULT_DELTAS, Adapter, build


@dataclass
class RunConfig:
    adapter: str
    samples: int = 2
    layers: int = 4
    heads: int = 4
    densities: Sequence[float] = (0.01, 0.02, 0.05, 0.1, 0.2)
    deltas: Optional[Sequence[int]] = None  # None = the adapter's own grid strides
    max_blocks: int = 16
    minp_gaps: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    """How many e-folds below the row max counts as negligible.

    Not a free parameter: BLASST skips when ``exp(s - max) < threshold``, so
    ``gap = -ln(threshold) = ln(L / threshold_scale_factor)``. With blog16's
    calibrated factors for Qwen3-30B the operating points are

        sparsity 0.5 -> gap 1.9 (4k ctx) .. 5.1 (100k)
        sparsity 0.7 -> gap 0.2 (4k)      .. 3.4 (100k)
        sparsity 0.9 -> gap 0.6 (32k)     .. 1.7 (100k)

    so the aggressive settings live *below* 4, which the old default
    (4, 6, 8) never reached. Smaller gap means more tiles clear the bar.
    """
    device: str = "cpu"
    dtype: str = "float32"
    data_dir: Optional[str] = None
    query_block: Optional[int] = None
    key_block: Optional[int] = None
    seed: int = 0


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[name]


def run_adapter(cfg: RunConfig, out_dir: Path) -> Dict[str, object]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no CUDA device is visible")

    adapter: Adapter = build(cfg.adapter, device=device, dtype=_dtype(cfg.dtype))
    spec = adapter.spec
    query_block = cfg.query_block or spec.query_block
    key_block = cfg.key_block or spec.key_block
    def resolve_deltas() -> tuple:
        """Read the probe offsets *late*.

        Adapters may only learn their token grid once weights or inputs are
        loaded -- Wan derives strides from the latent resolution in ``load()``,
        Qwen2.5-VL from ``video_grid_thw`` per sample. Resolving this once up
        front silently discarded both, so the overlap table came back with only
        d=1 and every grid-aligned offset blank.
        """
        return tuple(cfg.deltas or spec.deltas or DEFAULT_DELTAS)

    t0 = time.time()
    adapter.load()
    load_s = time.time() - t0
    load_warnings = list(getattr(adapter, "load_warnings", []))
    for warning in load_warnings:
        print(f"[qk_sparse] WEIGHTS: {warning}", flush=True)

    record: Dict[str, object] = {
        # The registry key, not spec.name: two entries can share an adapter
        # class (text / text-small) and must not overwrite each other's file.
        "adapter": cfg.adapter,
        "adapter_class": spec.name,
        "modality": spec.modality,
        "model_id": spec.model_id,
        "causal": spec.causal,
        "note": spec.note,
        "query_block": query_block,
        "key_block": key_block,
        "densities": list(cfg.densities),
        "deltas": [],  # filled in after load/sample, see resolve_deltas
        "config": asdict(cfg) | {"densities": list(cfg.densities),
                                 "deltas": list(cfg.deltas or ()),
                                 "minp_gaps": list(cfg.minp_gaps)},
        "env": {"torch": torch.__version__, "python": platform.python_version(),
                "device": device, "dtype": cfg.dtype},
        "load_seconds": round(load_s, 2),
        "invalid_weights": any("NOT valid" in w for w in load_warnings),
        "samples": [],
    }
    if load_warnings:
        record["warnings"] = list(load_warnings)

    produced = 0

    # Adapters that already hold Q/K (e.g. replaying an external dump) skip the
    # two-pass SDPA capture entirely: there is no forward pass to intercept,
    # and running one on a 100k-token tensor merely to observe it is absurd.
    if hasattr(adapter, "iter_captures"):
        calls = list(adapter.iter_captures(
            cfg.samples, cfg.layers, cfg.heads, cfg.data_dir))
        sample_rec = {"sample_id": "dump", "n_attention_calls": len(calls),
                      "n_self_attention_calls": len(calls), "layers": []}
        for call in calls:
            heads = []
            for h in range(call.query.shape[0]):
                res = M.analyse_head(
                    call.query[h], call.key[h],
                    value=call.value[h] if call.value is not None else None,
                    causal=call.is_causal,
                    query_block=query_block, key_block=key_block,
                    densities=cfg.densities, deltas=resolve_deltas(),
                    max_blocks=cfg.max_blocks, scale=call.scale,
                    minp_gaps=cfg.minp_gaps, group_strides=spec.group_strides,
                )
                if res is not None:
                    tiles = T.analyse_tiles(
                        call.query[h], call.key[h], causal=call.is_causal,
                        query_tile=query_block, key_tile=key_block,
                        gaps=cfg.minp_gaps, scale=call.scale,
                    )
                    heads.append({"head": h, **res,
                                  **({"tiles": tiles} if tiles else {})})
            if heads:
                sample_rec["layers"].append({
                    "depth": len(sample_rec["layers"]),
                    "call_index": call.index, "is_causal": call.is_causal,
                    "had_mask": call.had_mask,
                    "n_query": int(call.query.shape[1]),
                    "n_key": int(call.key.shape[1]), "heads": heads,
                })
        if sample_rec["layers"]:
            record["samples"].append(sample_rec)
            produced = 1
        record["load_seconds"] = round(time.time() - t0, 2)
        record["deltas"] = list(resolve_deltas())
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{cfg.adapter}.json"
        path.write_text(json.dumps(record, indent=2))
        record["_path"] = str(path)
        return record

    for sample in adapter.samples(cfg.samples, cfg.data_dir):
        # -- pass 1: what attention calls does this model make?
        probe_cfg = CaptureConfig(layers=None, min_seq_len=spec.min_seq_len)
        with SDPACapture(probe_cfg) as probe:
            sample.run()
        self_calls = probe.self_attention_calls()
        if not self_calls:
            record.setdefault("warnings", []).append(
                f"{sample.sample_id}: no self-attention call of length "
                f">= {spec.min_seq_len} reached torch SDPA "
                f"({probe.n_calls} calls seen). Shapes, largest first: "
                f"{summarise_shapes(probe.seen)}"
            )
            continue

        layer_ids = pick_layers(self_calls, cfg.layers)

        # -- pass 2: capture the chosen depths
        cap_cfg = CaptureConfig(
            layers=set(layer_ids),
            min_seq_len=spec.min_seq_len,
            max_heads=cfg.heads,
            store_dtype=torch.float16,
            store_device=device,
        )
        with SDPACapture(cap_cfg) as cap:
            sample.run()

        sample_rec: Dict[str, object] = {
            "sample_id": sample.sample_id,
            "n_attention_calls": probe.n_calls,
            "n_self_attention_calls": len(self_calls),
            "layers": [],
        }

        for call in cap.calls:
            causal = call.is_causal or spec.causal
            heads: List[Dict[str, object]] = []
            for h in range(call.query.shape[0]):
                res = M.analyse_head(
                    call.query[h],
                    call.key[h],
                    value=call.value[h] if call.value is not None else None,
                    causal=causal,
                    query_block=query_block,
                    key_block=key_block,
                    densities=cfg.densities,
                    deltas=resolve_deltas(),
                    max_blocks=cfg.max_blocks,
                    scale=call.scale,
                    minp_gaps=cfg.minp_gaps,
                    group_strides=spec.group_strides,
                )
                if res is not None:
                    tiles = T.analyse_tiles(
                        call.query[h], call.key[h], causal=causal,
                        query_tile=query_block, key_tile=key_block,
                        gaps=cfg.minp_gaps, scale=call.scale,
                    )
                    heads.append({"head": h, **res,
                                  **({"tiles": tiles} if tiles else {})})
            if heads:
                sample_rec["layers"].append(
                    {
                        # Raw SDPA call index is not comparable across samples:
                        # Qwen's vision tower fires thousands of calls first, and
                        # its offset shifts with video resolution, so the same LM
                        # layer appeared as 4112 in one clip and 8144 in another.
                        # `depth` is the rank among the layers actually sampled.
                        "depth": len(sample_rec["layers"]),
                        "call_index": call.index,
                        "is_causal": causal,
                        "had_mask": call.had_mask,
                        "n_query": int(call.query.shape[1]),
                        "n_key": int(call.key.shape[1]),
                        "head_ids": list(call.heads),
                        "heads": heads,
                    }
                )

        record["samples"].append(sample_rec)
        produced += 1
        del cap

    # Adapters may add warnings while producing samples (e.g. "inputs were
    # synthetic"), so re-read them after the loop, not only after load().
    late = [w for w in getattr(adapter, "load_warnings", []) if w not in load_warnings]
    if late:
        record.setdefault("warnings", []).extend(late)
        for warning in late:
            print(f"[qk_sparse] INPUTS: {warning}", flush=True)
        record["synthetic_inputs"] = any("synthetic" in w for w in late)

    record["deltas"] = list(resolve_deltas())
    if produced == 0:
        record.setdefault("warnings", []).append("no sample produced usable captures")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.adapter}.json"
    path.write_text(json.dumps(record, indent=2))
    record["_path"] = str(path)
    return record
