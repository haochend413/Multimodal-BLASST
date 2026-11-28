# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect bounded attention summaries for Attention Lab, with optional exact score shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
from uuid import uuid4

import torch

import blasst_attn as ba
from answer_benchmarks import ANSWER_BENCHMARKS, VQAV2
from binary_tv_pipeline import BinaryTVPipeline
from binary_phase_controls import (
    DecodeBudgets, case_rng, latency_summary, modelopt_attention, random_prefill_gate, timed_generation,
)
from decode_visual_links import DecodeVisualLinks
from jl_blasst import JLPrecheck, JLStudy, ProjectedJLStudy, VISUAL_KEY_QUERY_SPLIT
from jl_gap_study import JLGapStudy, merge_sweeps
from jl_region_study import JLRegionStudy, merge_region_audits
from tile_layer_study import LayerReport, TileLayerRecorder
from tile_calibration import add_threshold_arguments, selected_calibration, write_json
from tv_guided_vv import SWEEP_SCHEMA as TV_VV_SCHEMA, TOKEN_GATE_COUNTS, VISUAL_QUERY_COUNTS, TVGuidedVV
from region_max import RegionMax
import vqa_shards

if TYPE_CHECKING:
    from transformers import Qwen2_5_VLProcessor
    from transformers.feature_extraction_utils import BatchFeature

REGIONS = ("T→T", "V→T", "T→V", "V→V")


class AttentionRecorder:
    """Observe selected query heads. Regions classify individual tokens, including mixed tiles."""

    def __init__(self, visual: torch.Tensor, layers: set[int], heads: set[int],
                 raw_dir: Path | None = None, max_tiles: int = 200000,
                 score_stream: BinaryIO | None = None) -> None:
        self.visual = visual.cpu().bool()
        self.layers = layers
        self.heads = heads
        self.raw_dir = raw_dir
        self.max_tiles = max_tiles
        self.views: dict[str, dict] = {}
        self.tile_count = 0
        self.raw_index = 0
        self.projected_study: ProjectedJLStudy | None = None
        self.score_stream = score_stream
        self.score_id = uuid4().hex if score_stream is not None else None
        if score_stream is not None:
            score_stream.write(self.score_id.encode("ascii"))
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)

    def observe(self, *, layer: int, phase: str, q_start: int, scores: torch.Tensor,
                margin: torch.Tensor, skip: torch.Tensor, ln_lambda: float,
                key_length: int) -> None:
        """Receive one query chunk; scores are [1,Hq,rows,L], before BLASST masking."""
        if layer not in self.layers:
            return
        step = -1 if phase == "prefill" else key_length - len(self.visual) - 1
        rows = scores.shape[-2]
        visual = torch.cat((self.visual, torch.zeros(max(0, key_length - len(self.visual)),
                                                     dtype=torch.bool)))[:key_length]
        q_visual = visual[q_start:q_start + rows]
        for head_index, head in enumerate(sorted(self.heads)):
            if head >= scores.shape[1]:
                raise ValueError(f"head {head} exceeds model's {scores.shape[1]} query heads")
            identity = f"{phase}:{step}:{layer}:{head}"
            view = self.views.setdefault(identity, {
                "id": identity, "phase": phase, "step": step, "layer": layer, "head": head,
                "key_length": key_length, "tiles": [], "regions": {
                    r: {"count": 0, "sum": 0.0, "abs_sum": 0.0, "mass": 0.0,
                        "hist": [0] * 80, "underflow": 0, "overflow": 0} for r in REGIONS}})
            s = scores[0, head].detach().float().cpu()
            probabilities = torch.softmax(s, -1).nan_to_num()
            if self.projected_study is not None and self.projected_study.split and phase == "prefill":
                view.setdefault("split_tiles", []).extend(
                    self.projected_study.record_split(head_index, q_start, s, probabilities, visual)
                )
            pure_visual_query = (
                phase == "prefill" and ba.STATE["vv_prefix_visual"] is not None and bool(q_visual.all())
            )
            if self.score_stream is not None:
                cols = min(key_length, q_start + rows)
                if pure_visual_query:
                    visible_keys = torch.isfinite(s).any(0).nonzero().flatten()
                    if visible_keys.numel():
                        last_tile_end = (int(visible_keys[-1]) // ba.TILE + 1) * ba.TILE
                        wide_cols = min(key_length, last_tile_end)
                        if wide_cols > cols:
                            cols = wide_cols
                            view.setdefault("score_widths", {})[q_start] = cols
                view.setdefault("score_chunks", {})[q_start] = self.score_stream.tell()
                self.score_stream.write(s[:, :cols].numpy().astype("<f4", copy=False).tobytes())
            if self.raw_dir:
                torch.save({"scores": s, "q_start": q_start, "visual": visual,
                            "layer": layer, "head": head, "phase": phase, "step": step},
                           self.raw_dir / f"scores_{self.raw_index:06d}.pt")
                self.raw_index += 1
            for key_start in range(0, key_length, ba.TILE):
                key_end = min(key_start + ba.TILE, key_length)
                block = s[:, key_start:key_end]
                valid = torch.isfinite(block)
                if not valid.any():
                    continue
                self.tile_count += 1
                if self.tile_count > self.max_tiles:
                    raise ValueError("Export exceeds --max-tiles; select fewer layers/heads or tokens")
                values = block[valid]
                tile_index = key_start // ba.TILE
                dense_weight = float(probabilities[:, key_start:key_end].sum())
                measurement = {
                    "q": q_start, "k": key_start, "rows": rows, "cols": key_end - key_start,
                    "mean": float(values.mean()), "abs": float(values.abs().mean()),
                    "max": float(values.max()), "mass": dense_weight / rows,
                    "margin": float(margin[0, head, tile_index]),
                    "skipped": bool(skip[0, head, tile_index]),
                    "q_visual": float(q_visual.float().mean()),
                    "k_visual": float(visual[key_start:key_end].float().mean()),
                }
                if self.projected_study is not None:
                    measurement.update(self.projected_study.record(
                        phase=phase, head_index=head_index, q_start=q_start, tile_index=tile_index,
                        full_margin=measurement["margin"], q_visual=q_visual[valid.any(-1)],
                        k_visual=visual[key_start:key_end][valid.any(-2)], valid_pairs=values.numel(),
                        dense_weight=dense_weight,
                    ))
                view["tiles"].append(measurement)
                for qv in (False, True):
                    for kv in (False, True):
                        region = f"{'V' if qv else 'T'}→{'V' if kv else 'T'}"
                        selection = (q_visual[:, None] == qv) & (visual[None, key_start:key_end] == kv) & valid
                        chosen = block[selection]
                        if not chosen.numel():
                            continue
                        target = view["regions"][region]
                        target["count"] += chosen.numel()
                        target["sum"] += float(chosen.sum())
                        target["abs_sum"] += float(chosen.abs().sum())
                        target["mass"] += float(probabilities[:, key_start:key_end][selection].sum())
                        inside = chosen[(chosen >= -40) & (chosen <= 40)]
                        hist = torch.histc(inside, bins=80, min=-40, max=40).long().tolist()
                        target["hist"] = [a + b for a, b in zip(target["hist"], hist)]
                        target["underflow"] += int((chosen < -40).sum())
                        target["overflow"] += int((chosen > 40).sum())

    def export(self, output: Path, metadata: dict) -> None:
        """Write strict JSON; no full score arrays are embedded in the browser dataset."""
        payload = {"schema": "blasst-attention-v1", "metadata": metadata,
                   "histogram": {"min": -40, "max": 40, "bins": 80},
                   "visual_spans": _spans(self.visual), "views": list(self.views.values())}
        if self.score_id is not None:
            payload["score_file"] = {"id": self.score_id, "bytes": self.score_stream.tell()}
        if not payload["views"]:
            raise ValueError("No attention was recorded; check selected layer indices")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")))


def _spans(visual: torch.Tensor) -> list[list[int]]:
    positions = visual.nonzero().flatten().tolist()
    spans: list[list[int]] = []
    for position in positions:
        if spans and spans[-1][1] == position:
            spans[-1][1] += 1
        else:
            spans.append([position, position + 1])
    return spans


def _indices(value: str) -> set[int]:
    return {int(x) for x in value.split(",")}


def visual_token_mask(input_ids: torch.Tensor, *, image_token_id: int,
                      video_token_id: int) -> torch.Tensor:
    """Classify image and video placeholders as visual, leaving control tokens as text."""
    return (input_ids == image_token_id) | (input_ids == video_token_id)


def prepare_inputs(processor: Qwen2_5_VLProcessor,
                   args: argparse.Namespace) -> tuple[BatchFeature, dict]:
    """Build real image/video inputs on CPU; no model weights or CUDA are needed here."""
    content = []
    if args.text_before:
        content.append({"type": "text", "text": args.text_before})
    modality = "image" if args.image is not None else "video"
    content.extend([{"type": modality}, {"type": "text", "text": args.question}])
    text = processor.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
    if args.image is not None:
        from PIL import Image
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        metadata = {"modality": "image", "image": args.image.name,
                    "image_width": image.width, "image_height": image.height,
                    "image_max_pixels": args.image_max_pixels}
    else:
        from vlm_prepare import read_frames, smart_resize, video_info
        duration, height, width = video_info(str(args.video))
        resized_h, resized_w = smart_resize(height, width)
        frames, _ = read_frames(str(args.video), args.frames, resized_h, resized_w)
        inputs = processor(text=[text], videos=[frames], fps=[args.frames / duration],
                           return_tensors="pt")
        metadata = {"modality": "video", "video": args.video.name, "frames": args.frames}
    return inputs, metadata


def _select_sample(args, parser, sample):
    if ("image" in sample) == ("video" in sample):
        parser.error("each benchmark sample must contain exactly one image or video path")
    args.image = Path(sample["image"]) if "image" in sample else None
    args.video = Path(sample["video"]) if "video" in sample else None
    args.text_before, args.question = sample["text_before"], sample["question"]
    path = args.image if args.image is not None else args.video
    if not path.is_file():
        parser.error(f"Missing input media: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-max-pixels", type=int, default=1280 * 28 * 28,
                        help="image resize pixel budget; controls the number of visual tokens")
    parser.add_argument("--text-before", default="Inspect the visual content carefully.")
    parser.add_argument("--question", default="Describe what is shown.")
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--heads", default="0,4")
    parser.add_argument("--decode-tokens", type=int, help="maximum new tokens (default: 16 for VQA v2, 128 for VizWiz-LF)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0)
    parser.add_argument("--decode-lambda", type=float)
    parser.add_argument("--prefill-region-max", action=argparse.BooleanOptionalAction, default=False,
                        help="reset the prefill skip-test running max at each text/vision key boundary")
    parser.add_argument("--decode-region-max", action="store_true",
                        help="reset the decode skip-test running max at each text/vision boundary")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-tiles", type=int, default=200000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--score-data", action="store_true",
                        help="write exact token scores to a browser-readable .scores.bin companion")
    parser.add_argument("--vv-prefix-attention", action="store_true",
                        help="use full bidirectional attention in pure V→V prefill tiles")
    tv_vv = parser.add_mutually_exclusive_group()
    tv_vv.add_argument("--tv-vv-threshold", type=float,
                       help="gate causal V→V query tiles using the preceding layer's T→V keep fraction")
    tv_vv.add_argument("--tv-vv-top-percent", dest="tv_vv_token_percent", type=float, metavar="PERCENT",
                       help="select the top percentage of individual visual query tokens")
    parser.add_argument("--prefill-vv-start-layer", type=int, metavar="INDEX",
                        help="first zero-based layer to apply --tv-vv-top-percent (default: 1)")
    tv_vv.add_argument("--tv-vv-top-tile-percent", dest="tv_vv_top_percent", type=float, metavar="PERCENT",
                       help="select the top percentage of visual query tiles by previous-layer T→V votes")
    tv_vv.add_argument("--prefill-visual-query-study", action="store_true",
                       help="compare BLASST, V→V bypass, and whole visual query bypass with the same mask")
    tv_vv.add_argument("--prefill-vv-drop-sweep", nargs="?", const="99.9,99.5,99,95,90,80,60,25",
                       metavar="PERCENTS", help="compare frequency and random V→V query-row drop rates")
    parser.add_argument("--prefill-vv-lambda-sweep", metavar="LAMBDAS",
                        help="sweep prefill lambdas across all --prefill-vv-drop-sweep percentages")
    parser.add_argument("--prefill-tile-size", type=int, choices=(32, 64, 128, 256),
                        help="square prefill tile size for --prefill-vv-drop-sweep (default: 128)")
    parser.add_argument("--prefill-vv-drop-highest", action="store_true",
                        help="drop highest-frequency rows first in the prefill frequency arm")
    tv_vv.add_argument("--decode-tv-drop-sweep", nargs="?", const="5,10,20,25,35,50,60,75,80,90,95,99,99.5,99.9",
                       metavar="PERCENTS", help="compare guided and random decode T→V key drop rates")
    tv_vv.add_argument("--tv-vv-sweep", metavar="FRACTIONS",
                       help="compare standard BLASST and comma-separated T→V relevance thresholds")
    tv_vv.add_argument("--decode-visual-links", action="store_true",
                       help="compare BLASST with decode that selects visual keys using prefill T→V links")
    tv_vv.add_argument("--decode-visual-history-study", action="store_true",
                       help="compare BLASST, fixed prefill T→V links, and links extended during decode")
    tv_vv.add_argument("--binary-tv-pipeline", action="store_true",
                       help="compare BLASST, binary T→T-guided T→V, and the full T→V-guided V→V pipeline")
    tv_vv.add_argument("--binary-tv-phase-study", action="store_true",
                       help="compare BLASST with prefill-only and decode-only binary T→T-guided T→V")
    tv_vv.add_argument("--binary-mask-study", action="store_true",
                       help="compare binary prefill V→V, growing decode T→V, and their combination")
    tv_vv.add_argument("--binary-tv-prefill-diagnostics", action="store_true",
                       help="compare linked and all-candidate staged T→V prefill")
    parser.add_argument("--matched-random", action="store_true",
                        help="add seeded random masks matched to guided QK budgets")
    parser.add_argument("--random-seed", type=int, default=20261126)
    parser.add_argument("--time-modelopt", action="store_true",
                        help="benchmark fixed-length generation against ModelOpt's PyTorch simulator")
    parser.add_argument("--vqa-accuracy", "--score-answers", action="store_true",
                        help="score paired predictions using the selected benchmark's metric")
    parser.add_argument("--matched-region-study", action="store_true",
                        help="compare four running-max policies at matched post-QK keep rates")
    parser.add_argument("--calibration-count", type=int, default=20,
                        help="number of questions before the matched-study evaluation range")
    parser.add_argument("--budget-probes", type=int, default=8, help="maximum live calibration probes per stage")
    parser.add_argument("--budget-tolerance-pp", type=float, default=1.,
                        help="allowed post-QK keep-rate error in percentage points")
    parser.add_argument("--budget-config", type=Path, help="save or reuse matched-study calibration settings")
    parser.add_argument("--budget-keep-percent", type=float, default=50.,
                        help="requested post-QK tile keep percentage in each phase")
    parser.add_argument("--gate-lambda", type=float,
                        help="prefill lambda for recording votes in fixed-gate runs (default: 1)")
    parser.add_argument("--gate-distribution", action="store_true",
                        help="run only the fixed prefill gate and count tokens and causal pairs by modality")
    parser.add_argument("--vqa-shards", type=int, help="split the whole VQA manifest into this many shards")
    parser.add_argument("--vqa-shard-index", type=int, help="zero-based index of this restartable shard")
    parser.add_argument("--jl-dim", type=int, help="audit a JL precheck at this sketch dimension")
    parser.add_argument("--jl-sweep", metavar="DIMS",
                        help="comma-separated JL widths for a paired skip-rate and gap histogram study")
    parser.add_argument("--jl-region-audit", action="store_true",
                        help="compare original, modality-packed, and per-key-region max policies")
    parser.add_argument("--tile-layer-study", action="store_true",
                        help="track full-score BLASST decisions on the same physical tile across layers")
    parser.add_argument("--score-distributions", action="store_true",
                        help="collect causal score and margin histograms by modality")
    add_threshold_arguments(parser)
    parser.add_argument("--study-max-indices", action="store_true",
                        help="track tile and running-max indices, and replay the first layer's indices")
    parser.add_argument("--jl-epsilon", type=float,
                        help="use a JL norm-product margin instead of the orthogonal residual bound")
    parser.add_argument("--jl-lambda", type=float, default=0.1,
                        help="fixed lambda for the JL audit and its exact baseline")
    parser.add_argument("--jl-seed", type=int, default=0)
    parser.add_argument("--jl-live", action="store_true",
                        help="use the precheck during generation in selected layers, for all heads")
    parser.add_argument("--jl-projected-only", action="store_true",
                        help="compare projected-only BLASST masks with full BLASST on the same inputs")
    parser.add_argument("--jl-split-tiles", choices=("query", "key", "both", VISUAL_KEY_QUERY_SPLIT),
                        help="audit modality splits; query-on-visual-keys only splits mixed query "
                             "rows against entirely visual keys")
    parser.add_argument("--demo", action="store_true", help="synthetic CPU Q/K; no model downloads")
    parser.add_argument("--manifest", type=Path, help="benchmark manifest")
    parser.add_argument("--benchmark", choices=tuple(ANSWER_BENCHMARKS), default="vqav2",
                        help="answer-evaluation protocol for paired runs")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int,
                        help="number of manifest cases to test, before splitting into shards")
    args = parser.parse_args()
    args.answer_protocol = ANSWER_BENCHMARKS[args.benchmark]
    if args.decode_tokens is None:
        args.decode_tokens = args.answer_protocol.decode_tokens
    if args.answer_protocol is not VQAV2:
        args.vqa_accuracy = True
    binary_study = args.binary_tv_pipeline or args.binary_tv_phase_study or args.binary_tv_prefill_diagnostics
    if (args.matched_random or args.time_modelopt) and not args.binary_mask_study:
        parser.error("--matched-random and --time-modelopt require --binary-mask-study")
    if args.matched_random and args.time_modelopt:
        parser.error("run matched random accuracy and fixed-length timing separately")
    if args.time_modelopt and args.vqa_shards is not None:
        parser.error("run --time-modelopt as one process on one GPU")
    decode_drop_study = args.decode_tv_drop_sweep is not None
    decode_link_study = args.decode_visual_links or args.decode_visual_history_study or decode_drop_study
    drop_study = args.prefill_vv_drop_sweep is not None
    if args.prefill_tile_size is not None and not drop_study:
        parser.error("--prefill-tile-size requires --prefill-vv-drop-sweep")
    if drop_study or decode_drop_study:
        option = "decode_tv_drop_sweep" if decode_drop_study else "prefill_vv_drop_sweep"
        try:
            rates = list(dict.fromkeys(float(rate) for rate in getattr(args, option).split(",")))
            if not all(0 < rate < 100 for rate in rates):
                raise ValueError
        except ValueError:
            parser.error(f"--{option.replace('_', '-')} expects comma-separated drop percentages in (0, 100)")
        setattr(args, option, rates)
    if args.prefill_vv_drop_highest and not drop_study:
        parser.error("--prefill-vv-drop-highest requires --prefill-vv-drop-sweep")
    if args.prefill_vv_lambda_sweep is not None:
        if not drop_study:
            parser.error("--prefill-vv-lambda-sweep requires --prefill-vv-drop-sweep")
        if args.lam != 0 or args.decode_lambda is None or not 0 < args.decode_lambda <= 1:
            parser.error("--prefill-vv-lambda-sweep requires a fixed --decode-lambda in (0, 1] and omits --lambda")
        try:
            lambdas = list(dict.fromkeys(float(value) for value in args.prefill_vv_lambda_sweep.split(",")))
            if not all(0 < value <= 1 for value in lambdas):
                raise ValueError
        except ValueError:
            parser.error("--prefill-vv-lambda-sweep expects comma-separated lambdas in (0, 1]")
        args.prefill_vv_lambda_sweep = lambdas
    query_study = args.prefill_visual_query_study or args.binary_mask_study or drop_study
    if decode_link_study or binary_study or query_study:
        flag = "--decode-visual-links"
        if args.decode_visual_history_study:
            flag = "--decode-visual-history-study"
        if binary_study:
            flag = "--binary-tv-phase-study" if args.binary_tv_phase_study else "--binary-tv-pipeline"
        if query_study:
            flag = "--prefill-visual-query-study"
        if args.binary_tv_prefill_diagnostics:
            flag = "--binary-tv-prefill-diagnostics"
        if args.binary_mask_study:
            flag = "--binary-mask-study"
        if drop_study:
            flag = "--prefill-vv-drop-sweep"
        if decode_drop_study:
            flag = "--decode-tv-drop-sweep"
        minimum = 1 if args.prefill_visual_query_study or drop_study else 2
        needs_fixed_lambdas = args.binary_mask_study or args.binary_tv_prefill_diagnostics or drop_study or decode_drop_study
        if needs_fixed_lambdas and not args.prefill_vv_lambda_sweep and (
            not 0 < args.lam <= 1 or not 0 < (args.lam if args.decode_lambda is None else args.decode_lambda) <= 1
        ):
            parser.error(f"{flag} requires positive prefill and decode lambdas in (0, 1]")
        if not args.manifest or args.demo or args.decode_tokens < minimum:
            parser.error(f"{flag} requires a benchmark manifest and --decode-tokens of at least {minimum}")
        if (args.matched_region_study or args.gate_distribution or args.prefill_region_max
                or args.decode_region_max or args.vv_prefix_attention or args.jl_dim is not None
                or args.jl_sweep or args.jl_region_audit or args.tile_layer_study
                or args.score_distributions or args.score_data or args.raw_dir):
            parser.error(f"{flag} compares against causal BLASST with shared running maxima")
        args.vqa_accuracy = True
    has_top_k = args.tv_vv_token_percent is not None or args.tv_vv_top_percent is not None
    if args.prefill_vv_start_layer is not None:
        if args.prefill_vv_start_layer < 1:
            parser.error("--prefill-vv-start-layer must be at least 1 to allow a preceding vote layer")
        if args.tv_vv_token_percent is None or args.matched_region_study or args.gate_distribution:
            parser.error("--prefill-vv-start-layer requires live --tv-vv-top-percent")
    args.tv_vv_top_accuracy = (args.vqa_accuracy and has_top_k
                               and not (args.matched_region_study or args.gate_distribution))
    if args.tv_vv_top_accuracy and (args.prefill_region_max or args.decode_region_max):
        parser.error("The paired top-k accuracy run uses shared running maxima in both variants")
    if args.matched_region_study or args.gate_distribution:
        from matched_region_study import validate_args

        if args.gate_lambda is None:
            args.gate_lambda = 1.
        validate_args(args, parser)
        args.vqa_accuracy = True
    elif args.budget_config is not None:
        parser.error("--budget-config requires --matched-region-study")
    elif args.gate_lambda is not None:
        parser.error("--gate-lambda requires a fixed-gate run. Use --lambda for live T→V votes.")
    requested_sample_count = args.sample_count
    if args.sample_count is None:
        args.sample_count = 1
    if (args.prefill_region_max or args.decode_region_max) and (
        args.jl_dim is not None or args.jl_sweep or args.jl_region_audit
        or args.tile_layer_study or args.score_distributions
    ):
        parser.error("The live region-max switches cannot be used with audits that compute their own running max")
    if args.vqa_accuracy and not (args.manifest and (args.tv_vv_sweep is not None or args.tv_vv_top_accuracy
                                                    or decode_link_study or binary_study or query_study
                                                    or args.matched_region_study or args.gate_distribution)):
        parser.error("Answer scoring requires a compatible benchmark manifest and a paired study")
    args.vqa_shard = None
    if args.vqa_shards is not None or args.vqa_shard_index is not None:
        if not args.vqa_accuracy or args.vqa_shards is None or args.vqa_shard_index is None:
            parser.error("use both VQA shard flags with --vqa-accuracy")
    if args.tv_vv_sweep is not None:
        try:
            args.tv_vv_sweep = sorted({float(value) for value in args.tv_vv_sweep.split(",")})
        except ValueError:
            parser.error("--tv-vv-sweep expects comma-separated fractions")
    tv_vv_thresholds = args.tv_vv_sweep or (
        [args.tv_vv_threshold] if args.tv_vv_threshold is not None else []
    )
    if any(value is not None and not 0 <= value <= 100
           for value in (args.tv_vv_token_percent, args.tv_vv_top_percent)):
        parser.error("Top-k percentages must be in [0, 100]")
    if tv_vv_thresholds or has_top_k:
        if any(not 0 <= threshold <= 1 for threshold in tv_vv_thresholds):
            parser.error("T→V relevance thresholds must be in [0, 1]")
        if (args.vv_prefix_attention or args.jl_dim is not None or args.jl_sweep
                or args.tile_layer_study or args.score_distributions):
            parser.error("T→V-guided V→V compares live causal BLASST masks; omit separate mask audits")
        if not (args.matched_region_study or args.gate_distribution) and not 0 < args.lam <= 1:
            parser.error("set the BLASST vote threshold with --lambda in (0, 1]")
    if (args.tv_vv_sweep or args.tv_vv_top_accuracy) and (args.score_data or args.raw_dir):
        parser.error("The paired T→V-guided study collects answers and tile counts")
    if args.vv_prefix_attention and (args.jl_dim is not None or args.jl_sweep or args.jl_region_audit):
        parser.error("JL audits use causal QK reconstruction; omit JL flags with --vv-prefix-attention")
    if args.sample_index < 0:
        parser.error("--sample-index must be nonnegative")
    if args.study_max_indices and not args.tile_layer_study:
        parser.error("--study-max-indices requires --tile-layer-study")
    calibration = selected_calibration(args, parser)
    if (calibration or args.study_lambda is not None) and not (args.tile_layer_study or args.score_distributions):
        parser.error("study thresholds require --tile-layer-study or --score-distributions")
    if args.score_distributions and (args.tile_layer_study or args.vv_prefix_attention or args.jl_sweep
                                     or args.jl_region_audit or args.jl_dim is not None or args.lam
                                     or args.decode_lambda or args.score_data or args.raw_dir):
        parser.error("--score-distributions collects histograms from a dense causal trajectory")
    if args.tile_layer_study:
        if (args.jl_sweep or args.jl_region_audit or args.jl_dim is not None or args.lam
                or args.decode_lambda or args.score_data or args.raw_dir):
            parser.error("--tile-layer-study uses dense attention and its own report; set --study-lambda")
        if args.layers != "all" and not args.demo and len(_indices(args.layers)) < 2:
            parser.error("--tile-layer-study requires at least two layers")
    if args.jl_sweep is not None:
        try:
            args.jl_sweep = sorted({int(dim) for dim in args.jl_sweep.split(",")})
        except ValueError:
            parser.error("--jl-sweep expects comma-separated integer widths")
        if args.jl_sweep[0] <= 0 or (args.demo and args.jl_sweep[-1] >= 32):
            parser.error("JL sweep widths must be positive and below 32 for the demo")
        if not 0 < args.jl_lambda <= 1:
            parser.error("--jl-lambda must be in (0, 1]")
        if (args.jl_dim is not None or args.jl_epsilon is not None or args.jl_live or
                args.jl_projected_only or args.jl_split_tiles or args.jl_region_audit or
                args.score_data or args.raw_dir):
            parser.error("--jl-sweep produces its own report; omit other JL modes and raw-score flags")
        if args.lam or args.decode_lambda:
            parser.error("--jl-sweep uses a dense trajectory; set the audit threshold with --jl-lambda")
    if args.jl_region_audit:
        if args.jl_dim is None:
            parser.error("--jl-region-audit requires --jl-dim")
        if (args.jl_epsilon is not None or args.jl_live or args.jl_projected_only or
                args.jl_split_tiles or args.score_data or args.raw_dir or args.lam or args.decode_lambda):
            parser.error("--jl-region-audit uses a dense trajectory and its own report; set --jl-lambda")
    paired_audit = (
        args.jl_sweep or args.jl_region_audit or args.tile_layer_study or args.score_distributions
        or args.tv_vv_sweep or args.tv_vv_top_accuracy or args.matched_region_study or args.gate_distribution
        or decode_link_study or binary_study or query_study
    )
    if args.sample_count < 1:
        parser.error("--sample-count must be positive")
    if args.sample_count > 1 and not (
        (args.manifest and (args.jl_projected_only or paired_audit)) or (args.demo and paired_audit)
    ):
        parser.error("batch collection requires a manifest audit or a supported synthetic audit")
    if args.sample_count > 1 and (args.score_data or args.raw_dir):
        parser.error("use a single sample when exporting raw scores")
    benchmark = None
    sample_id = None
    if args.manifest is not None:
        if args.demo or args.video is not None or args.image is not None:
            parser.error("--manifest cannot be combined with --demo, --video or --image")
        manifest_bytes = args.manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema") != "blasst-benchmark-manifest-v1":
            parser.error("unsupported benchmark manifest schema")
        samples = manifest["samples"]
        if args.answer_protocol is not VQAV2 and requested_sample_count is None:
            reserved = args.calibration_count if args.matched_region_study else 0
            args.sample_count = len(samples) - args.sample_index - reserved
            if args.sample_count < 1:
                parser.error("No questions remain after the selected offset and calibration range")
        if args.matched_region_study:
            args.budget_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if args.vqa_shards is not None:
            args.vqa_shard = dict(index=args.vqa_shard_index, count=args.vqa_shards, manifest_questions=len(samples),
                                  manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
            selected = len(samples) - args.sample_index if requested_sample_count is None else requested_sample_count
            if args.sample_index or selected != len(samples):
                args.vqa_shard.update(selection_start=args.sample_index, selection_count=selected)
            try:
                start, stop = vqa_shards.shard_bounds(args.vqa_shard)
            except ValueError as error:
                parser.error(str(error))
            args.sample_index, args.sample_count = start, stop - start
        del manifest_bytes, manifest
        end = args.sample_index + args.sample_count + (args.calibration_count if args.matched_region_study else 0)
        if not 0 <= args.sample_index < end <= len(samples):
            parser.error(f"requested sample range is outside the {len(samples)} manifest pairs")
        samples = samples[args.sample_index:end]
        if args.vqa_accuracy:
            try:
                args.answer_protocol.validate(samples)
            except ValueError as error:
                parser.error(str(error))
        sample = samples[0]
        if args.sample_count == 1 and not paired_audit:
            _select_sample(args, parser, sample)
        benchmark, sample_id = sample["benchmark"], sample["id"]
    if args.decode_tokens < 1 or args.max_tokens < 1:
        parser.error("decode-tokens and max-tokens must be positive")
    if args.frames < 2 or args.frames % 2 or args.max_tiles < 1:
        parser.error("frames must be a positive even number >= 2; max-tiles must be positive")
    if args.image_max_pixels < 4 * 28 * 28:
        parser.error("image-max-pixels must be at least 3136")
    if (args.jl_epsilon is not None or args.jl_live or args.jl_projected_only) and args.jl_dim is None:
        parser.error("JL modes require --jl-dim")
    if args.jl_projected_only and (args.jl_epsilon is not None or args.jl_live):
        parser.error("--jl-projected-only cannot be combined with --jl-epsilon or --jl-live")
    if args.jl_split_tiles is not None and not args.jl_projected_only:
        parser.error("--jl-split-tiles requires --jl-projected-only")
    if args.jl_live and (args.score_data or args.raw_dir):
        parser.error("--jl-live cannot export full scores; use audit mode for score exports")
    if args.jl_dim is not None:
        if not 0 < args.jl_lambda <= 1:
            parser.error("--jl-lambda must be in (0, 1]")
        if args.demo and args.jl_dim >= 32:
            parser.error("--jl-dim must be below 32 in the synthetic demo")
        try:
            args.jl_probe = JLPrecheck(args.jl_dim, epsilon=args.jl_epsilon, seed=args.jl_seed)
        except ValueError as error:
            parser.error(str(error))
    else:
        args.jl_probe = None
    if not args.demo:
        if args.manifest is None:
            if (args.image is None) == (args.video is None):
                parser.error("provide exactly one --image or --video, or use --manifest")
            media_path = args.image if args.image is not None else args.video
            if not media_path.is_file():
                parser.error(f"Missing input media: {media_path}")
        if not torch.cuda.is_available():
            parser.error("Real model collection requires CUDA; use --demo for CPU verification")
    if args.matched_region_study or args.gate_distribution:
        from matched_region_study import run

        try:
            with vqa_shards.claim(args.output):
                run(args, parser, samples, collect, _select_sample, ba.STATE, ba.TILE)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
    elif args.tv_vv_sweep or args.tv_vv_top_accuracy or decode_link_study or binary_study or query_study:
        _collect_tv_vv_sweep(args, parser, samples if args.manifest else [None] * args.sample_count)
    elif args.score_distributions:
        _collect_score_distributions(args, parser, samples if args.manifest else [None] * args.sample_count, calibration)
    elif args.tile_layer_study:
        _collect_layer_study(args, parser, samples if args.manifest else [None] * args.sample_count, calibration)
    elif paired_audit:
        merge = merge_region_audits if args.jl_region_audit else merge_sweeps
        _collect_paired_audit(args, parser, samples if args.manifest else [None] * args.sample_count, merge)
    elif args.sample_count > 1:
        _collect_batch(args, parser, samples)
    elif args.score_data:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        score_path = args.output.with_suffix(".scores.bin")
        if score_path == args.output:
            parser.error("--output must name the JSON summary, not the score companion")
        with score_path.open("wb") as score_stream:
            collect(args, parser, benchmark, sample_id, score_stream)
        print(f"Wrote token scores to {score_path}")
    else:
        collect(args, parser, benchmark, sample_id)


def _add_pair_counts(totals_by_group, groups):
    for group, counts in groups.items():
        totals = totals_by_group[group]
        for name, count in counts.items():
            totals[name] += count
        totals["pairs_observed"] += 1
        totals["pairs_with_false_skips"] += int(counts["false_skips"] > 0)


def _collect_score_distributions(args, parser, samples, calibration):
    from score_distribution_study import ScoreDistributions

    study = ScoreDistributions(args.study_lambda or .1, calibration)
    for index, sample in enumerate(samples, start=args.sample_index):
        run = argparse.Namespace(**dict(vars(args), sample_index=index, score_study=study))
        benchmark, sample_id = None, None
        if sample is not None:
            _select_sample(run, parser, sample)
            benchmark, sample_id = sample["benchmark"], sample["id"]
        run.score_case = sample_id or str(index)
        metadata = collect(run, parser, benchmark, sample_id)
        study.cases[run.score_case].update(model=metadata["model"], synthetic=metadata["synthetic"])
        write_json(args.output, study.report())
        print(f"Collected score distributions for {len(study.cases)}/{len(samples)} cases", flush=True)
    print(f"Wrote score distributions to {args.output}")


def _normalized_answer(answer):
    return " ".join("".join(char if char.isalnum() else " " for char in answer.casefold()).split())


def _collect_tv_vv_sweep(args, parser, samples):
    if getattr(args, "prefill_vv_lambda_sweep", None):
        try:
            with vqa_shards.claim(args.output):
                _collect_prefill_lambda_sweep(args, parser, samples)
        except ValueError as error:
            parser.error(str(error))
        return
    if getattr(args, "vqa_shard", None):
        try:
            with vqa_shards.claim(args.output):
                return _run_tv_vv_sweep(args, parser, samples)
        except ValueError as error:
            parser.error(str(error))
    return _run_tv_vv_sweep(args, parser, samples)


def _collect_prefill_lambda_sweep(args, parser, samples):
    report = dict(
        schema="blasst-prefill-vv-lambda-sweep-v1", benchmark=args.answer_protocol.name, model=args.model,
        prefill_vv_lambda_sweep=args.prefill_vv_lambda_sweep,
        prefill_vv_drop_sweep=args.prefill_vv_drop_sweep,
        prefill_vv_drop_order="highest" if args.prefill_vv_drop_highest else "lowest",
        prefill_tile_size=args.prefill_tile_size or ba.TILE, decode_tile_size=ba.TILE,
        decode_lambda=args.decode_lambda, random_seed=args.random_seed, sample_index=args.sample_index,
        pairs_requested=len(samples),
        reports={}, summary={},
    )
    if len(args.prefill_vv_drop_sweep) == 1:
        report["prefill_vv_drop_percent"] = args.prefill_vv_drop_sweep[0]
    for lam in args.prefill_vv_lambda_sweep:
        output = args.output.with_name(f"{args.output.stem}-lambda{lam}{args.output.suffix}")
        print(f"Running prefill lambda {lam:g} (decode {args.decode_lambda:g})", flush=True)
        run = argparse.Namespace(**dict(vars(args), lam=lam, output=output, prefill_vv_lambda_sweep=None))
        child = _collect_tv_vv_sweep(run, parser, samples)
        report["reports"][str(lam)] = output.name
        report["summary"][str(lam)] = child["summary"]
    write_json(args.output, report)
    print(f"Wrote prefill lambda sweep summary to {args.output}")


def _run_tv_vv_sweep(args, parser, samples):
    protocol = getattr(args, "answer_protocol", VQAV2)
    top_percent = getattr(args, "tv_vv_top_percent", None)
    token_percent = getattr(args, "tv_vv_token_percent", None)
    has_top_k = top_percent is not None or token_percent is not None
    extend_decode_links = getattr(args, "decode_visual_history_study", False)
    decode_drop_rates = getattr(args, "decode_tv_drop_sweep", None)
    decode_links = getattr(args, "decode_visual_links", False) or extend_decode_links or bool(decode_drop_rates)
    phase_study = getattr(args, "binary_tv_phase_study", False)
    diagnostics = getattr(args, "binary_tv_prefill_diagnostics", False)
    binary = getattr(args, "binary_tv_pipeline", False) or phase_study or diagnostics
    mask_study = getattr(args, "binary_mask_study", False)
    timing = getattr(args, "time_modelopt", False)
    matched_random = getattr(args, "matched_random", False)
    query_study = getattr(args, "prefill_visual_query_study", False)
    prefill_drop_rates = getattr(args, "prefill_vv_drop_sweep", None)
    drop_highest = getattr(args, "prefill_vv_drop_highest", False)
    drop_rates = prefill_drop_rates or decode_drop_rates
    drop_phase = "decode" if decode_drop_rates else "prefill"
    variants = [("baseline", None)]
    if drop_rates:
        for rate in drop_rates:
            name = f"{drop_phase}_drop_{str(rate).removesuffix('.0')}"
            variants.extend(((name, rate), (f"{name}_random", rate)))
    elif mask_study:
        variants.extend((("prefill_binary", "prefill"), ("decode_binary", "decode"),
                         ("combined_binary", "combined")))
        if matched_random:
            variants.extend((f"{name}_random", phase) for name, phase in tuple(variants[1:]))
        if timing:
            variants.append(("modelopt_simulator", None))
    elif diagnostics:
        variants.extend((("linked_prefill", "prefill"), ("all_candidate_prefill", "all_candidate")))
    elif query_study:
        variants.extend((("visual_vv_only", "vv"), ("whole_visual_query", "whole")))
    elif phase_study:
        variants.extend((("binary_tv_prefill", "prefill"), ("binary_tv_decode", "decode")))
    elif binary:
        variants.extend((("binary_tv", "tv"), ("binary_tv_vv", "tv_vv")))
    elif decode_links:
        variants.append(("decode_links", True))
        if extend_decode_links:
            variants.append(("decode_links_growing", "generated_history"))
    elif token_percent is not None:
        variants.append((f"top_token_{token_percent:g}", token_percent))
    elif top_percent is None:
        variants.extend((str(value), value) for value in args.tv_vv_sweep)
    else:
        variants.append((f"top_{top_percent:g}", top_percent))
    report = dict(schema=TV_VV_SCHEMA, model=args.model, synthetic=args.demo,
                  tile_size=getattr(args, "prefill_tile_size", None) or ba.TILE,
                  blasst_lambda=args.lam, decode_lambda=args.decode_lambda, pairs_requested=len(samples),
                  answer_metric="normalized full-response exact match to any supplied reference",
                  variants=dict(variants), cases=[])
    region_phases = [phase for phase in ("prefill", "decode") if getattr(args, f"{phase}_region_max", False)]
    for phase in region_phases:
        report[f"{phase}_max_policies"] = {
            name: "shared" if threshold is None else "per_region" for name, threshold in variants
        }
    if drop_rates:
        report.update(random_seed=args.random_seed, image_max_pixels=args.image_max_pixels,
                      max_prompt_tokens=args.max_tokens)
        report["decode_tv_drop_sweep" if decode_drop_rates else "prefill_vv_drop_sweep"] = drop_rates
        if drop_highest:
            report["prefill_vv_drop_order"] = "highest"
        report["variant_labels"] = {"baseline": "Original BLASST"}
        for name, rate in variants[1:]:
            method = "Random" if name.endswith("_random") else "T→V frequency"
            if drop_highest and not name.endswith("_random"):
                method = "Highest T→V frequency"
            report["variant_labels"][name] = f"{method} drop {rate:g}%"
    elif mask_study:
        report["binary_mask_study"] = True
        report.update(image_max_pixels=args.image_max_pixels, max_prompt_tokens=args.max_tokens)
        report["variant_labels"] = {
            "baseline": "Original BLASST", "prefill_binary": "Binary V→V prefill only",
            "decode_binary": "Growing T→V decode only", "combined_binary": "Binary prefill and decode",
        }
        if matched_random:
            report["random_seed"] = args.random_seed
            report["variant_labels"].update({name: f"Matched random ({phase})"
                                             for name, phase in variants if name.endswith("_random")})
        if timing:
            from importlib.metadata import version

            modelopt_attention(args.lam, args.decode_lambda or args.lam, ba.TILE)
            report["variant_labels"]["modelopt_simulator"] = "ModelOpt PyTorch simulator"
            report["timing"] = dict(batch_size=1, warmup_requests_per_variant=1,
                                    forced_new_tokens=args.decode_tokens, gpu=torch.cuda.get_device_name(),
                                    torch=torch.__version__, modelopt=version("nvidia-modelopt"))
    elif diagnostics:
        report["binary_tv_pipeline"] = True
        report["binary_tv_prefill_diagnostics"] = True
        report.update(image_max_pixels=args.image_max_pixels, max_prompt_tokens=args.max_tokens)
        report["variant_labels"] = {"baseline": "Original BLASST", "linked_prefill": "Linked T→V prefill",
                                    "all_candidate_prefill": "Staged prefill with all visual candidates"}
    elif query_study:
        report["prefill_visual_query_study"] = True
        report["variant_labels"] = {"baseline": "Original BLASST", "visual_vv_only": "Bypass V→V only",
                                    "whole_visual_query": "Bypass whole visual query"}
    elif phase_study:
        report["binary_tv_pipeline"] = True
        report["variant_labels"] = {"baseline": "Original BLASST", "binary_tv_prefill": "T→T-guided T→V in prefill",
                                    "binary_tv_decode": "T→T-guided T→V in decode"}
    elif binary:
        report["binary_tv_pipeline"] = True
        report["variant_labels"] = {"baseline": "Original BLASST", "binary_tv": "T→T-guided T→V",
                                    "binary_tv_vv": "T→T-guided T→V + T→V-guided V→V"}
    elif decode_links:
        report["decode_visual_link_study"] = True
        report["variant_labels"] = {"baseline": "Original BLASST", "decode_links": "Decode with prefill T→V links"}
        if extend_decode_links:
            report["variant_labels"]["decode_links"] = "Decode with fixed prefill T→V links"
            report["variant_labels"]["decode_links_growing"] = "Decode with prefill and generated-token T→V links"
    elif has_top_k:
        amount, unit = ((token_percent, "tokens") if token_percent is not None else (top_percent, "tile rows"))
        report["variant_labels"] = {"baseline": "Original BLASST", variants[1][0]: f"Top {amount:g}% visual {unit}"}
        if (start_layer := getattr(args, "prefill_vv_start_layer", None)) is not None:
            report["prefill_vv_start_layer"] = start_layer
            report["variant_labels"][variants[1][0]] += f" from layer {start_layer}"
        if token_percent is not None:
            report["token_top_k"] = True
        for phase in ("prefill", "decode"):
            report[f"{phase}_max_policies"] = dict.fromkeys(dict(variants), "shared")
    measured_phases = ("prefill", "decode") if has_top_k or decode_links or binary or query_study or mask_study or drop_rates else region_phases
    if args.vqa_accuracy:
        report["answer_metric"] = protocol.metric
        if protocol is not VQAV2 or mask_study or diagnostics or drop_rates:
            report["benchmark"] = protocol.name
        report["answer_protocol"] = dict(instruction=protocol.instruction,
                                         max_new_tokens=args.decode_tokens, min_new_tokens=args.decode_tokens if timing else 0)
    shard = getattr(args, "vqa_shard", None)
    if shard:
        report.update(vqa_shard=shard, image_max_pixels=args.image_max_pixels, max_prompt_tokens=args.max_tokens)
        try:
            vqa_shards.resume(report, args.output, samples, args.sample_index)
        except (ValueError, KeyError, OSError) as error:
            parser.error(f"{args.output}: {error}")
    totals = {name: defaultdict(int) for name, _ in variants}
    for case in report["cases"]:
        vqa_shards.accumulate(totals, case, protocol=protocol)
    completed = len(report["cases"])
    for index, sample in enumerate(samples[completed:], start=args.sample_index + completed):
        benchmark, sample_id = (sample["benchmark"], sample["id"]) if sample else ({}, str(index))
        case = dict(sample_index=index, sample_id=sample_id, results={})
        references = benchmark.get("reference_answers") or [benchmark.get("ground_truth", "")]
        answers = {normalized for answer in references if (normalized := _normalized_answer(answer))}
        if args.vqa_accuracy:
            case["question_id"] = benchmark["question_id"]
            if not shard:
                case.update(question=benchmark["question_text"], reference_answers=references)
        query_gate = None
        guided_traces = {}
        for name, value in variants:
            # An ungated vote trace provides the baseline's causal V→V denominator.
            threshold = value if not (has_top_k or decode_links or binary or query_study or mask_study or drop_rates) else None
            run = argparse.Namespace(**dict(vars(args), sample_index=index, tv_vv_threshold=threshold,
                                            tv_vv_top_percent=value if top_percent is not None else None,
                                            tv_vv_token_percent=value if token_percent is not None else None,
                                            decode_visual_links=decode_links and name != "baseline",
                                            decode_visual_link_study=decode_links,
                                            extend_decode_links=bool(decode_drop_rates) or name == "decode_links_growing",
                                            token_gate=value if query_study else None,
                                            gate_replay=query_gate if name == "whole_visual_query" else None,
                                            binary_tv_pipeline=binary,
                                            binary_tv_mode=value if binary else None))
            if prefill_drop_rates:
                run.token_gate = "vv"
                run.token_drop_percent = value or 0
                run.random_drop = value is None or name.endswith("_random")
            if decode_drop_rates:
                run.decode_drop_percent = value
                run.random_decode_drop = name.endswith("_random")
            if mask_study:
                prefill = value in ("prefill", "combined")
                decode = value in ("decode", "combined")
                run.token_gate = "vv" if prefill else None
                run.pipeline_decode_links = decode
                run.use_modelopt = name == "modelopt_simulator"
                run.decode_budgets = None
                if matched_random and not name.endswith("_random") and decode:
                    run.decode_budgets = DecodeBudgets()
                if name.endswith("_random"):
                    guided = guided_traces[name.removesuffix("_random")]
                    rng = case_rng(args.random_seed, index, name)
                    if prefill:
                        run.gate_replay = random_prefill_gate(guided["visual_positions"], guided["prefill_gate"], rng)
                    if decode:
                        run.decode_budgets = DecodeBudgets(rng, guided["decode_budgets"])
            for phase in region_phases:
                setattr(run, f"{phase}_region_max", threshold is not None)
            if sample:
                _select_sample(run, parser, sample)
            if args.vqa_accuracy:
                run.question = f"{benchmark['question_text']}\n{protocol.instruction}"
            before = {phase: ba.STATE["phase_stats"].get(phase, {}).copy() for phase in measured_phases}
            if timing and index == args.sample_index:
                collect(run, parser, benchmark, sample_id)
            metadata = collect(run, parser, benchmark, sample_id)
            if prefill_drop_rates:
                trace = {key: item for key, item in metadata["tv_guided_vv"].items() if key != "layers"}
            elif mask_study:
                trace = metadata.get("timing", {}) if timing else {
                    **metadata.get("tv_guided_vv", {}), **metadata.get("decode_visual_links", {}),
                }
                trace.update(metadata.get("timing", {}))
                if matched_random and value and not name.endswith("_random"):
                    guided_traces[name] = metadata
                if name.endswith("_random") and run.gate_replay is not None:
                    trace["random_prefill_rows_reassigned"] = sum(
                        len(set(dropped) - set(guided["prefill_gate"][layer]))
                        for layer, dropped in run.gate_replay.items()
                    )
                if name.endswith("_random") and run.decode_budgets is not None:
                    trace["random_decode_tiles_reassigned"] = run.decode_budgets.tiles_reassigned
                    trace["random_decode_unmatched_head_steps"] = run.decode_budgets.tail_head_steps
                    trace["random_decode_guided_only_head_steps"] = sum(
                        (len(rows) - len(run.decode_budgets.steps[layer])) * len(rows[0][0])
                        for layer, rows in run.decode_budgets.source.steps.items()
                        if len(rows) > len(run.decode_budgets.steps[layer])
                    )
            elif binary:
                trace = metadata.get("binary_tv_pipeline", {})
                if diagnostics and "candidate_audit" in metadata:
                    trace["candidate_audit"] = metadata["candidate_audit"]
            elif decode_links:
                trace = metadata.get("decode_visual_links", {})
            else:
                trace = metadata["tv_guided_vv"]
            if query_study and value is not None:
                observed_gate = {layer["layer"]: layer["dropped_token_positions"] for layer in trace["layers"]}
                if value == "vv":
                    query_gate = observed_gate
                elif observed_gate != query_gate:
                    raise ValueError("The whole-query gate differs from the recorded V→V-only mask")
                trace = {key: item for key, item in trace.items() if key != "layers"}
            if timing:
                case["results"][name] = dict(trace, generated_tokens=metadata["generated_tokens"])
                continue
            answer = metadata.get("answer")
            normalized = _normalized_answer(answer) if answer is not None else None
            result = dict(answer=answer, **trace)
            if "generated_tokens" in metadata:
                result["generated_tokens"] = metadata["generated_tokens"]
            for phase in measured_phases:
                after = ba.STATE["phase_stats"].get(phase, {})
                result[f"{phase}_tiles"] = after.get("causal", 0) - before[phase].get("causal", 0)
                result[f"{phase}_tiles_skipped"] = after.get("skipped", 0) - before[phase].get("skipped", 0)
                if phase == "prefill" and (has_top_k or query_study or mask_study or prefill_drop_rates):
                    result["prefill_qk_bypassed"] = after.get("qk_bypassed", 0) - before[phase].get("qk_bypassed", 0)
            if args.vqa_accuracy:
                result[protocol.case_score] = protocol.score(answer, references)
            else:
                match = normalized in answers if answers and normalized is not None else None
                result["reference_exact_match"] = match
                if match is not None:
                    totals[name]["answers_scored"] += 1
                    totals[name]["reference_exact_matches"] += match
            if name != "baseline" and answer is not None:
                base = case["results"]["baseline"]
                result["answer_changed"] = normalized != _normalized_answer(base["answer"])
            case["results"][name] = (
                {key: result[key] for key in vqa_shards.CASE_FIELDS if key in result} if shard else result
            )
            if not args.vqa_accuracy:
                totals[name]["pure_vv_tiles"] += trace["pure_vv_tiles"]
                totals[name]["vv_tiles_bypassed"] += trace["vv_tiles_bypassed"]
                if "answer_changed" in result:
                    totals[name]["answers_changed"] += result["answer_changed"]
                for phase in region_phases:
                    for suffix in ("tiles", "tiles_skipped"):
                        key = f"{phase}_{suffix}"
                        totals[name][key] += result[key]
        if args.vqa_accuracy and not timing:
            vqa_shards.accumulate(totals, case, protocol=protocol)
        report["cases"].append(case)
        done = len(report["cases"])
        if not shard or done % vqa_shards.CHECKPOINT_INTERVAL == 0 or done == len(samples):
            if timing:
                report["latency"] = latency_summary(report["cases"], dict(variants))
            else:
                report["summary"] = vqa_shards.summarize(totals, protocol=protocol) if args.vqa_accuracy else totals
            write_json(args.output, report)
            policy = "binary pipeline" if binary else "decode visual links" if decode_links else "T→V-guided V→V"
            if query_study:
                policy = "visual-query gates"
            if mask_study:
                policy = "binary prefill and decode gates"
            if prefill_drop_rates:
                policy = "frequency and random V→V query drops"
            if decode_drop_rates:
                policy = "guided and random T→V key drops"
            print(f"Compared {policy} for {done}/{len(samples)} cases", flush=True)
    print(f"Wrote paired attention report to {args.output}")
    if timing:
        for name, stats in report["latency"].items():
            print(f"{name}: median first token {stats['first_token_seconds_p50'] * 1000:.1f} ms · "
                  f"median request {stats['generation_seconds_p50'] * 1000:.1f} ms · "
                  f"decode {stats['decode_tokens_per_second']:.2f} tokens/s")
        return
    if args.vqa_accuracy:
        if not shard:
            predictions = {
                name: [{"question_id": case["question_id"], "answer": case["results"][name]["answer"]}
                       for case in report["cases"]]
                for name, _ in variants
            }
            vqa_shards.save_predictions(args.output, predictions)
        if "summary" not in report:
            report["summary"] = vqa_shards.summarize(totals, protocol=protocol)
        for name, result in report["summary"].items():
            print(f"{name}: {protocol.format_score(result)} on {result['answers_scored']} questions")
    return report


def _collect_layer_study(args, parser, samples, calibration):
    report = None
    for index, sample in enumerate(samples, start=args.sample_index):
        run = argparse.Namespace(**dict(vars(args), sample_index=index))
        benchmark, sample_id = None, None
        if sample is not None:
            _select_sample(run, parser, sample)
            benchmark, sample_id = sample["benchmark"], sample["id"]
        metadata, case = collect(run, parser, benchmark, sample_id)
        try:
            if report is None:
                report = LayerReport(metadata["model"], metadata["synthetic"], args.study_lambda or 0.1,
                                     ba.TILE, track_indices=args.study_max_indices, calibration=calibration,
                                     mask_policy=metadata.get("attention_mask_policy", "causal"))
            report.add(case)
        except ValueError as error:
            parser.error(str(error))
        report.write(args.output, len(samples))
        print(f"Collected layer trajectories for {len(report.cases)}/{len(samples)} cases", flush=True)
    print(f"Wrote the cross-layer report to {args.output}")


def _collect_paired_audit(args, parser, samples, merge):
    report, cases = None, []
    for index, sample in enumerate(samples, start=args.sample_index):
        run = argparse.Namespace(**dict(vars(args), sample_index=index))
        if sample is not None:
            _select_sample(run, parser, sample)
            benchmark, sample_id = sample["benchmark"], sample["id"]
        else:
            benchmark, sample_id = None, None
        result = collect(run, parser, benchmark, sample_id)
        cases.append({"sample_index": index, "sample_id": sample_id, "phases": result["phases"]})
        report = merge(report, result)
        _write_paired_audit(args.output, report, args.sample_count, cases)
        if args.sample_count > 1:
            print(f"Collected JL results for {len(cases)}/{args.sample_count} pairs", flush=True)
    print(f"Wrote the JL report to {args.output}")


def _write_paired_audit(output, report, pairs_requested, cases):
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(output.suffix + ".tmp")
    payload = dict(report, pairs_requested=pairs_requested, cases=cases)
    pending.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    pending.replace(output)


def _collect_batch(args, parser, samples):
    """Write individual audits and refresh the aggregate after every completed pair."""
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    details = output.parent / f"{output.stem}-samples"
    phases = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    split_prefill = defaultdict(lambda: defaultdict(int))
    completed = []
    for index, sample in enumerate(samples, start=args.sample_index):
        _select_sample(args, parser, sample)
        args.output = details / f"{index:05d}.json"
        metadata = collect(args, parser, sample["benchmark"], sample["id"])
        projected = metadata["jl_projected"]
        for phase, groups in projected["phases"].items():
            _add_pair_counts(phases[phase], groups)
        pair = {
            "sample_index": index, "sample_id": sample["id"],
            "audit": str(args.output.relative_to(output.parent)),
            "phases": {phase: groups["all"] for phase, groups in projected["phases"].items()},
        }
        if "split_prefill" in projected:
            _add_pair_counts(split_prefill, projected["split_prefill"])
            pair["split_prefill"] = projected["split_prefill"]["all"]
        completed.append(pair)
        summary = {
            "schema": "blasst-jl-projected-batch-v1", "model": args.model,
            "manifest": str(args.manifest), "sample_index": args.sample_index,
            "pairs_requested": len(samples), "pairs_completed": len(completed),
            "dimension": projected["dimension"], "seed": projected["seed"],
            "lambda": projected["lambda"], "layers": sorted(_indices(args.layers)),
            "heads": sorted(_indices(args.heads)), "decode_tokens": args.decode_tokens,
            "phases": phases, "samples": completed,
        }
        if "split_prefill" in projected:
            summary.update(split=projected["split"], split_prefill=split_prefill)
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        pending.replace(output)
    print(f"Wrote the aggregate for {len(completed)} pairs to {output}")


def collect(args, parser, benchmark, sample_id, score_stream: BinaryIO | None = None) -> dict:
    old_observers = ba.STATE["observer"], ba.STATE["qkv_hook"]
    old_prefix = ba.STATE["vv_prefix_visual"]
    old_tv_vv = ba.STATE["tv_guided_vv"]
    old_links = ba.STATE["decode_visual_links"]
    old_binary = ba.STATE["binary_tv_pipeline"]
    old_regions = ba.STATE["prefill_region_max"], ba.STATE["decode_region_max"]
    precision = torch.get_float32_matmul_precision()
    old_collect_stats = ba.STATE["collect_stats"]
    try:
        if getattr(args, "time_modelopt", False):
            ba.STATE["collect_stats"] = False
        if args.jl_probe is not None or args.jl_sweep:
            torch.set_float32_matmul_precision("highest")
        return _collect(args, parser, benchmark, sample_id, score_stream)
    finally:
        ba.STATE["collect_stats"] = old_collect_stats
        ba.STATE["observer"], ba.STATE["qkv_hook"] = old_observers
        ba.STATE["vv_prefix_visual"] = old_prefix
        ba.STATE["tv_guided_vv"] = old_tv_vv
        ba.STATE["decode_visual_links"] = old_links
        ba.STATE["binary_tv_pipeline"] = old_binary
        ba.STATE["prefill_region_max"], ba.STATE["decode_region_max"] = old_regions
        if args.jl_probe is not None or args.jl_sweep:
            torch.set_float32_matmul_precision(precision)


def _attach_study(args, recorder):
    if getattr(args, "binary_mask_study", False):
        ba.STATE["observer"] = ba.STATE["qkv_hook"] = None
        if args.use_modelopt:
            ba.STATE["qkv_hook"] = modelopt_attention(args.lam, args.decode_lambda or args.lam, ba.TILE)
        if args.token_gate:
            ba.STATE["tv_guided_vv"] = TVGuidedVV(
                recorder.visual, ba.TILE, None, replay=args.gate_replay, token_gate=args.token_gate,
            )
        if args.pipeline_decode_links:
            ba.STATE["decode_visual_links"] = DecodeVisualLinks(
                recorder.visual, ba.TILE, extend_history=True, budgets=args.decode_budgets,
                collect_counts=not args.time_modelopt,
            )
        return None
    if getattr(args, "binary_tv_pipeline", False):
        if args.binary_tv_mode is not None:
            mode = args.binary_tv_mode
            ba.STATE["binary_tv_pipeline"] = BinaryTVPipeline(
                recorder.visual, ba.TILE, gate_vv=mode == "tv_vv",
                guide_prefill=mode != "decode", guide_decode=mode not in ("prefill", "all_candidate"),
                all_candidates=mode == "all_candidate",
                audit_candidates=mode == "prefill" and getattr(args, "binary_tv_prefill_diagnostics", False),
            )
        ba.STATE["observer"] = ba.STATE["qkv_hook"] = None
        return None
    if getattr(args, "decode_visual_link_study", False):
        if args.decode_visual_links:
            ba.STATE["decode_visual_links"] = DecodeVisualLinks(
                recorder.visual, ba.TILE, extend_history=getattr(args, "extend_decode_links", False),
                drop_percent=getattr(args, "decode_drop_percent", None),
                random_drop=getattr(args, "random_decode_drop", False),
                drop_seed=f"{getattr(args, 'random_seed', 0)}:{args.sample_index}",
            )
        ba.STATE["observer"] = ba.STATE["qkv_hook"] = None
        return None
    if args.prefill_region_max or args.decode_region_max:
        regions = RegionMax(recorder.visual, ba.TILE)
        ba.STATE["prefill_region_max"] = regions if args.prefill_region_max else None
        ba.STATE["decode_region_max"] = regions if args.decode_region_max else None
    top_percent = getattr(args, "tv_vv_top_percent", None)
    token_percent = getattr(args, "tv_vv_token_percent", None)
    paired = (args.tv_vv_sweep or getattr(args, "tv_vv_top_accuracy", False)
              or getattr(args, "prefill_visual_query_study", False)
              or getattr(args, "prefill_vv_drop_sweep", None))
    if (args.tv_vv_threshold is not None or top_percent is not None or token_percent is not None or paired
            or args.matched_region_study or args.gate_distribution):
        tile_size = getattr(args, "prefill_tile_size", None) or ba.TILE
        ba.STATE["tv_guided_vv"] = TVGuidedVV(recorder.visual, tile_size, args.tv_vv_threshold,
                                            replay=getattr(args, "gate_replay", None), top_percent=top_percent,
                                            token_percent=token_percent, token_gate=getattr(args, "token_gate", None),
                                            start_layer=getattr(args, "prefill_vv_start_layer", None) or 1,
                                            token_drop_percent=getattr(args, "token_drop_percent", None),
                                            random_drop=getattr(args, "random_drop", False),
                                            drop_highest=getattr(args, "prefill_vv_drop_highest", False),
                                            drop_seed=f"{getattr(args, 'random_seed', 0)}:{getattr(args, 'sample_index', 0)}")
        ba.STATE["qkv_hook"] = None
        if paired or args.matched_region_study or args.gate_distribution:
            ba.STATE["observer"] = None
        return None
    if args.score_distributions:
        ba.STATE["observer"] = args.score_study.observer(recorder, ba.TILE, args.score_case)
        ba.STATE["qkv_hook"] = None
        return None
    if args.tile_layer_study:
        study = TileLayerRecorder(recorder, track_indices=args.study_max_indices)
        ba.STATE["observer"] = study.observe
        ba.STATE["qkv_hook"] = None
        return study
    if args.jl_region_audit:
        study = JLRegionStudy(args.jl_dim, args.jl_lambda, args.jl_seed, recorder.visual,
                              recorder.layers, recorder.heads)
        ba.STATE["observer"] = None
        ba.STATE["qkv_hook"] = study.observe
        return study
    if args.jl_sweep:
        study = JLGapStudy(args.jl_sweep, args.jl_lambda, args.jl_seed, recorder.visual,
                           recorder.layers, recorder.heads)
        ba.STATE["observer"] = study.record
        ba.STATE["qkv_hook"] = study.observe
        return study
    if args.jl_probe is None:
        return None
    if args.jl_projected_only:
        study = ProjectedJLStudy(args.jl_probe, recorder.layers, recorder.heads, args.jl_lambda,
                                 split=args.jl_split_tiles, visual=recorder.visual)
        recorder.projected_study = study
    else:
        study = JLStudy(args.jl_probe, recorder.visual, recorder.layers, recorder.heads,
                        args.jl_lambda, ba.TILE, live=args.jl_live)
    ba.STATE["qkv_hook"] = study.observe
    return study


@lru_cache(maxsize=2)
def _load_processor(model_name, image_max_pixels):
    from transformers import AutoProcessor

    options = ({"min_pixels": 4 * 28 * 28, "max_pixels": image_max_pixels}
               if image_max_pixels is not None else {})
    return AutoProcessor.from_pretrained(model_name, **options)


@lru_cache(maxsize=1)
def _load_model(model_name):
    from transformers import Qwen2_5_VLForConditionalGeneration

    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}).eval()


def _collect(args, parser, benchmark, sample_id, score_stream: BinaryIO | None) -> dict:
    ba.set_threshold(args.lam or None)
    ba.set_decode_threshold(args.decode_lambda or None, inherit=args.decode_lambda is None)
    ba.STATE["vv_prefix_visual"] = None
    ba.STATE["tv_guided_vv"] = None
    ba.STATE["decode_visual_links"] = None
    ba.STATE["binary_tv_pipeline"] = None
    ba.STATE["prefill_region_max"] = ba.STATE["decode_region_max"] = None
    if args.demo:
        start_layer = getattr(args, "prefill_vv_start_layer", None)
        if start_layer is not None and start_layer >= 3:
            parser.error("--prefill-vv-start-layer is outside the three-layer demo")
        sweep = args.jl_sweep or args.jl_region_audit or args.tile_layer_study or args.score_distributions or args.tv_vv_sweep
        torch.manual_seed(98 + args.sample_index if sweep else 98)
        length = 640
        visual = torch.zeros(length, dtype=torch.bool)
        tv_demo = (args.tv_vv_sweep or args.tv_vv_threshold is not None or args.tv_vv_top_percent is not None
                   or getattr(args, "tv_vv_token_percent", None) is not None)
        if args.jl_region_audit:
            visual[15:544] = True
        elif tv_demo:
            visual[128:544] = True
        else:
            visual[128:512] = True
        recorder = AttentionRecorder(visual, {0, 1, 2}, {0, 1}, args.raw_dir, args.max_tiles,
                                     score_stream)
        if args.vv_prefix_attention:
            ba.STATE["vv_prefix_visual"] = visual
        ba.STATE["observer"] = None if args.jl_live else recorder.observe
        jl_study = _attach_study(args, recorder)
        from types import SimpleNamespace
        for layer in range(3):
            k, v = torch.randn(1, 2, length, 32), torch.randn(1, 2, length, 32)
            k[:, :, 0, 0] += 12
            for step in range(min(args.decode_tokens, 8) + 1):
                q = torch.randn(1, 2, length if step == 0 else 1, 32)
                q[..., 0] += 4
                if tv_demo and step == 0:
                    q[..., 544:, :] = 0
                    q[..., 544:, -1] = math.sqrt(32)
                    k[..., -1] = 3
                    k[..., 0, -1] = 8
                    for head, visual_tile in enumerate((layer + 1, (layer + 1) % 3 + 1)):
                        k[:, head, visual_tile * ba.TILE:(visual_tile + 1) * ba.TILE, -1] = 9
                ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, v, None)
                k = torch.cat((k, torch.randn(1, 2, 1, 32)), dim=2)
                v = torch.cat((v, torch.randn(1, 2, 1, 32)), dim=2)
        metadata = {"model": "Synthetic Q/K demonstration", "synthetic": True,
                    "prompt_tokens": length, "note": "Not measured model or Video-MME results"}
    else:
        processor = _load_processor(args.model, args.image_max_pixels if args.image is not None else None)
        inputs, media_metadata = prepare_inputs(processor, args)
        inputs = inputs.to("cuda")
        length = inputs.input_ids.shape[1]
        if length > args.max_tokens:
            parser.error(f"Prompt has {length} tokens, exceeds --max-tokens; reduce --frames or --image-max-pixels")
        model = _load_model(args.model)
        config = model.config.text_config
        start_layer = getattr(args, "prefill_vv_start_layer", None)
        if start_layer is not None and start_layer >= config.num_hidden_layers:
            parser.error("--prefill-vv-start-layer is outside the language model")
        selected_layers = (set(range(config.num_hidden_layers))
                           if args.tile_layer_study and args.layers == "all" else _indices(args.layers))
        selected_heads = _indices(args.heads)
        if not all(0 <= x < config.num_hidden_layers for x in selected_layers):
            parser.error("selected layer index is outside the language model")
        if not all(0 <= x < config.num_attention_heads for x in selected_heads):
            parser.error("selected query head index is outside the language model")
        visual = visual_token_mask(inputs.input_ids[0].cpu(),
                                   image_token_id=model.config.image_token_id,
                                   video_token_id=model.config.video_token_id)
        if not visual.any():
            raise ValueError("Processor produced no image/video tokens; check the media input")
        recorder = AttentionRecorder(visual, selected_layers, selected_heads,
                                     args.raw_dir, args.max_tiles, score_stream)
        if args.vv_prefix_attention:
            ba.STATE["vv_prefix_visual"] = visual.to(inputs.input_ids.device)
        ba.STATE["observer"] = None if args.jl_live else recorder.observe
        jl_study = _attach_study(args, recorder)
        timing = {}
        with torch.inference_mode():
            if getattr(args, "time_modelopt", False):
                generated, timing = timed_generation(model, inputs, args.decode_tokens)
            elif args.vqa_accuracy:
                generated = model.generate(**inputs, max_new_tokens=args.decode_tokens,
                                           min_new_tokens=0, do_sample=False)
            else:
                generated = model.generate(**inputs, max_new_tokens=args.decode_tokens + 1,
                                           min_new_tokens=args.decode_tokens + 1, do_sample=False)
        metadata = {"model": args.model, "synthetic": False, "prompt_tokens": length,
                    "generated_tokens": generated.shape[-1] - length,
                    **media_metadata,
                    "answer": processor.tokenizer.decode(generated[0, length:], skip_special_tokens=True)}
    if getattr(args, "binary_mask_study", False):
        metadata["timing"] = timing
        if args.time_modelopt:
            return metadata
        if (policy := ba.STATE["tv_guided_vv"]) is not None:
            metadata["tv_guided_vv"] = {
                key: sum(layer[key] for layer in policy.layers) for key in TOKEN_GATE_COUNTS
            }
            if args.matched_random and args.gate_replay is None:
                metadata["visual_positions"] = visual.nonzero().flatten().tolist()
                metadata["prefill_gate"] = {row["layer"]: row["dropped_token_positions"] for row in policy.layers}
        if (links := ba.STATE["decode_visual_links"]) is not None:
            metadata["decode_visual_links"] = dict(links.counts)
            metadata["decode_budgets"] = args.decode_budgets
        return metadata
    if args.score_distributions:
        return metadata
    if getattr(args, "binary_tv_pipeline", False):
        if (binary := ba.STATE["binary_tv_pipeline"]) is not None:
            metadata["binary_tv_pipeline"] = dict(binary.counts)
            if binary.candidate_audit:
                metadata["candidate_audit"] = binary.candidate_audit
        return metadata
    if getattr(args, "decode_visual_link_study", False):
        if (links := ba.STATE["decode_visual_links"]) is not None:
            metadata["decode_visual_links"] = dict(links.counts)
        return metadata
    if args.decode_region_max:
        metadata["decode_max_policy"] = "per_region"
    if args.prefill_region_max:
        metadata["prefill_max_policy"] = "per_region"
    if (policy := ba.STATE["tv_guided_vv"]) is not None:
        metadata["tv_guided_vv"] = dict(threshold=policy.threshold, layers=policy.layers,
                                        pure_vv_tiles=sum(layer["pure_vv_tiles"] for layer in policy.layers),
                                        vv_tiles_bypassed=sum(layer["vv_tiles_bypassed"] for layer in policy.layers))
        if policy.top_percent is not None:
            metadata["tv_guided_vv"]["top_percent"] = policy.top_percent
        if policy.token_percent is not None:
            metadata["tv_guided_vv"]["top_token_percent"] = policy.token_percent
            if getattr(args, "prefill_vv_start_layer", None) is not None:
                metadata["tv_guided_vv"]["prefill_vv_start_layer"] = policy.first_gated_layer
        if policy.token_percent is not None or policy.token_gate is not None:
            counts = (*TOKEN_GATE_COUNTS, *VISUAL_QUERY_COUNTS) if policy.token_gate else TOKEN_GATE_COUNTS
            metadata["tv_guided_vv"].update({key: sum(layer[key] for layer in policy.layers)
                                              for key in counts})
        if policy.token_drop_percent is not None:
            metadata["tv_guided_vv"]["visual_query_tokens_gateable"] = sum(
                layer["visual_query_tokens_gateable"] for layer in policy.layers
            )
        if args.gate_distribution:
            metadata["gate_distribution"] = policy.distribution()
        if (args.tv_vv_sweep or getattr(args, "tv_vv_top_accuracy", False)
                or getattr(args, "prefill_visual_query_study", False)
                or getattr(args, "prefill_vv_drop_sweep", None)
                or args.matched_region_study or args.gate_distribution):
            return metadata
    if args.vv_prefix_attention:
        metadata["attention_mask_policy"] = "vv_prefix"
    if args.tile_layer_study:
        return metadata, jl_study.case(args.sample_index, sample_id)
    if args.jl_sweep or args.jl_region_audit:
        return jl_study.summary(metadata)
    if jl_study is not None:
        key = "jl_projected" if args.jl_projected_only else "jl_precheck"
        metadata[key] = jl_study.summary()
    if benchmark is not None:
        metadata.update(benchmark=benchmark, sample_id=sample_id)
    metadata.update(lambda_prefill=args.lam, lambda_decode=args.decode_lambda,
                    recorded_trajectory="live_sparse" if args.jl_live or args.lam or args.decode_lambda else "dense",
                    score_definition="QK^T / sqrt(head_dim), causal-masked, before softmax",
                    region_definition="query modality → key modality; token pairs, not mixed tile labels",
                    sampling="selected layers and query heads only", tile_size=128)
    if args.vv_prefix_attention:
        metadata["score_definition"] = "QK^T / sqrt(head_dim), bidirectional for pure V→V prefill tiles, before softmax"
    if args.jl_live:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema": "blasst-jl-live-v1", "metadata": metadata},
                                         indent=2, allow_nan=False))
        print(f"Wrote JL decisions from live attention to {args.output}")
    else:
        recorder.export(args.output, metadata)
        print(f"Wrote {len(recorder.views)} views and {recorder.tile_count} tiles to {args.output}")
    return metadata


if __name__ == "__main__":
    main()
