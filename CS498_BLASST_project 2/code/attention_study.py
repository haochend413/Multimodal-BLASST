# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect bounded attention summaries for Attention Lab, with optional exact score shards."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

import blasst_attn as ba

REGIONS = ("T→T", "V→T", "T→V", "V→V")


class AttentionRecorder:
    """Observe selected query heads. Regions classify individual tokens, including mixed tiles."""

    def __init__(self, visual: torch.Tensor, layers: set[int], heads: set[int],
                 raw_dir: Path | None = None, max_tiles: int = 200000) -> None:
        self.visual = visual.cpu().bool()
        self.layers = layers
        self.heads = heads
        self.raw_dir = raw_dir
        self.max_tiles = max_tiles
        self.views: dict[str, dict] = {}
        self.tile_count = 0
        self.raw_index = 0
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
        for head in sorted(self.heads):
            if head >= scores.shape[1]:
                raise ValueError(f"head {head} exceeds model's {scores.shape[1]} query heads")
            identity = f"{phase}:{step}:{layer}:{head}"
            view = self.views.setdefault(identity, {
                "id": identity, "phase": phase, "step": step, "layer": layer, "head": head,
                "key_length": key_length, "tiles": [], "regions": {
                    r: {"count": 0, "sum": 0.0, "abs_sum": 0.0, "mass": 0.0,
                        "hist": [0] * 80, "underflow": 0, "overflow": 0} for r in REGIONS}})
            s = scores[0, head].detach().float().cpu()
            probabilities = torch.softmax(s, -1)
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
                view["tiles"].append({
                    "q": q_start, "k": key_start, "rows": rows, "cols": key_end - key_start,
                    "mean": float(values.mean()), "abs": float(values.abs().mean()),
                    "max": float(values.max()), "mass": float(probabilities[:, key_start:key_end].sum() / rows),
                    "margin": float(margin[0, head, tile_index]),
                    "skipped": bool(skip[0, head, tile_index]),
                    "q_visual": float(q_visual.float().mean()),
                    "k_visual": float(visual[key_start:key_end].float().mean()),
                })
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--text-before", default="Observe the video carefully.")
    parser.add_argument("--question", default="Describe what happens in the video.")
    parser.add_argument("--layers", default="0,7,14,21,27")
    parser.add_argument("--heads", default="0,4")
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--lambda", dest="lam", type=float, default=0)
    parser.add_argument("--decode-lambda", type=float)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-tiles", type=int, default=200000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--demo", action="store_true", help="synthetic CPU Q/K; no model downloads")
    parser.add_argument("--manifest", type=Path, help="Video-MME manifest from prepare_videomme_study.py")
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    benchmark = None
    sample_id = None
    if args.manifest is not None:
        if args.demo or args.video is not None:
            parser.error("--manifest cannot be combined with --demo or --video")
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("schema") != "blasst-benchmark-manifest-v1":
            parser.error("unsupported benchmark manifest schema")
        samples = manifest["samples"]
        if not 0 <= args.sample_index < len(samples):
            parser.error(f"sample-index must be between 0 and {len(samples) - 1}")
        sample = samples[args.sample_index]
        args.video = Path(sample["video"])
        args.text_before, args.question = sample["text_before"], sample["question"]
        benchmark, sample_id = sample["benchmark"], sample["id"]
    if args.decode_tokens < 1 or args.max_tokens < 1:
        parser.error("decode-tokens and max-tokens must be positive")
    if args.frames < 2 or args.frames % 2 or args.max_tiles < 1:
        parser.error("frames must be a positive even number >= 2; max-tiles must be positive")
    ba.set_threshold(args.lam or None)
    ba.set_decode_threshold(args.decode_lambda or None, inherit=args.decode_lambda is None)
    if args.demo:
        torch.manual_seed(98)
        length = 640
        visual = torch.zeros(length, dtype=torch.bool)
        visual[128:512] = True
        recorder = AttentionRecorder(visual, {0, 1, 2}, {0, 1}, args.raw_dir, args.max_tiles)
        ba.STATE["observer"] = recorder.observe
        from types import SimpleNamespace
        for layer in range(3):
            k, v = torch.randn(1, 2, length, 32), torch.randn(1, 2, length, 32)
            k[:, :, 0, 0] += 12
            for step in range(min(args.decode_tokens, 8) + 1):
                q = torch.randn(1, 2, length if step == 0 else 1, 32)
                q[..., 0] += 4
                ba.blasst_attention_forward(SimpleNamespace(layer_idx=layer), q, k, v, None)
                k = torch.cat((k, torch.randn(1, 2, 1, 32)), dim=2)
                v = torch.cat((v, torch.randn(1, 2, 1, 32)), dim=2)
        metadata = {"model": "Synthetic Q/K demonstration", "synthetic": True,
                    "prompt_tokens": length, "note": "Not measured model or Video-MME results"}
    else:
        if not torch.cuda.is_available():
            parser.error("Real model collection requires CUDA; use --demo for CPU verification")
        if args.video is None or not args.video.is_file():
            parser.error("--video must point to a local video file")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from vlm_prepare import read_frames, smart_resize, video_info
        duration, height, width = video_info(str(args.video))
        resized_h, resized_w = smart_resize(height, width)
        frames, _ = read_frames(str(args.video), args.frames, resized_h, resized_w)
        processor = AutoProcessor.from_pretrained(args.model)
        content = []
        if args.text_before:
            content.append({"type": "text", "text": args.text_before})
        content.extend([{"type": "video"}, {"type": "text", "text": args.question}])
        text = processor.apply_chat_template([{"role": "user", "content": content}],
                                            tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], videos=[frames], fps=[args.frames / duration],
                           return_tensors="pt").to("cuda")
        length = inputs.input_ids.shape[1]
        if length > args.max_tokens:
            parser.error(f"Prompt has {length} tokens, exceeds --max-tokens; use fewer --frames")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="cuda",
            attn_implementation={"text_config": "blasst", "vision_config": "sdpa"}).eval()
        selected_layers, selected_heads = _indices(args.layers), _indices(args.heads)
        config = model.config.text_config
        if not all(0 <= x < config.num_hidden_layers for x in selected_layers):
            parser.error("selected layer index is outside the language model")
        if not all(0 <= x < config.num_attention_heads for x in selected_heads):
            parser.error("selected query head index is outside the language model")
        visual = inputs.input_ids[0].cpu() == model.config.video_token_id
        recorder = AttentionRecorder(visual, selected_layers, selected_heads,
                                     args.raw_dir, args.max_tiles)
        ba.STATE["observer"] = recorder.observe
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.decode_tokens + 1,
                                       min_new_tokens=args.decode_tokens + 1, do_sample=False)
        metadata = {"model": args.model, "synthetic": False, "prompt_tokens": length,
                    "video": args.video.name, "frames": args.frames,
                    "answer": processor.tokenizer.decode(generated[0, length:], skip_special_tokens=True)}
    ba.STATE["observer"] = None
    if benchmark is not None:
        metadata.update(benchmark=benchmark, sample_id=sample_id)
    metadata.update(lambda_prefill=args.lam, lambda_decode=args.decode_lambda,
                    recorded_trajectory="dense" if not args.lam and not args.decode_lambda else "live_sparse",
                    score_definition="QK^T / sqrt(head_dim), causal-masked, before softmax",
                    region_definition="query modality → key modality; token pairs, not mixed tile labels",
                    sampling="selected layers and query heads only", tile_size=128)
    recorder.export(args.output, metadata)
    print(f"Wrote {len(recorder.views)} views and {recorder.tile_count} tiles to {args.output}")


if __name__ == "__main__":
    main()
