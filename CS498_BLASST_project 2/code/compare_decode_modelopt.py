# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare decode masks AND outputs against installed ModelOpt on identical Q/K/V."""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import blasst_attn as ba


def compare(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            lam: float, mask: torch.Tensor | None = None) -> dict:
    """Compare a [B,Hq,1,D] decode call; K/V are [B,Hkv,L,D]."""
    from modelopt.torch.sparsity.attention_sparsity.methods.flash_skip_softmax import (
        FlashSkipSoftmax,
    )
    from modelopt.torch.sparsity.attention_sparsity.config import SparseAttentionAttributeConfig

    method = FlashSkipSoftmax(SparseAttentionAttributeConfig(
        thresholds={"prefill": [lam], "decode": [lam]},
        br=128, bc=128, backend="pytorch", collect_stats=True,
    ).model_dump())
    repeated_k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1).float()
    repeated_v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = (q.float() @ repeated_k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    keep, _ = method.calculate_sparsity(scores)
    expected = (torch.softmax(method.apply_sparsity(scores, keep), -1).to(v.dtype)
                @ repeated_v).transpose(1, 2)
    old = {key: ba.STATE[key] for key in ("ln_lam", "a", "decode_override", "decode_enabled",
                                        "decode_lam", "decode_a", "record", "observer")}
    try:
        ba.set_threshold(lam)
        ba.set_decode_threshold(inherit=True)
        ba.STATE["observer"] = None
        ba.reset_records(record=True)
        actual, _ = ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, k, v, mask)
        margins = ba.get_decode_margins()[0][0]
        skip = margins < math.log(lam)
        ours_elements = skip.repeat_interleave(ba.TILE, dim=-1)[..., :k.shape[2]].to(q.device)
        mismatches = int((ours_elements != ~keep).sum())
        error = float((actual.float() - expected.float()).abs().max())
        ties = int((margins == math.log(lam)).sum())
        return {"lambda": lam, "key_length": k.shape[2], "mask_mismatches": mismatches,
                "max_abs_output_error": error, "threshold_ties": ties,
                "skipped_tiles": int(skip.sum()), "total_tiles": skip.numel()}
    finally:
        ba.STATE.update(old)
        ba.reset_records(record=old["record"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(98)
    results = []
    for length in (129, 256, 513, 700):
        for heads, kv_heads in ((4, 4), (8, 2)):
            q = torch.randn(1, heads, 1, 32, device=args.device)
            k = torch.randn(1, kv_heads, length, 32, device=args.device)
            v = torch.randn_like(k)
            # Different strengths exercise both retained and discarded tiles.
            q[..., 0] += 4
            k[:, :, 0, 0] += 12
            for lam in (1e-4, 0.01, 0.1, 0.5):
                row = compare(q, k, v, lam)
                row.update(query_heads=heads, kv_heads=kv_heads)
                results.append(row)
    import modelopt
    payload = {"modelopt_version": modelopt.__version__, "kind": "synthetic_decode_parity",
               "device": args.device, "dtype": "float32", "results": results,
               "passed": all(r["mask_mismatches"] == 0 and r["max_abs_output_error"] < 1e-5
                             for r in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
