# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
BLASST skip-softmax attention, simulated in PyTorch and registered with
Hugging Face transformers' AttentionInterface under the name "blasst".

What it does (prefill, i.e. more than one query token):
  * take 128 query rows at a time ("chunk"),
  * compute their scaled scores against ALL keys, apply the causal mask,
  * split the key axis into 128-wide tiles and take each tile's row maxima,
  * walk the tiles left to right keeping a running row max (cummax, which
    includes the current tile, exactly as BLASST's online softmax does),
  * skip a tile only if EVERY row in the chunk is below its running max by
    more than ln(lambda):   max_rows(tile_max - running_max) < ln(lambda),
  * set skipped tiles to -inf, do a normal softmax, multiply by V.
  Because a tile that holds some row's running max has (tile_max - running_max) = 0
  for that row, it can never be skipped, so every row keeps at least one tile.

Decode uses 1 x 128 tiles over the cached keys, with the same skip rule.
set_decode_threshold(enabled=False) restores the original dense decode path.

The skip decision is "simulated": QK^T is still computed for every tile, which
is also true of the real BLASST kernel; here the softmax/PV work is not actually
saved either, we only measure what WOULD be skipped and make the output match
what a skipping kernel would return.

Threshold control (module-level STATE, see set_threshold):
  * fixed lambda           -> STATE["ln_lam"] = ln(lambda)
  * calibrated lambda = a/L -> STATE["a"] = a   (L = number of keys)
  * no skipping             -> set_threshold(None)   (ln_lam = -inf)

Recording (for teammates B and C):
  * STATE["record"] = True saves, per layer, the tile MARGIN
        margin[h, q_tile, k_tile] = max over the 128 rows of (tile_max - running_max)
    a tile is skipped at threshold lambda  <=>  margin < ln(lambda).
    The margin does not depend on lambda, so ONE forward pass with skipping off
    gives the skip mask for EVERY lambda offline (see masks_from_margins).
    Non-causal tiles have margin = -inf.
  * STATE["dump_qk_layers"] = [list of layer indices] additionally stores Q and
    K (before GQA repetition, bf16, CPU) for those layers.

Shapes follow the transformers contract (verified on transformers 4.57 and 5.9):
  query [B, Hq, Sq, D], key/value [B, Hkv, Skv, D] (K/V NOT yet repeated for GQA),
  return (attn_output [B, Sq, Hq, D], None).
Assumes batch entries are unpadded (batch size 1 is what we use). If an
attention_mask is passed it is honoured in addition to our own causal mask.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

TILE = 128
NEG_INF = float("-inf")

STATE: Dict = {
    "ln_lam": NEG_INF,       # ln(lambda) for a fixed threshold; -inf = never skip
    "a": None,               # if not None, lambda = a / Skv (BLASST's calibrated rule)
    "record": False,         # save per-layer tile margins
    "records": {},           # layer_idx -> list of [B, H, n_kt] chunk margins (CPU fp32)
    "record_meta": {},       # layer_idx -> dict(Sq, Skv, offset)
    "dump_qk_layers": [],    # layer indices whose Q, K (pre-repeat) are stored
    "qk": {},                # layer_idx -> (q, k) bf16 CPU
    "stats": {},             # layer_idx -> {"skipped": [H] long, "causal": [H] long}
    "collect_stats": True,
    "decode_override": False,
    "decode_enabled": True,
    "decode_lam": None,
    "decode_a": None,
    "phase_stats": {},
    "observer": None,         # optional synchronous callback; must not retain GPU tensors
    "qkv_hook": None,         # optional post-RoPE callback returning an output or None
    "vv_prefix_visual": None, # opt-in prompt mask for bidirectional V→V tiles
    "tv_guided_vv": None,    # prompt-local previous-layer text vote policy
    "decode_records": {},    # layer -> list of [B,H,1,key_tiles], one tensor per step
}


# ----------------------------------------------------------------------------- control
def set_threshold(lam: Optional[float] = None, a: Optional[float] = None) -> None:
    """lam: fixed lambda (None -> no skipping). a: calibrated rule lambda = a / L (overrides lam)."""
    if lam is not None and not (0.0 < lam <= 1.0):
        raise ValueError(f"lambda must be in (0, 1], got {lam}")
    STATE["a"] = a
    STATE["ln_lam"] = NEG_INF if lam is None else math.log(lam)


def reset_records(record: Optional[bool] = None, dump_qk_layers: Optional[List[int]] = None) -> None:
    STATE["records"] = {}
    STATE["record_meta"] = {}
    STATE["qk"] = {}
    STATE["stats"] = {}
    STATE["phase_stats"] = {}
    STATE["decode_records"] = {}
    if record is not None:
        STATE["record"] = record
    if dump_qk_layers is not None:
        STATE["dump_qk_layers"] = list(dump_qk_layers)


def set_decode_threshold(lam: float | None = None, a: float | None = None,
                         *, enabled: bool = True, inherit: bool = False) -> None:
    """Configure decode separately; inherit=True uses the prefill threshold."""
    if lam is not None and not 0 < lam <= 1:
        raise ValueError("decode lambda must be in (0, 1]")
    if a is not None and (not math.isfinite(a) or a <= 0):
        raise ValueError("decode scale factor must be finite and positive")
    STATE.update(decode_override=not inherit, decode_enabled=enabled,
                 decode_lam=lam, decode_a=a)


def get_margins() -> Dict[int, torch.Tensor]:
    """Prefill margins [B,H,query_tiles,key_tiles], fp32 CPU; decode is recorded separately."""
    return {l: torch.stack(chunks, dim=2) for l, chunks in STATE["records"].items()}


def get_decode_margins() -> dict[int, list[torch.Tensor]]:
    """Decode margins per layer and step; key tile count can grow with the cache."""
    return STATE["decode_records"]


def sparsity_from_stats() -> Dict[str, object]:
    """Overall and per-layer fraction of causal tiles skipped in the recorded prefill(s)."""
    per_layer = {}
    tot_s = tot_c = 0
    for l, s in sorted(STATE["stats"].items()):
        sk, ca = int(s["skipped"].sum()), int(s["causal"].sum())
        per_layer[l] = sk / max(ca, 1)
        tot_s += sk
        tot_c += ca
    return {"overall": tot_s / max(tot_c, 1), "per_layer": per_layer,
            "per_head": {l: (s["skipped"].double() / s["causal"].clamp(min=1).double())
                         for l, s in STATE["stats"].items()}}


def masks_from_margins(margin: torch.Tensor, lam: float) -> torch.Tensor:
    """Boolean skip mask for threshold lam from a recorded margin tensor (any shape)."""
    return margin < math.log(lam)


def causal_tiles(Sq: int, Skv: int, offset: int) -> torch.Tensor:
    """[n_q_tiles, n_k_tiles] bool: tile has at least one allowed (query, key) pair."""
    n_qt = math.ceil(Sq / TILE)
    n_kt = math.ceil(Skv / TILE)
    q_last = torch.arange(n_qt) * TILE + (TILE - 1)
    q_last = torch.minimum(q_last, torch.tensor(Sq - 1)) + offset   # last query position in tile
    k_first = torch.arange(n_kt) * TILE
    return k_first[None, :] <= q_last[:, None]


def score_tiles(scores, allowed, mask=None):
    """Apply the attention masks and partition scores along the key axis."""
    scores = scores.masked_fill(~allowed, NEG_INF)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, NEG_INF)
        else:
            scores = scores + mask.to(torch.float32)
    pad = -scores.shape[-1] % TILE
    if pad:
        scores = F.pad(scores, (0, pad), value=NEG_INF)
    return scores.unflatten(-1, (-1, TILE))


def tile_margins(tiles):
    """Reduce each tile against its own row-wise running maximum."""
    return row_tile_margins(tiles.amax(dim=-1)).amax(dim=-2)


def row_tile_margins(tile_max):
    running = torch.cummax(tile_max, dim=-1).values
    return torch.where(torch.isfinite(running), tile_max - running, NEG_INF)


def mask_chunk(mask, start, end, query_len, key_len):
    if mask is None:
        return None
    mask = mask[..., -query_len:, :key_len]
    return mask if mask.shape[-2] == 1 else mask[..., start:end, :]


# ----------------------------------------------------------------------------- the kernel
def blasst_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
):
    B, Hq, Sq, D = query.shape
    Hkv, Skv = key.shape[1], key.shape[2]
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)
    layer_idx = getattr(module, "layer_idx", -1)

    if STATE["dump_qk_layers"] and layer_idx in STATE["dump_qk_layers"] and Sq > 1:
        STATE["qk"][layer_idx] = (query.detach().to(torch.bfloat16).cpu(),
                                  key.detach().to(torch.bfloat16).cpu())

    # ---------------- decode: one query token, standard SDPA ----------------
    phase = "decode" if Sq == 1 else "prefill"
    qkv_hook = STATE["qkv_hook"]
    if qkv_hook is not None:
        override = qkv_hook(layer=layer_idx, phase=phase, query=query, key=key, value=value,
                            attention_mask=attention_mask, scaling=scaling)
        if override is not None:
            return override, None

    n_rep = Hq // Hkv
    if n_rep > 1:                      # GQA: same ordering as transformers' repeat_kv
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    if Sq == 1 and not STATE["decode_enabled"]:
        mask = attention_mask
        if mask is not None and mask.dtype != torch.bool:
            mask = mask[:, :, -1:, :Skv].to(query.dtype)
        elif mask is not None:
            mask = mask[:, :, -1:, :Skv]
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    # ---------------- prefill: chunked skip-softmax ----------------
    ln_lam = math.log(STATE["a"] / Skv) if STATE["a"] is not None else STATE["ln_lam"]
    if phase == "decode" and STATE["decode_override"]:
        decode_a, decode_lam = STATE["decode_a"], STATE["decode_lam"]
        ln_lam = (math.log(decode_a / Skv) if decode_a is not None else
                  NEG_INF if decode_lam is None else math.log(decode_lam))
    # Cap lambda at 1 (ln_lam <= 0). Above 1 even the tile holding a row's max (margin 0) would be
    # skipped, the row would lose every tile and softmax would return NaN. Calibrated a/L hits this
    # when L is short (e.g. a = 1000, L = 650).
    ln_lam = min(ln_lam, 0.0)
    offset = Skv - Sq                 # queries are the LAST Sq positions of the key axis
    n_kt = math.ceil(Skv / TILE)
    key32 = key.to(torch.float32)
    kpos = torch.arange(Skv, device=query.device)
    out = torch.empty_like(query)
    vv_prefix = STATE["vv_prefix_visual"] if phase == "prefill" else None
    policy = STATE["tv_guided_vv"] if phase == "prefill" else None
    tv_layer = None
    if policy is not None:
        if B != 1 or Sq != Skv or len(policy.visual) != Skv or layer_idx < 0:
            raise ValueError("T→V-guided V→V requires one uncached prompt and indexed layers")
        if vv_prefix is not None:
            raise ValueError("T→V-guided V→V uses causal attention")
        tv_layer = policy.start_layer(layer_idx, query.device)
    if vv_prefix is not None:
        if B != 1 or Sq != Skv or vv_prefix.shape != (Skv,):
            raise ValueError("V→V prefix attention requires one unpadded, uncached prompt")
        visual = vv_prefix.to(device=query.device, dtype=torch.bool)
        pure_visual_tiles = F.pad(visual, (0, -Skv % TILE), value=True).reshape(-1, TILE).all(-1)

    stats = STATE["stats"].setdefault(layer_idx, {
        "skipped": torch.zeros(Hq, dtype=torch.long), "causal": torch.zeros(Hq, dtype=torch.long)})
    rec = (STATE["records"].setdefault(layer_idx, [])
           if STATE["record"] and phase == "prefill" else None)
    if STATE["record"] and phase == "prefill":
        STATE["record_meta"][layer_idx] = {"Sq": Sq, "Skv": Skv, "offset": offset}

    for r0 in range(0, Sq, TILE):
        r1 = min(r0 + TILE, Sq)
        q32 = query[:, :, r0:r1].to(torch.float32)
        qpos = torch.arange(r0, r1, device=query.device) + offset
        allowed = kpos[None, :] <= qpos[:, None]                            # [r, Skv] causal
        if vv_prefix is not None:
            full_vv = pure_visual_tiles[r0 // TILE] & pure_visual_tiles
            allowed = allowed | full_vv.repeat_interleave(TILE)[:Skv]
        query_tile = r0 // TILE
        mask = mask_chunk(attention_mask, r0, r1, Sq, Skv)
        bypass = tv_layer.bypass_keys(query_tile) if tv_layer is not None else []
        if bypass:
            omitted = set(bypass)
            columns = torch.cat([kpos[index * TILE:(index + 1) * TILE]
                                 for index in range(query_tile + 1) if index not in omitted])
            retained = key32.index_select(-2, columns)
            computed = torch.matmul(q32, retained.transpose(-1, -2)) * scaling
            s = computed.new_full((B, Hq, r1 - r0, Skv), NEG_INF)
            s.index_copy_(-1, columns, computed)
            s_t = score_tiles(s, allowed, mask)
            if mask is not None and not s_t.isfinite().flatten(-2).any(-1).all():
                # An extra mask can hide the protected key. Revert this query tile.
                bypass = []
                tv_layer.mask_fallback.add(query_tile)
            else:
                tv_layer.bypassed[query_tile] = len(bypass)
        if not bypass:
            s = torch.matmul(q32, key32.transpose(-1, -2)) * scaling      # [B, H, r, Skv] fp32
            s_t = score_tiles(s, allowed, mask)
        margin = tile_margins(s_t)
        skip = margin < ln_lam                                              # all rows far below running max
        if tv_layer is not None:
            tv_layer.observe_text(query_tile, s_t, ln_lam)
            if bypass:
                skip[..., bypass] = True
            if (policy.threshold is not None and tv_layer.previous is not None and query_tile in policy.visual_set
                    and query_tile not in tv_layer.dropped):
                # An accepted visual query keeps all causally visible pure V→V tiles.
                skip[..., [index for index in policy.pure_visual if index <= query_tile]] = False
        if STATE["record"] and phase == "decode":
            STATE["decode_records"].setdefault(layer_idx, []).append(
                margin.detach().float().cpu().unsqueeze(2))
        observer = STATE["observer"]
        if observer is not None:
            observer(layer=layer_idx, phase=phase, q_start=r0 + offset,
                     scores=s_t.flatten(-2)[..., :Skv], margin=margin, skip=skip,
                     ln_lambda=ln_lam, key_length=Skv)

        # bookkeeping (causal tiles only; fully-masked tiles are not "skipped", they do not exist)
        if STATE["collect_stats"] or rec is not None:
            k_first = torch.arange(n_kt, device=query.device) * TILE
            causal = (k_first <= (r1 - 1 + offset))                         # [n_kt]
            if vv_prefix is not None:
                # The historical stats field includes the newly reachable tiles.
                causal = causal | full_vv
            if STATE["collect_stats"]:
                sk = (skip & causal[None, None, :]).sum(dim=(0, 2)).cpu()   # [H]
                stats["skipped"] += sk
                stats["causal"] += int(causal.sum()) * B
                phase_stats = STATE["phase_stats"].setdefault(
                    phase, {"skipped": 0, "causal": 0})
                phase_stats["skipped"] += int(sk.sum())
                phase_stats["causal"] += int(causal.sum()) * B * Hq
            if rec is not None:
                rec.append(margin.detach().float().cpu())

        s_t = s_t.masked_fill(skip[:, :, None, :, None], NEG_INF)
        p = torch.softmax(s_t.view(B, Hq, r1 - r0, n_kt * TILE)[..., :Skv], dim=-1)
        out[:, :, r0:r1] = torch.matmul(p.to(value.dtype), value)

    if tv_layer is not None:
        tv_layer.finish()
    return out.transpose(1, 2).contiguous(), None


# ----------------------------------------------------------------------------- registration
def register() -> str:
    """Register under the name 'blasst'. Safe to call more than once."""
    from transformers import AttentionInterface
    try:
        AttentionInterface.register("blasst", blasst_attention_forward)
    except Exception:
        # already registered (or an older API); overwrite the mapping directly
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["blasst"] = blasst_attention_forward
    return "blasst"


def use_blasst(model) -> None:
    """Switch an already-loaded model to the blasst implementation."""
    register()
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("blasst")
    else:
        model.config._attn_implementation = "blasst"
        for sub in model.modules():
            cfg = getattr(sub, "config", None)
            if cfg is not None and hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "blasst"


register()
