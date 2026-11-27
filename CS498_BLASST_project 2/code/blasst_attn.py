# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
BLASST skip-softmax attention, simulated in PyTorch and registered with
Hugging Face transformers' AttentionInterface under the name "blasst".

What it does (prefill, i.e. more than one query token):
  * take one tile of query rows at a time (128 by default),
  * compute their scaled scores against ALL keys, apply the causal mask,
  * split the key axis into tiles of the same size and take each tile's row maxima,
  * walk the tiles left to right keeping a running row max (cummax, which
    includes the current tile, exactly as BLASST's online softmax does),
  * skip a tile only if EVERY row in the chunk is below its running max by
    more than ln(lambda):   max_rows(tile_max - running_max) < ln(lambda),
  * set skipped tiles to -inf, do a normal softmax, multiply by V.
  Because a tile that holds some row's running max has (tile_max - running_max) = 0
  for that row, it can never be skipped, so every row keeps at least one tile.

Decode uses 1 x 128 tiles over the cached keys, with the same skip rule.
set_decode_threshold(enabled=False) restores the original dense decode path.
With decode_region_max configured, each contiguous key run supplies its own
running max for the skip test. The retained tiles share one softmax.

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
        margin[h, q_tile, k_tile] = max over the query rows of (tile_max - running_max)
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

from region_max import row_tile_margins

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
    "prefill_region_max": None,
    "decode_region_max": None,  # optional prompt-local key-run layout
    "phase_stats": {},
    "observer": None,         # optional synchronous callback; must not retain GPU tensors
    "qkv_hook": None,         # optional post-RoPE callback returning an output or None
    "vv_prefix_visual": None, # opt-in prompt mask for bidirectional V→V tiles
    "tv_guided_vv": None,    # prompt-local previous-layer text vote policy
    "decode_visual_links": None,  # prompt-local prefill text-to-vision links
    "binary_tv_pipeline": None,  # optional cross-layer T→V masks and V→V row gate
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


def score_tiles(scores, allowed, mask=None, *, tile_size=None):
    """Apply the attention masks and partition scores along the key axis."""
    tile_size = TILE if tile_size is None else tile_size
    scores = scores.masked_fill(~allowed, NEG_INF)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, NEG_INF)
        else:
            scores = scores + mask.to(torch.float32)
    pad = -scores.shape[-1] % tile_size
    if pad:
        scores = F.pad(scores, (0, pad), value=NEG_INF)
    return scores.unflatten(-1, (-1, tile_size))


def tile_margins(tiles):
    """Reduce each tile against its own row-wise running maximum."""
    return row_tile_margins(tiles.amax(dim=-1)).amax(dim=-2)


def mask_chunk(mask, start, end, query_len, key_len):
    if mask is None:
        return None
    mask = mask[..., -query_len:, :key_len]
    return mask if mask.shape[-2] == 1 else mask[..., start:end, :]


def token_gated_scores(query, key, dropped, visual_keys, scaling, *, skip_whole_query=False):
    """Compute QK only on the query and key subsets admitted by the gate."""
    scores = query.new_full((*query.shape[:-1], key.shape[-2]), NEG_INF)
    kept_rows = (~dropped).nonzero().flatten()
    if kept_rows.numel():
        scores[..., kept_rows, :] = query[..., kept_rows, :] @ key.transpose(-1, -2) * scaling
    if not skip_whole_query:
        dropped_rows = dropped.nonzero().flatten()
        text_columns = (~visual_keys).nonzero().flatten()
        scores[..., dropped_rows[:, None], text_columns] = (
            query[..., dropped_rows, :] @ key[..., text_columns, :].transpose(-1, -2) * scaling
        )
    return scores, kept_rows


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
    binary = STATE["binary_tv_pipeline"]
    if binary is not None and phase == "decode" and not binary.guide_decode:
        binary = None
    if binary is not None:
        if B != 1 or layer_idx < 0 or (phase == "prefill" and (Sq != Skv or Sq != len(binary.visual))):
            raise ValueError("The binary T→V pipeline requires one unpadded prompt, indexed layers, and a full prefill")
        if layer_idx and (phase == "decode" or binary.guide_prefill):
            mask = mask_chunk(attention_mask, 0, Sq, Sq, Skv)
            output, kept = binary.forward(layer_idx, query, key, value, mask, scaling, ln_lam, phase)
            if STATE["collect_stats"]:
                eligible = causal_tiles(Sq, Skv, offset).to(query.device).unsqueeze(0)
                phase_stats = STATE["phase_stats"].setdefault(phase, {"skipped": 0, "causal": 0})
                phase_stats["skipped"] += int((eligible & ~kept).sum())
                phase_stats["causal"] += int(eligible.sum()) * Hq
            return output, None
        binary.start(layer_idx, Hq, Skv, query.device)
    links = STATE["decode_visual_links"]
    if links is not None:
        if B != 1 or layer_idx < 0:
            raise ValueError("Decode visual links require one prompt and indexed layers")
        if phase == "decode":
            if layer_idx not in links.layers or Skv < len(links.visual):
                raise ValueError("Decode visual links require a prefilled, position-aligned KV cache")
            mask = mask_chunk(attention_mask, 0, 1, Sq, Skv)
            output, skip = links.decode(layer_idx, query, key, value, mask, scaling, ln_lam, score_tiles)
            if STATE["collect_stats"]:
                phase_stats = STATE["phase_stats"].setdefault("decode", {"skipped": 0, "causal": 0})
                phase_stats["skipped"] += int(skip.sum())
                phase_stats["causal"] += skip.numel()
            return output.transpose(1, 2).contiguous(), None
        if Sq != Skv or Skv != len(links.visual):
            raise ValueError("Decode visual links require one uncached prefill")
        links.start_layer(layer_idx, Hq, query.device)
    key32 = key.to(torch.float32)
    kpos = torch.arange(Skv, device=query.device)
    out = torch.empty_like(query)
    vv_prefix = STATE["vv_prefix_visual"] if phase == "prefill" else None
    policy = STATE["tv_guided_vv"] if phase == "prefill" else None
    tile_size = policy.tile_size if policy is not None else TILE
    n_kt = math.ceil(Skv / tile_size)
    regions = STATE["prefill_region_max"] if phase == "prefill" else STATE["decode_region_max"]
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
        pure_visual_tiles = F.pad(visual, (0, -Skv % tile_size), value=True).reshape(-1, tile_size).all(-1)

    stats = STATE["stats"].setdefault(layer_idx, {
        "skipped": torch.zeros(Hq, dtype=torch.long), "causal": torch.zeros(Hq, dtype=torch.long)})
    rec = (STATE["records"].setdefault(layer_idx, [])
           if STATE["record"] and phase == "prefill" else None)
    if STATE["record"] and phase == "prefill":
        STATE["record_meta"][layer_idx] = {"Sq": Sq, "Skv": Skv, "offset": offset}

    for r0 in range(0, Sq, tile_size):
        r1 = min(r0 + tile_size, Sq)
        q32 = query[:, :, r0:r1].to(torch.float32)
        qpos = torch.arange(r0, r1, device=query.device) + offset
        allowed = kpos[None, :] <= qpos[:, None]                            # [r, Skv] causal
        if vv_prefix is not None:
            full_vv = pure_visual_tiles[r0 // tile_size] & pure_visual_tiles
            allowed = allowed | full_vv.repeat_interleave(tile_size)[:Skv]
        query_tile = r0 // tile_size
        mask = mask_chunk(attention_mask, r0, r1, Sq, Skv)
        token_drops = tv_layer.token_drops[r0:r1] if tv_layer is not None and tv_layer.token_drops is not None else None
        token_bypass = token_drops is not None and bool(token_drops.any())
        skip_whole_query = token_bypass and policy.token_gate == "whole"
        bypass = tv_layer.bypass_keys(query_tile) if tv_layer is not None else []
        if bypass:
            omitted = set(bypass)
            columns = torch.cat([kpos[index * tile_size:(index + 1) * tile_size]
                                 for index in range(query_tile + 1) if index not in omitted])
            retained = key32.index_select(-2, columns)
            computed = torch.matmul(q32, retained.transpose(-1, -2)) * scaling
            s = computed.new_full((B, Hq, r1 - r0, Skv), NEG_INF)
            s.index_copy_(-1, columns, computed)
            s_t = score_tiles(s, allowed, mask, tile_size=tile_size)
            if mask is not None and not s_t.isfinite().flatten(-2).any(-1).all():
                # An extra mask can hide the protected key. Revert this query tile.
                bypass = []
                tv_layer.mask_fallback.add(query_tile)
            else:
                tv_layer.bypassed[query_tile] = len(bypass)
        if token_bypass:
            s, kept_rows = token_gated_scores(q32, key32, token_drops.to(query.device),
                                             policy.visual.to(query.device), scaling, skip_whole_query=skip_whole_query)
            if mask is not None and mask.dtype != torch.bool:
                s = s.masked_fill(mask <= torch.finfo(mask.dtype).min, NEG_INF)
            s_t = score_tiles(s, allowed, mask, tile_size=tile_size)
        elif not bypass:
            s = torch.matmul(q32, key32.transpose(-1, -2)) * scaling      # [B, H, r, Skv] fp32
            s_t = score_tiles(s, allowed, mask, tile_size=tile_size)
        row_margins = (regions.row_margins(s_t, Skv) if regions is not None
                       else row_tile_margins(s_t.amax(-1)))
        if links is not None:
            links.record(layer_idx, r0, s_t, ln_lam, mask)
        if binary is not None:
            binary.record_standard_links(layer_idx, r0 + offset, s_t, mask, ln_lam, phase)
        margin = row_margins.amax(-2)
        skip = margin < ln_lam                                              # all rows far below running max
        fully_bypassed = bypass
        if token_bypass and bool(token_drops.all()):
            fully_bypassed = (list(range(query_tile + 1)) if skip_whole_query
                              else [tile for tile in policy.pure_visual if tile <= query_tile])
            skip[..., fully_bypassed] = True
        if tv_layer is not None:
            if token_drops is None:
                tv_layer.observe_text(query_tile, row_margins, ln_lam)
            elif policy.replay is None or policy.token_gate is None:
                tv_layer.observe_tokens(r0, s_t, ln_lam, mask)
            if bypass:
                skip[..., bypass] = True
            if (policy.threshold is not None and policy.top_percent is None and tv_layer.previous is not None
                    and query_tile in policy.visual_set
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
            k_first = torch.arange(n_kt, device=query.device) * tile_size
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
                if fully_bypassed:
                    phase_stats["qk_bypassed"] = phase_stats.get("qk_bypassed", 0) + len(fully_bypassed) * B * Hq
            if rec is not None:
                rec.append(margin.detach().float().cpu())

        s_t = s_t.masked_fill(skip[:, :, None, :, None], NEG_INF)
        scores = s_t.flatten(-2)[..., :Skv]
        if skip_whole_query:
            out[:, :, r0:r1] = 0
            if kept_rows.numel():
                p = torch.softmax(scores[:, :, kept_rows], dim=-1).nan_to_num()
                out[:, :, r0 + kept_rows] = torch.matmul(p.to(value.dtype), value)
            continue
        p = torch.softmax(scores, dim=-1)
        if token_bypass:
            p = p.nan_to_num()
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
