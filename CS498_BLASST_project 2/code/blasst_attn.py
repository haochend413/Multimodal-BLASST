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

Decode (one query token): plain scaled_dot_product_attention, unchanged.

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
    if record is not None:
        STATE["record"] = record
    if dump_qk_layers is not None:
        STATE["dump_qk_layers"] = list(dump_qk_layers)


def get_margins() -> Dict[int, torch.Tensor]:
    """layer_idx -> margin tensor [B, H, n_q_tiles, n_k_tiles] (fp32, CPU)."""
    return {l: torch.stack(chunks, dim=2) for l, chunks in STATE["records"].items()}


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

    n_rep = Hq // Hkv
    if n_rep > 1:                      # GQA: same ordering as transformers' repeat_kv
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    # ---------------- decode: one query token, standard SDPA ----------------
    if Sq == 1:
        mask = attention_mask
        if mask is not None and mask.dtype != torch.bool:
            mask = mask[:, :, -1:, :Skv].to(query.dtype)
        elif mask is not None:
            mask = mask[:, :, -1:, :Skv]
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    # ---------------- prefill: chunked skip-softmax ----------------
    ln_lam = math.log(STATE["a"] / Skv) if STATE["a"] is not None else STATE["ln_lam"]
    # Cap lambda at 1 (ln_lam <= 0). Above 1 even the tile holding a row's max (margin 0) would be
    # skipped, the row would lose every tile and softmax would return NaN. Calibrated a/L hits this
    # when L is short (e.g. a = 1000, L = 650).
    ln_lam = min(ln_lam, 0.0)
    offset = Skv - Sq                 # queries are the LAST Sq positions of the key axis
    n_kt = math.ceil(Skv / TILE)
    pad = n_kt * TILE - Skv
    key32 = key.to(torch.float32)
    kpos = torch.arange(Skv, device=query.device)
    out = torch.empty_like(query)

    stats = STATE["stats"].setdefault(layer_idx, {
        "skipped": torch.zeros(Hq, dtype=torch.long), "causal": torch.zeros(Hq, dtype=torch.long)})
    rec = STATE["records"].setdefault(layer_idx, []) if STATE["record"] else None
    if STATE["record"]:
        STATE["record_meta"][layer_idx] = {"Sq": Sq, "Skv": Skv, "offset": offset}

    for r0 in range(0, Sq, TILE):
        r1 = min(r0 + TILE, Sq)
        q32 = query[:, :, r0:r1].to(torch.float32)
        s = torch.matmul(q32, key32.transpose(-1, -2)) * scaling          # [B, H, r, Skv] fp32
        qpos = torch.arange(r0, r1, device=query.device) + offset
        allowed = kpos[None, :] <= qpos[:, None]                            # [r, Skv] causal
        s = s.masked_fill(~allowed, NEG_INF)
        if attention_mask is not None:                                      # honour an explicit mask too
            m = attention_mask[:, :, r0:r1, :Skv] if attention_mask.shape[-2] == Sq else attention_mask[:, :, -Sq:, :Skv][:, :, r0:r1]
            if m.dtype == torch.bool:
                s = s.masked_fill(~m, NEG_INF)
            else:
                s = s + m.to(torch.float32)
        if pad:
            s = F.pad(s, (0, pad), value=NEG_INF)
        s_t = s.view(B, Hq, r1 - r0, n_kt, TILE)
        tile_max = s_t.amax(dim=-1)                                         # [B, H, r, n_kt]
        running = torch.cummax(tile_max, dim=-1).values                     # includes current tile
        diff = tile_max - running                                           # <= 0, -inf past causal edge
        diff = torch.where(torch.isfinite(running), diff, torch.full_like(diff, NEG_INF))
        margin = diff.amax(dim=2)                                           # max over rows -> [B, H, n_kt]
        skip = margin < ln_lam                                              # all rows far below running max

        # bookkeeping (causal tiles only; fully-masked tiles are not "skipped", they do not exist)
        if STATE["collect_stats"] or rec is not None:
            k_first = torch.arange(n_kt, device=query.device) * TILE
            causal = (k_first <= (r1 - 1 + offset))                         # [n_kt]
            if STATE["collect_stats"]:
                sk = (skip & causal[None, None, :]).sum(dim=(0, 2)).cpu()   # [H]
                stats["skipped"] += sk
                stats["causal"] += int(causal.sum()) * B
            if rec is not None:
                rec.append(margin.detach().float().cpu())

        s_t = s_t.masked_fill(skip[:, :, None, :, None], NEG_INF)
        p = torch.softmax(s_t.view(B, Hq, r1 - r0, n_kt * TILE)[..., :Skv], dim=-1)
        out[:, :, r0:r1] = torch.matmul(p.to(value.dtype), value)

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
