"""
Region-aware BLASST attention for the VLM study (VLM_REGIONS_2.md), registered as
attn_implementation "blasst_regions". Prefill only, batch size 1, no KV cache (offset 0).

The tiles, the skip rule and the margins are exactly those of blasst_attn.py (128 x 128 tiles, running
row max over key tiles left to right, skip a tile iff every row is below its running max by more than
ln(lambda), i.e. margin < ln(lambda)). Two things are new:

1. A lambda per REGION. Every 128-token block of the prompt has a kind: 0 = text, 1 = mixed, 2 = video
   (set with set_kinds). A tile's region is (query block kind, key block kind), and its threshold is
   ln_lam_table[query kind, key kind], a 3 x 3 table (set_table). -inf = never skip that region.
   A table with the same value everywhere is plain BLASST with one lambda.

2. A PROBE mode (set_probe): attention stays dense (nothing is skipped), and for each lambda in a list
   we measure, for every query row, how much of its softmax weight sits on the tiles that lambda WOULD
   skip, split into weight on video keys and on all other keys. We also keep each row's dense weight on
   video keys. Results per layer: STATE["probe"][layer] = {"dense_vid": [H, S], "lost_vid": [n_lams, H, S],
   "lost_other": [n_lams, H, S]} (fp16, CPU).

Margins can be recorded as in blasst_attn (STATE["record"]); per-region skip counts are kept in
STATE["stats"][layer] = {"skipped": [3, 3], "causal": [3, 3]} (summed over heads; rows = query kind).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

TILE = 128
NEG_INF = float("-inf")
TEXT, MIXED, VIDEO = 0, 1, 2

STATE: Dict = {
    "kinds": None,            # [n_blocks] long on the GPU; None = every block is text
    "q_kinds": None,          # optional override of the query-side kinds (default: kinds)
    "k_kinds": None,          # optional override of the key-side kinds (default: kinds)
    "is_vision": None,        # [S] bool on the GPU (token level); needed by the probe
    "ln_lam_table": None,     # [3, 3] float on the GPU; None = no skipping
    "probe_ln_lams": [],      # ln(lambda) values to probe; empty = no probe
    "record": False,
    "records": {},            # layer -> list of [H, n_kt] chunk margins (fp32, CPU)
    "probe": {},
    "stats": {},
}


# ----------------------------------------------------------------------------- control
def block_kinds(is_vision: torch.Tensor) -> torch.Tensor:
    """0 = text, 1 = mixed, 2 = video for each 128-token block (the last block may be shorter)."""
    n = math.ceil(is_vision.numel() / TILE)
    pad = n * TILE - is_vision.numel()
    v = F.pad(is_vision.float(), (0, pad), value=float("nan")).view(n, TILE)
    frac = torch.nanmean(v, dim=1)
    return torch.where(frac == 0, TEXT, torch.where(frac == 1, VIDEO, MIXED)).long()


def set_kinds(is_vision: Optional[torch.Tensor], q_kinds: Optional[torch.Tensor] = None,
              k_kinds: Optional[torch.Tensor] = None) -> None:
    """Tell the function which positions are video tokens (bool [S]) before each forward pass.
    q_kinds / k_kinds (optional, [n_blocks]) relabel the query side or key side, e.g. to call a mixed block
    that holds the question "text" on the query side."""
    STATE["is_vision"] = is_vision
    STATE["kinds"] = None if is_vision is None else block_kinds(is_vision)
    STATE["q_kinds"], STATE["k_kinds"] = q_kinds, k_kinds


def set_table(lams: Optional[Dict[tuple, float]] = None, default: Optional[float] = None) -> None:
    """lams: {(query kind, key kind): lambda}, kinds from TEXT / MIXED / VIDEO; missing entries use
    `default`. lambda None or 0 = never skip that region. Both None = no skipping at all."""
    if lams is None and default is None:
        STATE["ln_lam_table"] = None
        return
    t = torch.full((3, 3), NEG_INF if not default else math.log(default))
    for (qk, kk), lam in (lams or {}).items():
        t[qk, kk] = NEG_INF if not lam else math.log(lam)
    STATE["ln_lam_table"] = t.clamp(max=0.0)            # lambda above 1 would empty a row (see blasst_attn)


def set_probe(lams: Sequence[float] = ()) -> None:
    STATE["probe_ln_lams"] = [math.log(l) for l in lams]


def reset(record: bool = False) -> None:
    STATE.update(record=record, records={}, probe={}, stats={})


def get_margins() -> torch.Tensor:
    """[layers, heads, q tiles, k tiles] fp32 CPU, non-causal tiles = -inf."""
    return torch.stack([torch.stack(STATE["records"][l], dim=1) for l in sorted(STATE["records"])])


def region_counts() -> Dict[str, torch.Tensor]:
    """Skipped and causal tile counts [3, 3] summed over all layers and heads."""
    sk = sum(s["skipped"] for s in STATE["stats"].values())
    ca = sum(s["causal"] for s in STATE["stats"].values())
    return {"skipped": sk, "causal": ca}


# ----------------------------------------------------------------------------- the function
def blasst_regions_forward(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    B, Hq, Sq, D = query.shape
    Hkv, Skv = key.shape[1], key.shape[2]
    if scaling is None:
        scaling = 1.0 / math.sqrt(D)
    n_rep = Hq // Hkv
    if n_rep > 1:                                   # GQA: same ordering as transformers' repeat_kv
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)
    if Sq == 1:                                     # decode is not studied here: standard attention
        out = F.scaled_dot_product_attention(query, key, value, scale=scaling)
        return out.transpose(1, 2).contiguous(), None
    assert B == 1 and Sq == Skv, "prefill of one unpadded prompt without a KV cache"

    dev = query.device
    layer = getattr(module, "layer_idx", -1)
    n_kt = math.ceil(Skv / TILE)
    pad = n_kt * TILE - Skv
    kinds = STATE["kinds"] if STATE["kinds"] is not None else torch.zeros(n_kt, dtype=torch.long, device=dev)
    q_kinds = (STATE["q_kinds"] if STATE["q_kinds"] is not None else kinds).to(dev)
    kinds = (STATE["k_kinds"] if STATE["k_kinds"] is not None else kinds).to(dev)     # key side
    table = STATE["ln_lam_table"]
    table = table.to(dev) if table is not None else None
    probe = STATE["probe_ln_lams"]
    if probe:
        assert table is None, "probe mode measures a dense pass; switch skipping off"
        n_l = len(probe)
        ln_probe = torch.tensor(probe, device=dev)
        vis_k = (STATE["is_vision"].to(dev).float() if STATE["is_vision"] is not None
                 else torch.zeros(Skv, device=dev))
        res = {"dense_vid": torch.zeros(Hq, Sq, device=dev),
               "lost_vid": torch.zeros(n_l, Hq, Sq, device=dev), "lost_other": torch.zeros(n_l, Hq, Sq, device=dev)}
    stats = STATE["stats"].setdefault(layer, {"skipped": torch.zeros(3, 3, dtype=torch.long),
                                              "causal": torch.zeros(3, 3, dtype=torch.long)})
    rec = STATE["records"].setdefault(layer, []) if STATE["record"] else None

    key32 = key[0].to(torch.float32)                # [H, Skv, D]
    kpos = torch.arange(Skv, device=dev)
    k_first = torch.arange(n_kt, device=dev) * TILE
    out = torch.empty_like(query)
    for r0 in range(0, Sq, TILE):
        r1 = min(r0 + TILE, Sq)
        s = torch.matmul(query[0, :, r0:r1].to(torch.float32), key32.transpose(-1, -2)) * scaling    # [H, r, Skv]
        qpos = torch.arange(r0, r1, device=dev)
        s = s.masked_fill(kpos[None, :] > qpos[:, None], NEG_INF)                                  # causal
        if pad:
            s = F.pad(s, (0, pad), value=NEG_INF)
        s_t = s.view(Hq, r1 - r0, n_kt, TILE)
        tile_max = s_t.amax(dim=-1)                                                                # [H, r, n_kt]
        running = torch.cummax(tile_max, dim=-1).values
        diff = torch.where(torch.isfinite(running), tile_max - running, torch.full_like(tile_max, NEG_INF))
        margin = diff.amax(dim=1)                                                                  # [H, n_kt]
        causal = k_first <= (r1 - 1)                                                               # [n_kt]
        q_kind = q_kinds[r0 // TILE]
        if table is not None:
            skip = margin < table[q_kind, kinds][None, :]                                          # per-tile lambda
        else:
            skip = torch.zeros_like(margin, dtype=torch.bool)
        # bookkeeping: skipped and causal tiles per key kind for this query kind (summed over heads)
        sk_by_kind = torch.zeros(3, device=dev).index_add_(0, kinds, (skip & causal).sum(0).float())
        ca_by_kind = torch.zeros(3, device=dev).index_add_(0, kinds, causal.float()) * Hq
        stats["skipped"][int(q_kind)] += sk_by_kind.long().cpu()
        stats["causal"][int(q_kind)] += ca_by_kind.long().cpu()
        if rec is not None:
            rec.append(margin.detach().float().cpu())

        s_t = s_t.masked_fill(skip[:, None, :, None], NEG_INF)
        p = torch.softmax(s_t.view(Hq, r1 - r0, n_kt * TILE)[..., :Skv], dim=-1)                    # [H, r, Skv]
        out[0, :, r0:r1] = torch.matmul(p.to(value.dtype), value[0])
        if probe:
            pv = p * vis_k                                                                         # weight on video keys
            w = F.pad(p, (0, pad)).view(Hq, r1 - r0, n_kt, TILE).sum(-1)                          # [H, r, n_kt]
            wv = F.pad(pv, (0, pad)).view(Hq, r1 - r0, n_kt, TILE).sum(-1)
            would_skip = (margin[None] < ln_probe[:, None, None]).float()                         # [n_l, H, n_kt]
            res["dense_vid"][:, r0:r1] = wv.sum(-1)
            res["lost_vid"][:, :, r0:r1] = torch.einsum("hrk,lhk->lhr", wv, would_skip)
            res["lost_other"][:, :, r0:r1] = torch.einsum("hrk,lhk->lhr", w - wv, would_skip)
    if probe:
        STATE["probe"][layer] = {k: v.half().cpu() for k, v in res.items()}
    return out.transpose(1, 2).contiguous(), None


def register() -> str:
    from transformers import AttentionInterface
    AttentionInterface.register("blasst_regions", blasst_regions_forward)
    return "blasst_regions"


register()
