"""Model-agnostic capture of post-RoPE Q/K tensors.

Every metric in this package needs the queries and keys *as the attention
kernel sees them* -- after projection, head splitting, and any positional
rotation.  Hooking ``q_proj``/``k_proj`` would give pre-RoPE tensors and a
different, wrong answer for anything position-dependent.

So instead of per-architecture hooks we intercept the one function nearly every
modern implementation funnels into: ``torch.nn.functional.scaled_dot_product_attention``.
HuggingFace (``attn_implementation="sdpa"``), diffusers attention processors,
timm and most hand-rolled blocks all call it with ``q``/``k`` already in
``[B, H, N, D]`` form.

Two wrinkles this handles:

* Modules that did ``from torch.nn.functional import scaled_dot_product_attention``
  at import time hold their own reference, so patching the attribute on
  ``torch.nn.functional`` alone would miss them.  We sweep ``sys.modules`` and
  rebind every alias that points at the original.
* Grouped-query attention may arrive either already expanded (HF's ``repeat_kv``)
  or via SDPA's ``enable_gqa``.  We expand the second case so that head ``h`` of
  ``q`` always lines up with head ``h`` of ``k``.

A model that reaches none of these paths (anything still doing explicit
``torch.matmul`` attention) captures nothing; the runner turns that into a clear
error rather than an empty result file.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


_ENABLED = True


def set_capture_enabled(flag: bool) -> None:
    """Gate recording globally.

    A diffusion pipeline calls attention once per layer *per denoising step*,
    so capturing the whole run mixes timesteps with depths -- and layer
    selection by call index would then pick calls from different steps. The
    adapter toggles this from a step callback to capture exactly one timestep.
    """
    global _ENABLED
    _ENABLED = flag


@dataclass
class AttentionCall:
    """One captured attention call."""

    index: int  # call order within the forward pass == depth
    query: torch.Tensor  # [H, N, D]
    key: torch.Tensor  # [H, S, D]
    value: Optional[torch.Tensor]  # [H, S, D]; needed for output-error metrics
    heads: Sequence[int]  # which head indices these are
    is_causal: bool
    had_mask: bool
    scale: Optional[float]

    @property
    def shape_key(self) -> str:
        h, n, d = self.query.shape
        return f"H{h}_N{n}_D{d}"


@dataclass
class CaptureConfig:
    """What to keep.  Defaults are tuned to keep memory small."""

    layers: Optional[Sequence[int]] = None  # call indices; None = count only
    min_seq_len: int = 256  # ignore tiny attentions (e.g. cross-attn to text)
    max_heads: Optional[int] = 8  # how many heads to keep per layer
    store_dtype: torch.dtype = torch.float16
    store_device: str = "cpu"
    store_value: bool = True  # V is required to measure output error, not just mass
    head_seed: int = 0
    """Heads are sampled at random, not taken as 0..N-1.

    Attention heads are heterogeneous -- induction, positional and sink heads
    behave nothing alike -- so the first four are not a sample of the rest.
    Taking them was a silent bias in every per-model number.
    """


class SDPACapture:
    """Context manager that records Q/K from scaled-dot-product attention."""

    def __init__(self, config: Optional[CaptureConfig] = None) -> None:
        self.config = config or CaptureConfig()
        self.calls: List[AttentionCall] = []
        self.seen: List[Dict[str, object]] = []  # metadata for every call
        self._counter = 0
        self._patched: List[tuple] = []
        self._original = None

    # -- patching -------------------------------------------------------

    def __enter__(self) -> "SDPACapture":
        self._original = F.scaled_dot_product_attention
        original = self._original
        capture = self

        def wrapper(query, key, value, *args, **kwargs):
            if _ENABLED:
                capture._record(query, key, value, kwargs, args)
            return original(query, key, value, *args, **kwargs)

        F.scaled_dot_product_attention = wrapper
        self._patched.append((F, "scaled_dot_product_attention"))

        # Rebind aliases held by already-imported modules.
        for module in list(sys.modules.values()):
            if module is None or module is F:
                continue
            try:
                attr = getattr(module, "scaled_dot_product_attention", None)
            except Exception:  # modules with exotic __getattr__
                continue
            if attr is original:
                setattr(module, "scaled_dot_product_attention", wrapper)
                self._patched.append((module, "scaled_dot_product_attention"))
        return self

    def __exit__(self, *exc) -> None:
        for holder, name in self._patched:
            setattr(holder, name, self._original)
        self._patched.clear()

    # -- recording ------------------------------------------------------

    def _record(self, query, key, value, kwargs, args) -> None:
        idx = self._counter
        self._counter += 1

        if not (torch.is_tensor(query) and torch.is_tensor(key) and query.dim() == 4):
            return

        b, h, n, d = query.shape
        kh, s = key.shape[1], key.shape[2]
        is_causal = bool(kwargs.get("is_causal", False))
        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is None and args:
            attn_mask = args[0]
        scale = kwargs.get("scale", None)

        self.seen.append(
            {"index": idx, "heads": h, "n_query": n, "n_key": s, "head_dim": d,
             "is_causal": is_causal, "kv_heads": kh}
        )

        cfg = self.config
        if cfg.layers is None or idx not in cfg.layers:
            return
        if n < cfg.min_seq_len or s < cfg.min_seq_len:
            return

        # Grouped-query attention that SDPA would have expanded internally.
        if kh != h:
            if h % kh != 0:
                return
            key = key.repeat_interleave(h // kh, dim=1)

        keep = h if cfg.max_heads is None else min(h, cfg.max_heads)
        generator = torch.Generator().manual_seed(cfg.head_seed + idx)
        head_ids = torch.randperm(h, generator=generator)[:keep].sort().values
        q = query[0, head_ids].detach().to(cfg.store_dtype).to(cfg.store_device)
        k = key[0, head_ids].detach().to(cfg.store_dtype).to(cfg.store_device)
        v = None
        if cfg.store_value and torch.is_tensor(value) and value.dim() == 4:
            val = value
            if val.shape[1] != h and h % val.shape[1] == 0:
                val = val.repeat_interleave(h // val.shape[1], dim=1)
            if val.shape[1] == h:
                v = val[0, head_ids].detach().to(cfg.store_dtype).to(cfg.store_device)
        self.calls.append(
            AttentionCall(
                index=idx,
                query=q,
                key=k,
                value=v,
                heads=[int(i) for i in head_ids],
                is_causal=is_causal,
                had_mask=attn_mask is not None,
                scale=float(scale) if scale is not None else None,
            )
        )

    # -- results --------------------------------------------------------

    @property
    def n_calls(self) -> int:
        return self._counter

    def self_attention_calls(self) -> List[Dict[str, object]]:
        """Metadata for calls that look like self-attention over a long sequence.

        Cross-attention (to a short text conditioning sequence, say) has
        ``n_key != n_query``; those tell us nothing about vertical vectors
        between neighbouring queries of the same modality, so the runner
        selects layers from this filtered list.
        """
        return [
            c
            for c in self.seen
            if c["n_query"] == c["n_key"] and c["n_query"] >= self.config.min_seq_len
        ]


def summarise_shapes(seen: Sequence[Dict[str, object]], top: int = 6) -> str:
    """Human-readable census of the attention shapes a forward pass produced.

    When nothing qualifies for analysis, "2076 calls seen" is useless on its
    own: it cannot distinguish "the model never reached SDPA" from "it did, but
    every call was a small windowed one" from "the long call is cross-attention".
    This prints the distinct shapes, largest first, with counts.
    """
    if not seen:
        return "no SDPA calls at all"
    counts: Dict[tuple, int] = {}
    for call in seen:
        key = (int(call["n_query"]), int(call["n_key"]), int(call["heads"]))
        counts[key] = counts.get(key, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -max(kv[0][0], kv[0][1]))
    parts = [
        f"{n}x [q={q}, k={k}, heads={h}]{' self' if q == k else ' cross'}"
        for (q, k, h), n in ordered[:top]
    ]
    extra = f" (+{len(ordered) - top} more shapes)" if len(ordered) > top else ""
    return "; ".join(parts) + extra


def pick_layers(calls: Sequence[Dict[str, object]], count: int) -> List[int]:
    """Evenly spaced layer indices, always including the first and last.

    Attention structure is famously depth-dependent -- early layers look local
    and positional, later ones semantic -- so sampling a spread of depths is the
    difference between a finding and an artefact of one layer.
    """
    if not calls:
        return []
    indices = [int(c["index"]) for c in calls]
    if count >= len(indices):
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    step = (len(indices) - 1) / (count - 1)
    return sorted({indices[int(round(i * step))] for i in range(count)})
