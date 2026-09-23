"""Tests for the metric definitions.

These check the properties the conclusions rest on, using inputs whose answer is
known analytically.  Run with ``python -m pytest tests/`` or directly:
``python tests/test_metrics.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qk_sparse import metrics as M  # noqa: E402
from qk_sparse.capture import CaptureConfig, SDPACapture, pick_layers  # noqa: E402


def test_chance_jaccard_matches_simulation():
    """The analytic chance line must match random sets, or every "above chance"
    claim in the report is calibrated against the wrong baseline."""
    n, m, trials = 1000, 50, 400
    g = torch.Generator().manual_seed(0)
    total = 0.0
    for _ in range(trials):
        a = torch.zeros(n, dtype=torch.bool)
        b = torch.zeros(n, dtype=torch.bool)
        a[torch.randperm(n, generator=g)[:m]] = True
        b[torch.randperm(n, generator=g)[:m]] = True
        total += float(M.jaccard(a, b))
    empirical = total / trials
    analytic = M.chance_jaccard(m, n)
    assert abs(empirical - analytic) < 0.01, (empirical, analytic)


def test_patterns_are_ordered_by_freedom():
    """At one budget, a pattern with strictly more freedom cannot do worse.

    oracle >= vertical >= block, and oracle >= horizontal >= block, by
    construction: each is the previous one with a constraint removed.  If this
    ever fails, the budgets are not actually matched.
    """
    torch.manual_seed(0)
    attn = torch.rand(64, 512)
    attn = attn / attn.sum(-1, keepdim=True)
    for m in (8, 25, 64):
        res = M.pattern_masses(attn, m=m, key_block=8)
        assert res["oracle"] >= res["vertical"] - 1e-6
        assert res["oracle"] >= res["horizontal"] - 1e-6
        assert res["vertical"] >= res["block"] - 1e-6
        assert res["horizontal"] >= res["block"] - 1e-6


def test_identical_queries_make_vertical_optimal():
    """With identical queries in a block, sharing one key set costs nothing.

    This is the upper pole of the scale the verdict is read against.
    """
    torch.manual_seed(0)
    d, s, p = 32, 512, 64
    q = torch.randn(1, d).expand(p, d).contiguous()
    k = torch.randn(s, d)
    out = M.analyse_head(
        q, k, causal=False, query_block=p, key_block=8,
        densities=[0.05], deltas=[1], max_blocks=1,
    )
    block = out["blocks"][0]
    assert block["d0.05/vertical"] == block["d0.05/oracle"]
    # and the cheap pooled selector loses nothing either
    assert block["d0.05/sel_pooled"] == block["d0.05/vertical"]
    assert block["cos_to_mean"] > 0.999


def test_independent_queries_collapse_to_chance():
    """With independent queries, key sets must not agree above chance."""
    torch.manual_seed(0)
    d, s, p = 32, 2048, 128
    q = torch.randn(4 * p, d)
    k = torch.randn(s, d)
    out = M.analyse_head(
        q, k, causal=False, query_block=p, key_block=8,
        densities=[0.05], deltas=[1, 4], max_blocks=4,
    )
    probe = out["probes"][0]
    assert probe["d0.05/jaccard@1"] < 5 * probe["d0.05/chance_jaccard"] + 0.02
    # The budget must be recorded, since a tiny one makes overlap meaningless.
    assert probe["d0.05/budget_keys"] == round(0.05 * s)
    block = out["blocks"][0]
    # A shared key set should retain far less than per-row selection.
    assert block["d0.05/vertical"] < 0.6 * block["d0.05/oracle"]


def test_causal_blocks_use_only_past_keys():
    """Causal analysis must never score a block against its own or future keys."""
    torch.manual_seed(0)
    d, n, p = 16, 1024, 128
    q = torch.randn(n, d)
    k = torch.randn(n, d)
    out = M.analyse_head(
        q, k, causal=True, query_block=p, key_block=8,
        densities=[0.1], deltas=[1], max_blocks=8,
    )
    for block in out["blocks"]:
        assert block["n_allowed"] <= block["start"]


def test_short_sequences_are_rejected():
    """Too-short inputs return None rather than contributing biased numbers."""
    q = torch.randn(16, 8)
    k = torch.randn(16, 8)
    assert M.analyse_head(
        q, k, causal=False, query_block=128, key_block=64,
        densities=[0.1], deltas=[1], max_blocks=4,
    ) is None


def test_capture_sees_sdpa_and_expands_gqa():
    """Capture must record grouped-query attention with q/k heads aligned."""
    import torch.nn.functional as F

    q = torch.randn(1, 8, 512, 16)
    k = torch.randn(1, 2, 512, 16)
    v = torch.randn(1, 2, 512, 16)
    with SDPACapture(CaptureConfig(layers={0}, min_seq_len=128, max_heads=8)) as cap:
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call.query.shape[0] == call.key.shape[0] == 8


def test_capture_restores_the_original_function():
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention
    with SDPACapture(CaptureConfig(layers=set())):
        assert F.scaled_dot_product_attention is not original
    assert F.scaled_dot_product_attention is original


def test_pick_layers_spans_depth():
    calls = [{"index": i, "n_query": 512, "n_key": 512} for i in range(30)]
    picked = pick_layers(calls, 4)
    assert picked[0] == 0 and picked[-1] == 29 and len(picked) == 4


def test_wan_latent_grid_math():
    """Token grid must match Wan's 8x spatial / 4x temporal VAE plus 2x2 patching."""
    from qk_sparse.models import VideoDiTAdapter

    a = VideoDiTAdapter(height=480, width=832, num_frames=17)
    assert a._latent_grid() == (5, 30, 52)
    a = VideoDiTAdapter(height=256, width=256, num_frames=9)
    assert a._latent_grid() == (3, 16, 16)


def test_cross_attention_is_excluded():
    """Cross-attention to a short text sequence must not enter the analysis.

    Video DiTs interleave self-attention over latents with cross-attention to
    the prompt embedding. Only the former can answer whether neighbouring video
    queries share keys; the capture filters on n_query == n_key.
    """
    import torch.nn.functional as F

    cfg = CaptureConfig(layers=None, min_seq_len=256)
    with SDPACapture(cfg) as cap:
        q = torch.randn(1, 4, 1024, 16)
        F.scaled_dot_product_attention(q, torch.randn(1, 4, 1024, 16),
                                       torch.randn(1, 4, 1024, 16))   # self
        F.scaled_dot_product_attention(q, torch.randn(1, 4, 512, 16),
                                       torch.randn(1, 4, 512, 16))    # cross
    calls = cap.self_attention_calls()
    assert len(calls) == 1 and calls[0]["n_query"] == calls[0]["n_key"] == 1024


def _run_all():
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures.append(name)
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
