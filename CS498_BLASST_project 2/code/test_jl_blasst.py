import math
from types import SimpleNamespace

import pytest
import torch

import blasst_attn as ba
from attention_study import AttentionRecorder
from jl_blasst import JLAudit, JLPrecheck, JLStudy, ProjectedJLStudy


@pytest.mark.parametrize("query_len,epsilon", [(258, None), (1, None), (1, 0.5)])
def test_matches_exact_blasst_with_real_early_skips(monkeypatch, query_len, epsilon):
    rng = torch.Generator().manual_seed(13)
    key_len, width, lam = 258, 16, 0.1
    q = torch.randn(query_len, width, generator=rng) * 0.1
    k = torch.randn(key_len, width, generator=rng) * 0.1
    v = torch.randn(key_len, width, generator=rng)
    q[:, 0] = 4
    k[0, 0] = 50
    visual = torch.arange(key_len) >= 128
    visual[-1] = False
    mask = torch.ones(query_len, key_len, dtype=torch.bool)
    mask[:, 6] = False
    local = JLAudit()
    probe = JLPrecheck(8, epsilon=epsilon)
    actual = probe.forward(q, k, v, lam=lam, mask=mask)
    audited = probe.forward(q, k, v, lam=lam, mask=mask, audit=local, visual=visual)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(lam))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    expected, _ = ba.blasst_attention_forward(
        SimpleNamespace(), q[None, None], k[None, None], v[None, None], mask[None, None],
    )
    torch.testing.assert_close(actual, expected[0, :, 0])
    torch.testing.assert_close(actual, audited)
    stats = local.summary()
    counts = stats["tile_decisions"]["all"]
    assert 0 < counts["early_skips"] <= counts["baseline_skips"]
    assert counts["false_early_skips"] == counts["skip_mask_disagreements"] == 0
    assert "T->V" in stats["token_pair_errors"]
    if epsilon is None:
        assert sum(row["upper_violations"] for row in stats["token_pair_errors"].values()) == 0


def test_aggressive_jl_can_false_skip_and_audit_catches_it():
    probe = JLPrecheck(1, epsilon=0)
    direction = probe.basis(4, torch.device("cpu"))[:, 0]
    # Projected logit is zero, but the second key is actually needed.
    null = torch.linalg.qr(probe.basis(4, torch.device("cpu")), mode="complete").Q[:, 1]
    q = (2 * direction + 2 * null).float()[None]
    k = torch.stack((2 * direction, 3 * null)).float()
    v = torch.tensor([[1.0], [2.0]])
    audit = JLAudit()
    probe.forward(q, k, v, tile=1, lam=0.5, audit=audit, visual=torch.tensor([False, True]))
    counts = audit.summary()["tile_decisions"]["all"]
    assert counts["false_early_skips"] == counts["skip_mask_disagreements"] == 1


def test_live_hook_preserves_grouped_query_attention(monkeypatch):
    rng = torch.Generator().manual_seed(4)
    q = torch.randn(1, 4, 1, 16, generator=rng)
    k = torch.randn(1, 2, 131, 16, generator=rng)
    v = torch.randn(k.shape, generator=rng)
    mask = torch.zeros((1, 1, 1, 131))
    mask[..., 4] = -torch.inf
    module = SimpleNamespace(layer_idx=1)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(0.1))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    expected, _ = ba.blasst_attention_forward(module, q, k, v, mask)
    live = JLStudy(JLPrecheck(8), torch.zeros(130, dtype=torch.bool), {1}, {0},
                   0.1, 128, live=True)
    monkeypatch.setitem(ba.STATE, "qkv_hook", live.observe)
    actual, _ = ba.blasst_attention_forward(module, q, k, v, mask)
    torch.testing.assert_close(actual, expected)
    assert live.summary()["phases"]["decode"]["tiles"] == 8


def test_projected_only_uses_its_own_running_max(monkeypatch):
    probe = JLPrecheck(1)
    basis = probe.basis(4, torch.device("cpu"))
    p, residual = torch.linalg.qr(basis, mode="complete").Q[:, :2].T
    q = (2 * (p + residual)).float()[None, None, None, :]
    # Tile scores are approximately [0, 2, -1] in full and [12, 0, 16] in projection.
    centers = torch.stack((3 * p - 3 * residual, 2 * residual, 4 * p - 5 * residual))
    k = centers.float().repeat_interleave(torch.tensor([128, 128, 1]), dim=0)[None, None]
    visual = torch.tensor([False] * 128 + [True] * 128)
    recorder = AttentionRecorder(visual, {1}, {0})
    projected = ProjectedJLStudy(probe, {1}, {0}, 0.1)
    recorder.projected_study = projected
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(0.1))
    monkeypatch.setitem(ba.STATE, "a", None)
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "decode_enabled", True)
    monkeypatch.setitem(ba.STATE, "observer", recorder.observe)
    monkeypatch.setitem(ba.STATE, "qkv_hook", projected.observe)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=1), q, k, k, None)
    tiles = recorder.views["decode:0:1:0"]["tiles"]
    assert [t["full_skipped_at_jl_lambda"] for t in tiles] == [False, False, True]
    assert [t["full_skipped_at_jl_lambda"] for t in tiles] == [t["skipped"] for t in tiles]
    assert [t["projected_skipped"] for t in tiles] == [False, True, False]
    counts = projected.summary()["phases"]["decode"]
    assert counts["all"]["missed_skips"] == counts["all"]["false_skips"] == 1
    assert counts["key:V"]["false_skips"] == 1
