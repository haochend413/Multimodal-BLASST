"""Check pure V→V relaxation and causal masks on the original physical tiles."""
from types import SimpleNamespace

import torch

import blasst_attn as ba


def test_bidirectional_vv_tiles_remain_eligible_for_skipping(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    visual = torch.tensor([False, True, True, True, True, True, True, False])
    q = torch.zeros(1, 1, 8, 8)
    q[..., 0] = 1
    k = torch.zeros_like(q)
    k[..., 0] = torch.tensor([4., 3., 1., 1., 0., 0., 0., 0.])
    values = torch.eye(8)[None, None]
    chunks = {}
    extra_mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    extra_mask[0, 0, 2, 5] = False

    def observe(**record):
        chunks[record["q_start"]] = record

    monkeypatch.setitem(ba.STATE, "observer", observe)
    monkeypatch.setitem(ba.STATE, "vv_prefix_visual", visual)
    monkeypatch.setitem(ba.STATE, "ln_lam", 0.)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    output, _ = ba.blasst_attention_forward(SimpleNamespace(), q, k, values, extra_mask, scaling=1.)
    all_rows = torch.cat([chunks[q_start]["scores"][0, 0] for q_start in range(0, 8, 2)])
    expected = torch.ones(8, 8, dtype=torch.bool).tril()
    expected[2:6, 2:6] = True
    expected &= extra_mask[0, 0]
    assert torch.equal(torch.isfinite(all_rows), expected)

    assert chunks[2]["margin"][0, 0, 2] < 0
    assert chunks[2]["skip"][0, 0, 2]
    assert output[0, 2, 0, 4] == 0
    torch.testing.assert_close(output.sum(-1), torch.ones(1, 8, 1))

    k[..., 4, 0] = 5
    output, _ = ba.blasst_attention_forward(SimpleNamespace(), q, k, values, extra_mask, scaling=1.)
    assert not chunks[2]["skip"][0, 0, 2]
    assert output[0, 2, 0, 4] > 0
    assert output[0, 2, 0, 5] == 0
    assert output[0, 1, 0, 2] == 0
    assert output[0, 4, 0, 6] == 0
    torch.testing.assert_close(output.sum(-1), torch.ones(1, 8, 1))


def test_prefix_setting_does_not_change_decode(monkeypatch):
    torch.manual_seed(3)
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 1, 7, 4)
    v = torch.randn_like(k)
    monkeypatch.setitem(ba.STATE, "vv_prefix_visual", None)
    monkeypatch.setitem(ba.STATE, "decode_enabled", True)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    monkeypatch.setitem(ba.STATE, "ln_lam", -float("inf"))
    expected, _ = ba.blasst_attention_forward(SimpleNamespace(), q, k, v, None)
    monkeypatch.setitem(ba.STATE, "vv_prefix_visual", torch.tensor([False, True, True, False, False, False]))
    actual, _ = ba.blasst_attention_forward(SimpleNamespace(), q, k, v, None)
    torch.testing.assert_close(actual, expected)
