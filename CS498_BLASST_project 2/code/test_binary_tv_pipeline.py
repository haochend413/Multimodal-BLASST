"""Check staged Boolean selection and causal score computation."""
import math
from types import SimpleNamespace

import pytest
import torch

import blasst_attn as ba
from binary_tv_pipeline import BinaryTVPipeline


def test_binary_pipeline_gates_tv_then_vv_and_updates_decode_history(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    visual = torch.tensor([False, False, True, True, True, True, False, False])
    q = torch.ones(1, 1, 8, 1)
    k = torch.tensor([1., 1., 8., 8., 1., 1., 1., 1.]).view_as(q)
    v = torch.arange(8.).view_as(q)
    first, second = SimpleNamespace(layer_idx=0), SimpleNamespace(layer_idx=1)
    baseline, _ = ba.blasst_attention_forward(first, q, k, v, None, scaling=1.)

    results = []
    for gate_vv in (False, True):
        pipeline = BinaryTVPipeline(visual, 2, gate_vv=gate_vv)
        monkeypatch.setitem(ba.STATE, "binary_tv_pipeline", pipeline)
        warmup, _ = ba.blasst_attention_forward(first, q, k, v, None, scaling=1.)
        torch.testing.assert_close(warmup, baseline)
        assert pipeline.layers[0][0][0, 6].tolist() == [True, False]
        prefill, _ = ba.blasst_attention_forward(second, q, k, v, None, scaling=1.)
        assert pipeline.layers[1][0][0, 6:].tolist() == [[True, False], [True, False]]
        assert pipeline.counts["prefill_tv_qk_pairs_bypassed"] == 4
        assert pipeline.counts["prefill_vv_qk_pairs_bypassed"] == (7 if gate_vv else 0)
        results.append(prefill)
        k_decode = torch.cat((k, k[:, :, -1:]), dim=2)
        v_decode = torch.cat((v, v[:, :, -1:]), dim=2)
        for module in (first, second):
            output, _ = ba.blasst_attention_forward(module, q[:, :, -1:], k_decode, v_decode, None, scaling=1.)
            assert output.isfinite().all()
            assert pipeline.layers[module.layer_idx][0][0, 8].tolist() == [True, False]
        assert pipeline.counts["decode_tv_qk_pairs_bypassed"] == 2
        assert pipeline.counts["decode_tv_qk_pairs"] == 8
    torch.testing.assert_close(results[0][:, 6:], results[1][:, 6:])
    torch.testing.assert_close(results[1][0, 4:6, 0, 0], torch.tensor([.5, .5]))
    assert not torch.allclose(results[0][:, 4:6], results[1][:, 4:6])


def test_binary_pipeline_distinguishes_empty_history_from_no_history(monkeypatch):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    pipeline = BinaryTVPipeline(torch.tensor([False, True, False]), 2, gate_vv=True)
    monkeypatch.setitem(ba.STATE, "binary_tv_pipeline", pipeline)
    q = torch.ones(1, 1, 3, 1)
    v = torch.arange(3.).view_as(q)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, q, v, None, scaling=1.)
    links, history = pipeline.layers[0]
    links.zero_()
    assert history[0, 2]
    layer = SimpleNamespace(layer_idx=1)
    ba.blasst_attention_forward(layer, q, q, v, None, scaling=1.)
    assert pipeline.counts["prefill_tv_qk_pairs_bypassed"] == 1
    assert pipeline.counts["prefill_tv_fallback_rows"] == 0
    history.zero_()
    mask = torch.zeros(1, 1, 3, 3)
    mask[..., 2, 1] = torch.finfo(mask.dtype).min
    out, _ = ba.blasst_attention_forward(layer, q, q, v, mask, scaling=1.)
    assert pipeline.counts["prefill_tv_fallback_rows"] == 0
    assert not pipeline.layers[1][1][0, 2]
    assert out.isfinite().all()
    ba.blasst_attention_forward(layer, q, q, v, None, scaling=1.)
    assert pipeline.counts["prefill_tv_fallback_rows"] == 1
    assert pipeline.layers[1][0][0, 2, 0]


@pytest.mark.parametrize("guided_phase", ["prefill", "decode"])
def test_binary_phase_study_uses_ordinary_blasst_in_the_other_phase(monkeypatch, guided_phase):
    monkeypatch.setattr(ba, "TILE", 2)
    monkeypatch.setitem(ba.STATE, "ln_lam", math.log(.1))
    monkeypatch.setitem(ba.STATE, "decode_override", False)
    monkeypatch.setitem(ba.STATE, "collect_stats", False)
    visual = torch.tensor([False, False, True, True, True, True, False, False])
    q = torch.ones(1, 1, 8, 1)
    k = torch.tensor([1., 1., 8., 8., 1., 1., 1., 1., 1.]).view(1, 1, 9, 1)
    v = torch.arange(9.).view_as(k)
    layer = SimpleNamespace(layer_idx=2)
    expected_prefill, _ = ba.blasst_attention_forward(layer, q, k[:, :, :8], v[:, :, :8], None, scaling=1.)
    expected_decode, _ = ba.blasst_attention_forward(layer, q[:, :, :1], k, v, None, scaling=1.)
    pipeline = BinaryTVPipeline(visual, 2, gate_vv=False,
                                guide_prefill=guided_phase == "prefill", guide_decode=guided_phase == "decode")
    monkeypatch.setitem(ba.STATE, "binary_tv_pipeline", pipeline)
    for index in range(3):
        module = SimpleNamespace(layer_idx=index)
        prefill, _ = ba.blasst_attention_forward(module, q, k[:, :, :8], v[:, :, :8], None, scaling=1.)
        assert pipeline.layers[index][0][0, 7].tolist() == [True, False]
    for index in range(3):
        module = SimpleNamespace(layer_idx=index)
        decode, _ = ba.blasst_attention_forward(module, q[:, :, :1], k, v, None, scaling=1.)
    if guided_phase == "prefill":
        torch.testing.assert_close(decode, expected_decode, rtol=0, atol=0)
        assert pipeline.counts["decode_tv_qk_pairs_bypassed"] == 0
        assert pipeline.counts["prefill_tv_qk_pairs_bypassed"] > 0
    else:
        torch.testing.assert_close(prefill, expected_prefill, rtol=0, atol=0)
        assert pipeline.counts["prefill_tv_qk_pairs_bypassed"] == 0
        assert pipeline.counts["decode_tv_qk_pairs_bypassed"] > 0
    assert pipeline.counts["prefill_vv_qk_pairs_bypassed"] == 0
