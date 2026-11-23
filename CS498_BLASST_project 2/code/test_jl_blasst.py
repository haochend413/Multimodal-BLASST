import argparse
import json
import math
from types import SimpleNamespace

import pytest
import torch

import blasst_attn as ba
import attention_study
from attention_study import AttentionRecorder
from jl_blasst import (
    JLAudit, JLPrecheck, JLStudy, ProjectedJLStudy, VISUAL_KEY_QUERY_SPLIT, _split_margins,
)


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


def test_split_margins_match_explicit_masked_padding():
    scores = torch.tensor([[3.0, -torch.inf, -torch.inf], [2.0, -1.0, 0.5]])
    query_spans, key_spans = [(0, 1), (1, 2)], [(0, 1), (1, 3)]
    expected = []
    for qlo, qhi in query_spans:
        padded = []
        for klo, khi in key_spans:
            tile = scores[qlo:qhi, klo:khi]
            padded.append(torch.nn.functional.pad(tile, (0, 128 - tile.shape[1],
                                                         0, 128 - tile.shape[0]), value=-torch.inf))
        expected.append(ba.tile_margins(torch.stack(padded, dim=-2)))
    torch.testing.assert_close(_split_margins(scores, query_spans, key_spans), torch.stack(expected))


@pytest.mark.parametrize("split", ["query", "key", "both"])
def test_split_prefill_preserves_positions_and_valid_pairs(monkeypatch, split):
    rng = torch.Generator().manual_seed(5)
    n = 132
    q, k = (torch.randn(1, 1, n, 16, generator=rng) for _ in range(2))
    visual = (torch.arange(n) >= 15) & (torch.arange(n) < 130)
    recorder = AttentionRecorder(visual, {1}, {0})
    projected = ProjectedJLStudy(JLPrecheck(8), {1}, {0}, 0.1, split=split, visual=visual)
    recorder.projected_study = projected
    monkeypatch.setitem(ba.STATE, "observer", recorder.observe)
    monkeypatch.setitem(ba.STATE, "qkv_hook", projected.observe)
    module = SimpleNamespace(layer_idx=1)
    ba.blasst_attention_forward(module, q, k, k, None)
    view = recorder.views["prefill:-1:1:0"]
    coverage = torch.zeros((n, n), dtype=torch.int32)
    positions = torch.arange(n)
    for tile in view["split_tiles"]:
        q_rows = slice(tile["q"], tile["q"] + tile["rows"])
        k_columns = slice(tile["k"], tile["k"] + tile["cols"])
        causal = positions[q_rows, None] >= positions[None, k_columns]
        assert causal.any()
        coverage[q_rows, k_columns] += causal
        if split in ("query", "both"):
            q_mod = visual[q_rows]
            assert q_mod.all() or not q_mod.any()
        if split in ("key", "both"):
            k_mod = visual[k_columns]
            assert k_mod.all() or not k_mod.any()
    torch.testing.assert_close(coverage, (positions[:, None] >= positions[None, :]).int())
    assert len(view["split_tiles"]) > len(view["tiles"])
    counts = projected.summary()
    assert counts["split_prefill"]["all"]["tiles"] == len(view["split_tiles"])
    assert counts["split_prefill"]["all"]["valid_pairs"] == n * (n + 1) // 2
    assert counts["phases"]["prefill"]["all"]["valid_pairs"] == n * (n + 1) // 2


def test_selective_split_only_partitions_mixed_queries_against_visual_keys(monkeypatch):
    rng = torch.Generator().manual_seed(5)
    n = 390
    q, k = (torch.randn(1, 1, n, 16, generator=rng) for _ in range(2))
    visual = (torch.arange(n) >= 15) & (torch.arange(n) < 386)
    recorder = AttentionRecorder(visual, {1}, {0})
    projected = ProjectedJLStudy(JLPrecheck(8), {1}, {0}, 0.1,
                                 split=VISUAL_KEY_QUERY_SPLIT, visual=visual)
    recorder.projected_study = projected
    monkeypatch.setitem(ba.STATE, "observer", recorder.observe)
    monkeypatch.setitem(ba.STATE, "qkv_hook", projected.observe)
    ba.blasst_attention_forward(SimpleNamespace(layer_idx=1), q, k, k, None)

    view = recorder.views["prefill:-1:1:0"]
    originals = {(tile["q"], tile["k"]): tile for tile in view["tiles"]}
    selective = {(tile["q"], tile["k"]): tile for tile in view["split_tiles"]}
    targeted = {(384, 128), (384, 256)}
    for position, tile in originals.items():
        kept = selective[position]
        if position in targeted:
            assert (kept["rows"], kept["q_kind"], kept["k_kind"]) == (2, "V", "V")
            text = selective[386, position[1]]
            assert (text["rows"], text["q_kind"], text["k_kind"]) == (4, "T", "V")
        else:
            for field in ("rows", "cols", "margin", "projected_margin", "projected_skipped",
                          "full_skipped_at_jl_lambda"):
                assert kept[field] == tile[field]
    assert len(selective) == len(originals) + len(targeted)
    summary = projected.summary()
    original_counts = summary["phases"]["prefill"]["all"]
    split_counts = summary["split_prefill"]["all"]
    assert split_counts["valid_pairs"] == original_counts["valid_pairs"]
    assert split_counts["dense_weight"] == pytest.approx(original_counts["dense_weight"])


def test_projected_batch_counts_pairs_with_false_skips(tmp_path, monkeypatch):
    output = tmp_path / "summary.json"
    image = tmp_path / "image.jpg"
    image.touch()
    samples = [
        {"image": str(image), "text_before": "", "question": "What is shown?",
         "benchmark": {}, "id": f"sample-{i}"}
        for i in range(2)
    ]
    args = argparse.Namespace(output=output, sample_index=3, model="test",
                              manifest=tmp_path / "vqa.json", layers="0", heads="4", decode_tokens=2)

    def collect_sample(args, parser, benchmark, sample_id):
        if sample_id == "sample-1":
            assert json.loads(output.read_text())["pairs_completed"] == 1
        false_skips = 2 if sample_id == "sample-0" else 0
        args.output.parent.mkdir(exist_ok=True)
        args.output.write_text(sample_id)
        return {"jl_projected": {"dimension": 32, "seed": 0, "lambda": 0.1,
                                 "phases": {"decode": {"all": {"tiles": 4, "false_skips": false_skips}}}}}

    monkeypatch.setattr(attention_study, "collect", collect_sample)
    attention_study._collect_batch(args, argparse.ArgumentParser(), samples)
    report = json.loads(output.read_text())
    assert report["pairs_completed"] == 2
    assert report["phases"]["decode"]["all"] == {
        "tiles": 8, "false_skips": 2, "pairs_observed": 2, "pairs_with_false_skips": 1,
    }
    assert [row["sample_index"] for row in report["samples"]] == [3, 4]
    assert (tmp_path / report["samples"][1]["audit"]).read_text() == "sample-1"
