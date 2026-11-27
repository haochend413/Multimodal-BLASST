from argparse import Namespace
from itertools import combinations
import json
from random import Random
import sys

import torch

import attention_study as study
from binary_phase_controls import DecodeBudgets, random_prefill_gate


def test_random_prefill_preserves_exact_row_and_pair_budgets():
    visual_positions = [3, 4, 5, 6, 8, 9]
    ranks = {position: i for i, position in enumerate(visual_positions, 1)}
    rng = Random(10)
    guided = {i: rows for i, rows in enumerate(
        rows for count in range(7) for rows in combinations(visual_positions, count)
    )}
    random = random_prefill_gate(visual_positions, guided, rng)
    for layer, dropped in random.items():
        assert len(set(dropped)) == len(guided[layer])
        assert sum(map(ranks.get, dropped)) == sum(map(ranks.get, guided[layer]))
    assert any(set(rows) != set(guided[layer]) for layer, rows in random.items())


def test_random_decode_preserves_tiles_pairs_and_fallback_and_reports_extra_steps():
    reference = DecodeBudgets()
    allowed = torch.ones((2, 5), dtype=torch.bool)
    columns = torch.tensor([0, 1, 1, 2, 2])
    chosen = torch.tensor([[True, False, True], [False, False, False]])
    fallback = torch.tensor([False, True])
    reference.select(2, allowed, columns, chosen, fallback)
    replay = DecodeBudgets(Random(10), reference)
    for _ in range(2):
        kept, used_fallback = replay.select(2, allowed, columns, ~chosen, ~fallback)
        assert used_fallback.equal(fallback)
        assert kept[0, 0]
        assert kept[0, 1:].sum() == 1
        assert kept[1].all()
        assert (kept[:, columns] | used_fallback[:, None]).sum().item() == 8
    assert replay.tail_head_steps == 2
    full, used_fallback = replay.select(3, allowed, columns, chosen, fallback)
    assert full.all() and used_fallback.all()
    assert replay.tail_head_steps == 4


def test_phase_study_replays_budget_from_each_guided_variant(monkeypatch, tmp_path):
    args = Namespace(
        answer_protocol=study.VQAV2, model="test", demo=False, lam=.1, decode_lambda=.1,
        sample_index=0, decode_tokens=16, vqa_accuracy=True, output=tmp_path / "study.json",
        binary_mask_study=True, matched_random=True, random_seed=12, image_max_pixels=3136, max_tokens=128,
    )
    sample = {"id": "example", "benchmark": {"question_id": 1, "question_text": "Question?",
                                           "reference_answers": ["yes"] * 10}}
    seen = []

    def collect(run, *_):
        seen.append((run.token_gate, run.pipeline_decode_links))
        result = {"answer": "yes", "generated_tokens": 2}
        if run.token_gate:
            gate = run.gate_replay or {0: [], 1: [1, 4]}
            result.update(visual_positions=[1, 2, 3, 4], prefill_gate=gate,
                          tv_guided_vv={"vv_pairs_bypassed": sum(sum(row) for row in gate.values())})
        if run.pipeline_decode_links:
            assert run.decode_budgets is not None
            allowed = torch.ones(1, 2, dtype=torch.bool)
            columns = torch.tensor([0, 1])
            chosen = torch.tensor([[True, False]])
            fall = torch.tensor([False])
            run.decode_budgets.select(0, allowed, columns, chosen, fall)
            result["decode_budgets"] = run.decode_budgets
        return result

    monkeypatch.setattr(study, "collect", collect)
    monkeypatch.setattr(study, "_select_sample", lambda *_: None)
    study._run_tv_vv_sweep(args, None, [sample])
    assert seen == [(None, False), ("vv", False), (None, True), ("vv", True),
                    ("vv", False), (None, True), ("vv", True)]
    report = json.loads(args.output.read_text())
    assert report["summary"]["combined_binary_random"]["vqa_delta_vs_guided_pp"] == 0
    assert report["summary"]["prefill_binary_random"]["vv_pairs_bypassed"] == 5
    assert report["summary"]["combined_binary_random"]["random_decode_unmatched_head_steps"] == 0


def test_fixed_length_timing_separates_first_token_and_decode(monkeypatch):
    from binary_phase_controls import timed_generation

    class FakeGeneration(torch.nn.Module):
        def forward(self):
            return None

        def generate(self, input_ids, *, min_new_tokens, max_new_tokens, do_sample):
            assert min_new_tokens == max_new_tokens == 3
            assert not do_sample
            for _ in range(max_new_tokens):
                self()
            return torch.cat((input_ids, torch.ones(1, max_new_tokens, dtype=torch.long)), dim=1)

    class Inputs(dict):
        @property
        def input_ids(self):
            return self["input_ids"]

    times = iter([10, 12, 17])
    monkeypatch.setattr("binary_phase_controls.perf_counter", lambda: next(times))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    model = FakeGeneration()
    output, timing = timed_generation(model, Inputs(input_ids=torch.arange(4).unsqueeze(0)), 3)
    assert output.shape == (1, 7)
    assert timing == {"first_token_seconds": 2, "decode_seconds": 5, "generation_seconds": 7}
    assert not model._forward_pre_hooks


def test_prefill_drop_sweep_pairs_each_rate_and_scores_baseline(monkeypatch, tmp_path):
    args = Namespace(
        answer_protocol=study.VQAV2, model="test", demo=False, lam=.1, decode_lambda=.1,
        sample_index=0, decode_tokens=16, vqa_accuracy=True, output=tmp_path / "sweep.json",
        prefill_vv_drop_sweep=[99.9, 25.0], random_seed=12, image_max_pixels=3136, max_tokens=128,
    )
    sample = {"id": "example", "benchmark": {"question_id": 1, "question_text": "Question?",
                                           "reference_answers": ["yes"] * 10}}
    seen = []

    def collect(run, *_):
        assert run.token_gate == "vv" and not run.decode_visual_links
        seen.append((run.token_drop_percent, run.random_drop))
        dropped = int(run.token_drop_percent * 4 / 100)
        return {"answer": "yes", "generated_tokens": 2, "tv_guided_vv": {
            "visual_query_tokens_gateable": 4, "visual_query_tokens": 8, "visual_query_tokens_selected": 8 - dropped,
            "vv_pairs": 20, "vv_pairs_bypassed": dropped * 2, "layers": [],
        }}

    monkeypatch.setattr(study, "collect", collect)
    monkeypatch.setattr(study, "_select_sample", lambda *_: None)
    study._run_tv_vv_sweep(args, None, [sample])
    assert seen == [(0, True), (99.9, False), (99.9, True), (25.0, False), (25.0, True)]
    report = json.loads(args.output.read_text())
    assert report["summary"]["baseline"]["prefill_visual_row_drop_pct"] == 0
    assert report["summary"]["prefill_drop_99.9"]["prefill_visual_row_drop_pct"] == 75
    assert report["summary"]["prefill_drop_99.9_random"]["prefill_vv_qk_bypass_pct"] == 30
    assert report["summary"]["prefill_drop_25_random"]["vqa_delta_vs_guided_pp"] == 0


def test_prefill_co_design_cli_sweeps_lambdas_and_supports_highest_drop(monkeypatch, tmp_path):
    manifest, output = tmp_path / "manifest.json", tmp_path / "sweep.json"
    sample = {"id": "example", "benchmark": {"name": "VQA v2", "question_id": 1, "question_text": "Question?",
                                           "reference_answers": ["yes"] * 10}}
    manifest.write_text(json.dumps({"schema": "blasst-benchmark-manifest-v1", "samples": [sample]}))
    seen = []

    def collect(run, *_):
        seen.append((run.lam, run.decode_lambda, run.token_drop_percent,
                     run.random_drop, run.prefill_vv_drop_highest))
        return {"answer": "yes", "generated_tokens": 2, "tv_guided_vv": {
            "vv_pairs": 20, "vv_pairs_bypassed": int(20 * run.token_drop_percent / 100), "layers": [],
        }}

    monkeypatch.setattr(study, "collect", collect)
    monkeypatch.setattr(study, "_select_sample", lambda *_: None)
    monkeypatch.setattr(study.torch.cuda, "is_available", lambda: True)
    common = [
        "attention_study.py", "--manifest", str(manifest), "--decode-lambda", "0.1",
        "--vqa-shards", "1", "--vqa-shard-index", "0",
    ]
    monkeypatch.setattr(sys, "argv", common + [
        "--output", str(output), "--prefill-vv-drop-sweep", "80", "--prefill-vv-lambda-sweep", "0.05,0.2",
    ])
    study.main()
    study.main()  # Completed child checkpoints are reusable by the sweep.
    assert seen == [(lam, .1, drop, random, False)
                    for lam in (.05, .2) for drop, random in ((0, True), (80, False), (80, True))]
    report = json.loads(output.read_text())
    assert set(report["summary"]) == {"0.05", "0.2"}
    for lam, filename in report["reports"].items():
        child = json.loads((tmp_path / filename).read_text())
        assert child["blasst_lambda"] == float(lam)
        assert child["summary"] == report["summary"][lam]

    seen.clear()
    reverse_output = tmp_path / "highest.json"
    monkeypatch.setattr(sys, "argv", common + [
        "--output", str(reverse_output), "--lambda", "0.1",
        "--prefill-vv-drop-sweep", "5,80", "--prefill-vv-drop-highest",
    ])
    study.main()
    assert seen == [(.1, .1, drop, random, True)
                    for drop, random in ((0, True), (5, False), (5, True), (80, False), (80, True))]
    highest = json.loads(reverse_output.read_text())
    assert highest["prefill_vv_drop_order"] == "highest"
    assert highest["variant_labels"]["prefill_drop_80"] == "Highest T→V frequency drop 80%"


def test_decode_drop_sweep_pairs_each_rate_and_reports_realized_budget(monkeypatch, tmp_path):
    manifest, output = tmp_path / "manifest.json", tmp_path / "sweep.json"
    sample = {"id": "example", "benchmark": {"name": "VQA v2", "question_id": 1, "question_text": "Question?",
                                           "reference_answers": ["yes"] * 10}}
    manifest.write_text(json.dumps({"schema": "blasst-benchmark-manifest-v1", "samples": [sample]}))
    seen = []

    def collect(run, *_):
        assert run.token_gate is None and run.decode_visual_link_study
        seen.append((run.decode_drop_percent, run.random_decode_drop))
        result = {"answer": "yes", "generated_tokens": 2}
        assert run.decode_visual_links == (run.decode_drop_percent is not None)
        if run.decode_visual_links:
            assert run.extend_decode_links
            result["decode_visual_links"] = {
                "decode_visual_keys": 101, "decode_visual_keys_bypassed": int(run.decode_drop_percent * 101 / 100)
            }
        return result

    monkeypatch.setattr(study, "collect", collect)
    monkeypatch.setattr(study, "_select_sample", lambda *_: None)
    monkeypatch.setattr(study.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [
        "attention_study.py", "--manifest", str(manifest), "--output", str(output),
        "--lambda", "0.1", "--decode-tv-drop-sweep", "5,99.9", "--vqa-shards", "1", "--vqa-shard-index", "0",
    ])
    study.main()
    study.main()  # Resume a completed shard without collecting any more variants.
    assert seen == [(None, False), (5, False), (5, True), (99.9, False), (99.9, True)]

    report = json.loads(output.read_text())
    assert report["decode_tv_drop_sweep"] == [5, 99.9]
    assert report["summary"]["decode_drop_99.9"]["decode_visual_key_drop_pct"] == 100 * 100 / 101
    assert report["summary"]["decode_drop_99.9_random"]["vqa_delta_vs_guided_pp"] == 0
