"""Check that restartable shard reports preserve all paired VQA questions."""
from argparse import ArgumentParser, Namespace
import json
import sys

import pytest

import attention_study
import vqa_shards


@pytest.mark.parametrize("window", [{}, {"selection_start": 2, "selection_count": 5}])
def test_shards_resume_and_merge_without_losing_pairs(tmp_path, monkeypatch, window):
    image = tmp_path / "example.jpg"
    image.touch()
    samples = [
        dict(id=str(index), image=str(image), text_before="", question="Original question",
             benchmark=dict(question_id=index, question_text="How many?", reference_answers=["2"] * 10))
        for index in range(8 if window else 5)
    ]
    scored_ids = []

    def collect(run, _parser, benchmark, _sample_id):
        scored_ids.append(benchmark["question_id"])
        baseline = run.tv_vv_threshold is None
        return dict(answer="2" if baseline else "3",
                    tv_guided_vv=dict(pure_vv_tiles=4, vv_tiles_bypassed=0 if baseline else 2, layers=[42]))

    monkeypatch.setattr(attention_study, "collect", collect)

    def run_shard(index):
        shard = dict(index=index, count=2, manifest_questions=len(samples), manifest_sha256="fixture", **window)
        start, stop = vqa_shards.shard_bounds(shard)
        path = tmp_path / f"shard-{index}.json"
        args = Namespace(tv_vv_sweep=[.5], model="test", demo=False, lam=.1, decode_lambda=None,
                         sample_index=start, output=path, vqa_accuracy=True, decode_tokens=16,
                         image_max_pixels=1280, max_tokens=8192,
                         vqa_shard=shard)
        attention_study._collect_tv_vv_sweep(args, ArgumentParser(), samples[start:stop])
        return path

    first = run_shard(0)
    offset = window.get("selection_start", 0)
    assert scored_ids == [offset, offset, offset + 1, offset + 1]
    saved = json.loads(first.read_text())
    assert "layers" not in saved["cases"][0]["results"]["baseline"]
    saved["cases"].pop()
    first.write_text(json.dumps(saved))
    run_shard(0)
    assert scored_ids == [offset, offset] + [offset + 1] * 4
    output = tmp_path / "merged.json"
    with pytest.raises(ValueError, match="missing shards"):
        vqa_shards.merge([first], output)
    second = run_shard(1)
    assert vqa_shards.merge([second, first], output)["pairs_completed"] == 5
    result = json.loads(output.read_text())
    assert result["summary"]["baseline"]["vqa_accuracy"] == 100
    assert result["summary"]["0.5"]["vqa_delta_pp"] == -100
    assert result["summary"]["0.5"]["vv_tiles_bypassed"] == 10
    predictions = json.loads((tmp_path / "merged-baseline-predictions.json").read_text())
    assert [row["question_id"] for row in predictions] == list(range(offset, offset + 5))
    assert all(row["answer"] == "2" for row in predictions)
    with pytest.raises(ValueError, match="duplicate shard"):
        vqa_shards.merge([first, second, second], output)


def test_cli_applies_sample_count_before_sharding(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    samples = [{"id": str(index), "benchmark": {
        "question_id": index, "name": "VQA v2", "question_text": "Question", "reference_answers": ["yes"] * 10,
    }} for index in range(8)]
    manifest.write_text(json.dumps({"schema": "blasst-benchmark-manifest-v1", "samples": samples}))
    monkeypatch.setattr(attention_study.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [
        "attention_study.py", "--manifest", str(manifest), "--output", str(tmp_path / "report.json"),
        "--sample-index", "2", "--sample-count", "4", "--vqa-shards", "2", "--vqa-shard-index", "1",
        "--lambda", ".1", "--tv-vv-sweep", ".5", "--vqa-accuracy", "--decode-region-max",
    ])

    def collect(args, _parser, selected):
        assert args.sample_index == 4
        assert args.sample_count == 2
        assert args.vqa_shard["selection_count"] == 4
        assert args.decode_region_max
        assert [case["id"] for case in selected] == ["4", "5"]

    monkeypatch.setattr(attention_study, "_collect_tv_vv_sweep", collect)
    attention_study.main()
