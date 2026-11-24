"""Check that restartable shard reports preserve all paired VQA questions."""
from argparse import ArgumentParser, Namespace
import json

import pytest

import attention_study
import vqa_shards


def test_shards_resume_and_merge_without_losing_pairs(tmp_path, monkeypatch):
    image = tmp_path / "example.jpg"
    image.touch()
    samples = [
        dict(id=str(index), image=str(image), text_before="", question="Original question",
             benchmark=dict(question_id=index, question_text="How many?", reference_answers=["2"] * 10))
        for index in range(5)
    ]
    scored_ids = []

    def collect(run, _parser, benchmark, _sample_id):
        scored_ids.append(benchmark["question_id"])
        baseline = run.tv_vv_threshold is None
        return dict(answer="2" if baseline else "3",
                    tv_guided_vv=dict(pure_vv_tiles=4, vv_tiles_bypassed=0 if baseline else 2, layers=[42]))

    monkeypatch.setattr(attention_study, "collect", collect)

    def run_shard(index):
        start, stop = vqa_shards.partition(len(samples), 2, index)
        path = tmp_path / f"shard-{index}.json"
        args = Namespace(tv_vv_sweep=[.5], model="test", demo=False, lam=.1, decode_lambda=None,
                         sample_index=start, output=path, vqa_accuracy=True, decode_tokens=16,
                         image_max_pixels=1280, max_tokens=8192,
                         vqa_shard=dict(index=index, count=2, manifest_questions=5, manifest_sha256="fixture"))
        attention_study._collect_tv_vv_sweep(args, ArgumentParser(), samples[start:stop])
        return path

    first = run_shard(0)
    assert scored_ids == [0, 0, 1, 1]
    saved = json.loads(first.read_text())
    assert "layers" not in saved["cases"][0]["results"]["baseline"]
    saved["cases"].pop()
    first.write_text(json.dumps(saved))
    run_shard(0)
    assert scored_ids == [0, 0, 1, 1, 1, 1]
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
    assert [row["question_id"] for row in predictions] == list(range(5))
    assert all(row["answer"] == "2" for row in predictions)
    with pytest.raises(ValueError, match="duplicate shard"):
        vqa_shards.merge([first, second, second], output)
