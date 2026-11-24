"""Check official scoring and the paired VQA sweep output."""
import json
from argparse import ArgumentParser, Namespace

import pytest

import attention_study
from vqav2_accuracy import score_answer, validate_cases


def test_official_consensus_and_normalization():
    references = ["two", "2", "two"] + ["3"] * 7
    assert score_answer("The two.", references) == .9 * 100
    assert score_answer("three", references) == 100
    assert score_answer("There are two.", references) == 0
    assert score_answer("no", ["no"] * 10) == 100
    assert references[:3] == ["two", "2", "two"]


def test_paired_short_answers_are_scored_and_saved(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    image.touch()
    benchmark = {"question_id": 7, "name": "VQA v2", "question_text": "How many?",
                 "reference_answers": ["2"] * 8 + ["3"] * 2}
    sample = dict(benchmark=benchmark, id="test", image=str(image), text_before="", question="Old prompt")
    validate_cases([sample])
    with pytest.raises(ValueError, match="appears more than once"):
        validate_cases([sample, sample])

    def collect(run, *_):
        assert run.question == "How many?\nAnswer with a single word or short phrase."
        assert run.vqa_accuracy
        baseline = run.tv_vv_threshold is None
        return {"answer": "two" if baseline else "three",
                "tv_guided_vv": dict(pure_vv_tiles=4, vv_tiles_bypassed=0 if baseline else 2, layers=[])}

    monkeypatch.setattr(attention_study, "collect", collect)
    output = tmp_path / "report.json"
    args = Namespace(tv_vv_sweep=[.5], model="test", demo=False, lam=.1, decode_lambda=None,
                     sample_index=0, output=output, vqa_accuracy=True, decode_tokens=16)
    attention_study._collect_tv_vv_sweep(args, ArgumentParser(), [sample])
    report = json.loads(output.read_text())
    assert report["answer_protocol"]["min_new_tokens"] == 0
    assert report["summary"]["baseline"]["vqa_accuracy"] == 100
    assert report["summary"]["0.5"]["vqa_accuracy"] == 60
    assert report["summary"]["0.5"]["vqa_delta_pp"] == -40
    assert report["summary"]["0.5"]["vqa_worse"] == 1
    predictions = json.loads((tmp_path / "report-0.5-predictions.json").read_text())
    assert predictions == [{"question_id": 7, "answer": "three"}]
