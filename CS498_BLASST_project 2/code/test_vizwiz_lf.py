"""Exercise the long-answer manifest protocol and paired report."""
import json
import sys

import pytest

import attention_study
from answer_benchmarks import VIZWIZ_LF, rouge_l


def test_rouge_l_uses_word_order():
    assert rouge_l("The light is red.", ["The light is red."]) == 100
    assert rouge_l("", ["The light is red."]) == 0
    assert rouge_l("RED light", ["light red"]) == 50


def test_vizwiz_lf_cli_uses_all_questions_and_long_answer_scoring(tmp_path, monkeypatch):
    image = tmp_path / "image.jpg"
    image.touch()
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "report.json"
    refs = ["There is a red light ahead."]
    samples = [
        dict(id=str(question_id), image=str(image), text_before="", question="What is shown?",
             benchmark=dict(name="VizWiz-LF", question_id=question_id,
                            question_text="What is shown?", reference_answers=refs))
        for question_id in range(2)
    ]
    manifest.write_text(json.dumps(dict(schema="blasst-benchmark-manifest-v1", samples=samples)))
    monkeypatch.setattr(sys, "argv", ["attention_study.py", "--benchmark", "vizwiz-lf",
                                    "--manifest", str(manifest), "--decode-visual-links", "--output", str(output)])
    monkeypatch.setattr(attention_study.torch.cuda, "is_available", lambda: True)

    def collect(args, *_):
        assert args.decode_tokens == 128
        assert args.question == f"What is shown?\n{VIZWIZ_LF.instruction}"
        answer = "red light" if args.decode_visual_links else refs[0]
        return dict(answer=answer, generated_tokens=12, decode_visual_links={})

    monkeypatch.setattr(attention_study, "collect", collect)
    attention_study.main()
    report = json.loads(output.read_text())
    assert report["benchmark"] == "VizWiz-LF"
    assert report["answer_metric"] == "ROUGE-L F1"
    assert len(report["cases"]) == 2
    assert report["summary"]["baseline"]["rouge_l"] == 100
    guided = report["summary"]["decode_links"]
    assert guided["rouge_l_delta_pp"] == pytest.approx(-50)
    assert guided["mean_generated_tokens"] == 12
    assert guided["mean_decode_steps"] == 11
    assert "vqa_accuracy" not in guided
