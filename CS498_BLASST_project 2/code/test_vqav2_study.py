# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VQA image/question identity, answer separation and visual-token recording checks."""
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import blasst_attn as ba
from attention_study import AttentionRecorder, visual_token_mask
from prepare_vqav2_study import select_samples


def test_vqa_selection_and_answer_separation(tmp_path: Path) -> None:
    questions = [{"image_id": image, "question_id": image * 10 + index,
                  "question": "What is on the table?"}
                 for image in (42, 73) for index in range(3)]
    answers = [{"image_id": q["image_id"], "question_id": q["question_id"],
                "question_type": "what", "answer_type": "other",
                "multiple_choice_answer": "teapot",
                "answers": [{"answer": "teapot"} for _ in range(10)]} for q in questions]
    options = dict(images=2, questions_per_image=3, seed=98)
    samples = select_samples(questions, answers, tmp_path, **options)
    assert len(samples) == 6
    assert samples == select_samples(questions[::-1], answers[::-1], tmp_path, **options)
    for sample in samples:
        assert sample["text_before"] == ""
        assert "teapot" not in sample["question"]
        assert sample["benchmark"]["reference_answers"] == ["teapot"] * 10
        assert str(sample["benchmark"]["image_id"]).zfill(12) in sample["image"]
    chosen = select_samples(questions, answers, tmp_path, images=1, questions_per_image=1,
                            seed=98, image_id=42)
    assert chosen[0]["benchmark"]["question_id"] == 420
    question_sample = dict(images=1, questions_per_image=1, sample_questions=4, seed=98)
    random_questions = select_samples(questions, answers, tmp_path, **question_sample)
    assert random_questions == select_samples(questions[::-1], answers[::-1], tmp_path, **question_sample)
    assert len({sample["benchmark"]["question_id"] for sample in random_questions}) == 4
    with pytest.raises(ValueError, match="cannot be combined"):
        select_samples(questions, answers, tmp_path, **question_sample, image_id=42)
    with pytest.raises(ValueError, match="not in VQA"):
        select_samples(questions, answers, tmp_path, images=1, questions_per_image=1,
                       seed=98, image_id=999)
    with pytest.raises(ValueError, match="ten reference"):
        select_samples(questions, [{**a, "answers": []} for a in answers], tmp_path,
                       images=1, questions_per_image=1, seed=98)


def test_image_regions_and_score_companion(tmp_path: Path) -> None:
    # Include both image/video placeholders and vision control tokens in a mixed tile.
    ids = torch.tensor([151644, 151652] + [151655] * 70 + [151656] * 60 + [151653, 42])
    visual = visual_token_mask(ids, image_token_id=151655, video_token_id=151656)
    assert int(visual.sum()) == 130
    assert not bool(visual[0]) and not bool(visual[-2])
    stream = io.BytesIO()
    recorder = AttentionRecorder(visual, {0}, {0}, score_stream=stream)
    old_state = dict(ba.STATE)
    try:
        ba.set_threshold(None)
        ba.set_decode_threshold(inherit=True)
        ba.reset_records(record=False)
        ba.STATE["observer"] = recorder.observe
        q = torch.ones(1, 1, len(ids), 8)
        ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q, q, q, None)
        keys = torch.ones(1, 1, len(ids) + 1, 8)
        ba.blasst_attention_forward(SimpleNamespace(layer_idx=0), q[:, :, :1], keys, keys, None)
    finally:
        ba.STATE.clear()
        ba.STATE.update(old_state)
    output = tmp_path / "image-attention.json"
    recorder.export(output, {"model": "test", "synthetic": True, "modality": "image"})
    payload = json.loads(output.read_text())
    assert payload["schema"] == "blasst-attention-v1"
    assert payload["score_file"]["bytes"] == len(stream.getvalue())
    assert stream.getvalue()[:32].decode() == payload["score_file"]["id"]
    prefill, decode = payload["views"]
    assert sum(s["count"] for s in prefill["regions"].values()) == len(ids) * (len(ids) + 1) // 2
    assert decode["regions"]["T→V"]["count"] == 130
    assert decode["regions"]["V→V"]["count"] == 0
