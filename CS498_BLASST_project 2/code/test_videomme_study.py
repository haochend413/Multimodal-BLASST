# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark identity, answer leakage and subtitle-layout checks without model weights."""
from pathlib import Path

import pytest

from prepare_videomme_study import prepare_samples


def test_sampling_and_prompt_identity(tmp_path: Path) -> None:
    rows = []
    media = {}
    for duration in ("short", "medium", "long"):
        for video in range(3):
            video_id = f"{duration}_{video}"
            media[video_id] = tmp_path / f"{video_id}.mp4"
            for question in range(3):
                rows.append(dict(videoID=video_id, question_id=f"{video_id}-{question}",
                                 question="Which action?", options=["A. Jump", "B. Run", "C. Sit", "D. Sleep"],
                                 answer="C", duration=duration, domain="Sports", task_type="Action"))
    kwargs = dict(layout="C", durations=["short", "medium", "long"], videos_per_duration=1,
                  questions_per_video=2, seed=98, subtitle_chars=1000)
    selected = prepare_samples(rows, media, {}, **kwargs)
    assert selected == prepare_samples(list(reversed(rows)), media, {}, **kwargs)
    assert len(selected) == 6
    assert len({r["benchmark"]["video_id"] for r in selected}) == 3
    for sample in selected:
        assert sample["text_before"] == ""
        assert sample["question"].endswith("The best answer is:")
        assert "C. Sit" in sample["question"]
        assert sample["benchmark"]["ground_truth"] == "C"
        assert "correct answer is C" not in sample["question"]
    with pytest.raises(ValueError, match="available short"):
        prepare_samples(rows, {}, {}, **kwargs)


@pytest.mark.parametrize("layout", ["A", "B"])
def test_subtitle_layout_is_explicit(tmp_path: Path, layout: str) -> None:
    subtitle = tmp_path / "video.srt"
    subtitle.write_text("1\n00:00:00,000 --> 00:00:02,000\nFirst event.\n\n2\n"
                        "00:00:02,000 --> 00:00:03,000\nSecond event.\n")
    row = dict(videoID="video", question_id="001-1", question="Which action?",
               options=["A", "B", "C", "D"], answer="A", duration="short", domain="Life", task_type="Action")
    sample = prepare_samples([row], {"video": tmp_path / "video.mp4"}, {"video": subtitle},
                             layout=layout, durations=["short"], videos_per_duration=1,
                             questions_per_video=1, seed=98, subtitle_chars=4000)[0]
    assert "not official" in sample["benchmark"]["protocol"]
    assert bool(sample["text_before"]) == (layout == "A")
    assert "-->" not in sample["question"] + sample["text_before"]
    assert "First event." in sample["question"] + sample["text_before"]
