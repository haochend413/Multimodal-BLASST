"""Select the prompting and grading protocol for a paired answer experiment."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re

import vqav2_accuracy


def validate_vizwiz_cases(samples):
    ids = set()
    for sample in samples:
        benchmark = sample["benchmark"]
        question_id = benchmark.get("question_id")
        refs = benchmark.get("reference_answers")
        if (benchmark.get("name") != "VizWiz-LF" or not isinstance(question_id, int)
                or not benchmark.get("question_text") or not isinstance(refs, list)
                or len(refs) != 1 or not isinstance(refs[0], str) or not refs[0].strip()):
            raise ValueError("VizWiz-LF requires each question's released expert paragraph")
        if question_id in ids:
            raise ValueError(f"VizWiz-LF question ID {question_id} appears more than once")
        ids.add(question_id)


def rouge_l(answer, references):
    """ROUGE-L F1 on a 0–100 scale, using lowercase alphanumeric tokens."""
    prediction = re.findall(r"[a-z0-9]+", answer.lower())
    reference = re.findall(r"[a-z0-9]+", references[0].lower())
    if not prediction or not reference:
        return 0.0
    lengths = [0] * (len(reference) + 1)
    for word in prediction:
        diagonal = 0
        for index, expected in enumerate(reference, start=1):
            above = lengths[index]
            lengths[index] = diagonal + 1 if word == expected else max(above, lengths[index - 1])
            diagonal = above
    return 200.0 * lengths[-1] / (len(prediction) + len(reference))


@dataclass(frozen=True)
class AnswerProtocol:
    name: str
    instruction: str
    metric: str
    case_score: str
    summary_score: str
    prefix: str
    decode_tokens: int
    validate: Callable
    score: Callable[[str, Sequence[str]], float]

    def format_score(self, values):
        score = values[self.summary_score]
        return f"VQA v2 accuracy {score:.2f}%" if self.case_score == "vqa_score" else f"{self.metric} {score:.2f}/100"


VQAV2 = AnswerProtocol(
    "VQA v2", vqav2_accuracy.SHORT_ANSWER_INSTRUCTION, "VQA v2 consensus",
    "vqa_score", "vqa_accuracy", "vqa", 16, vqav2_accuracy.validate_cases, vqav2_accuracy.score_answer,
)
VIZWIZ_LF = AnswerProtocol(
    "VizWiz-LF", "Answer the question in detail using the image. If you cannot tell from the image, explain why.",
    "ROUGE-L F1", "rouge_l", "rouge_l", "rouge_l", 128, validate_vizwiz_cases, rouge_l,
)
ANSWER_BENCHMARKS = {"vqav2": VQAV2, "vizwiz-lf": VIZWIZ_LF}
