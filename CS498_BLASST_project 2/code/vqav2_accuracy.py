"""Apply the official VQA evaluator to unedited predictions on a VQA v2 manifest."""
from collections.abc import Sequence
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

from vqa_eval_upstream import VQAEval

SHORT_ANSWER_INSTRUCTION = "Answer with a single word or short phrase."


def validate_cases(samples):
    ids = set()
    for sample in samples:
        benchmark = sample["benchmark"]
        question_id = benchmark.get("question_id")
        references = benchmark.get("reference_answers")
        if (not isinstance(question_id, int) or benchmark.get("name") != "VQA v2"
                or not benchmark.get("question_text") or not isinstance(references, list)
                or len(references) != 10 or not all(isinstance(answer, str) for answer in references)):
            raise ValueError("--vqa-accuracy needs VQA v2 questions with all ten reference answers")
        if question_id in ids:
            raise ValueError(f"VQA question ID {question_id} appears more than once")
        ids.add(question_id)


def score_answer(answer: str, references: Sequence[str]) -> float:
    """Return the official per-question score in percentage points."""
    # The upstream evaluator normalizes references in place. Use fresh records each time.
    annotations = SimpleNamespace(getQuesIds=lambda: [1], qa={1: {
        "answers": [{"answer_id": index, "answer": value} for index, value in enumerate(references)],
        "question_type": "all", "answer_type": "all",
    }})
    prediction = SimpleNamespace(qa={1: {"answer": answer}})
    evaluator = VQAEval(annotations, prediction, n=10)
    with redirect_stdout(StringIO()):
        evaluator.evaluate()
    return evaluator.evalQA[1]
