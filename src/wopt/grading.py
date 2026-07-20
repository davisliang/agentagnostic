"""Scoring an answer against the gold answer.

A workflow RETURNS its answer, so nothing is parsed out of prose on either side:
`numeric` requires the answer to BE a number, and the judge replies under a
schema. Every check returns a score in [0, 1]; a program's accuracy is the mean
over the dataset.
"""
import importlib.util
import re
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict

from . import prompts
from .llm import LLM


class JudgeScore(BaseModel):
    """The judge replies with a number, not a sentence containing one."""
    model_config = ConfigDict(extra="forbid")
    score: int


def extract_last_number(text) -> Optional[float]:
    """The last number in the text, or None. NOT used by grading — it is handed
    to workflow programs, which may want to pull a value out of a free-text
    intermediate result mid-pipeline."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def as_number(value) -> Optional[float]:
    """A numeric answer must BE a number: "42" parses, "42 apples" does not.
    Searching prose for the last number instead would grade "42 out of 100"
    as 100."""
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


@dataclass
class Checker:
    """Grades one prediction. `kind` picks the rule:

    numeric / exact  — 1.0 or 0.0.
    llm_judge        — a graded quality score from a cheap model against a
                       task-specific rubric, so "accuracy" is mean quality.
    custom           — an external benchmark's own metric (see `from_grader`),
                       which sees the whole item, not just the gold string.

    The judge's own API calls are the evaluator's cost, deliberately NOT counted
    as workflow cost.
    """
    kind: str
    llm: Optional[LLM] = None
    judge_model: str = "claude-haiku-4-5"
    task: str = ""
    rubric: str = ""
    grade_fn: Optional[Callable[[str, dict], float]] = None

    @classmethod
    def from_grader(cls, path: str) -> "Checker":
        """A task that ships its own metric: a .py exposing
        `grade(prediction, item) -> float` in [0, 1]."""
        spec = importlib.util.spec_from_file_location("task_grader", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cls(kind="custom", grade_fn=module.grade)

    def score(self, prediction, item: dict) -> float:
        if self.kind == "custom":
            return self.grade_fn(prediction, item)
        gold = item["answer"]
        if self.kind == "numeric":
            p, g = as_number(prediction), as_number(gold)
            return 1.0 if (p is not None and g is not None and abs(p - g) < 1e-6) else 0.0
        if self.kind == "exact":
            return 1.0 if str(prediction).strip().casefold() == str(gold).strip().casefold() else 0.0
        return self.judge(prediction, gold)

    def judge(self, prediction, gold) -> float:
        """Grade a free-form candidate 0-1 against the rubric. The gold is an
        example of a good answer, not the only acceptable one. Returns 0.0 if the
        judge refuses or its reply doesn't parse."""
        prompt = prompts.render(
            "judge", task=self.task,
            rubric=self.rubric.strip() or
                   "Does the candidate correctly and completely satisfy the task?",
            gold=gold, prediction=prediction)
        try:
            score = self.llm.parse(self.judge_model, prompt, JudgeScore).score
        except ValueError:                          # refusal, or a reply past the ceiling
            return 0.0
        return max(0.0, min(1.0, score / 100.0))    # a schema can't bound a range, so clamp
