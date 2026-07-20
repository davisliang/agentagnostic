"""Step 1 — read the task description, infer how to grade it, and get data.

The analyzer's outputs are Pydantic models rather than hand-written JSON Schema
dicts: one definition both constrains the model's reply and types the object we
read back, so the two can't drift.

NOTE: a Pydantic docstring becomes the JSON Schema "description" and is sent to
the model with the request. Those docstrings are prompt text as well as comments
— keep developer asides in `#` comments, which are not transmitted.
"""
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from . import prompts
from .grading import Checker


class TaskSpec(BaseModel):
    """How to run and grade this task, inferred from the task description."""
    model_config = ConfigDict(extra="forbid")   # -> additionalProperties: false

    description: str                                        # briefs the design agent; grounds the judge
    check_type: Literal["numeric", "exact", "llm_judge"]     # picks the grader
    judge_rubric: str                                        # llm_judge only; "" otherwise
    answer_examples: list[str]                               # validates the rubric; shown to the designer


class LabeledExample(BaseModel):
    """One graded item: the input a workflow sees, and the answer it should return."""
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str


class ExampleBatch(BaseModel):
    """What ONE generation call returns — a batch, not the finished dataset."""
    model_config = ConfigDict(extra="forbid")
    examples: list[LabeledExample]


class CaseTypes(BaseModel):
    """The kinds of case the dataset should cover, planned up front so batches
    don't all reach for the same obvious example. One short phrase per entry, e.g.
    ["division with a remainder", "percentage discount", "rate and distance"] for
    math, or ["discharge summary", "post-op complication"] for clinical notes."""
    model_config = ConfigDict(extra="forbid")
    case_types: list[str]


@dataclass
class Task:
    """Everything downstream needs: what the task is, how to grade it, the data."""
    spec: TaskSpec
    checker: Checker
    dev: list[dict]          # the design agent may tune against this
    test: list[dict]         # held out; only the final ranking touches it
    judge_status: str = ""

    @property
    def description(self) -> str:
        return self.spec.description


def prepare(cfg, llm, log=print) -> Task:
    """Analyze the task, build its checker, and load or generate its dataset."""
    dataset = _load_dataset(cfg.task.dataset)
    spec = analyze_task(cfg, llm, cfg.task.seed_prompt, dataset)

    if cfg.task.grader:
        checker, judge_status = Checker.from_grader(cfg.task.grader), "n/a (custom grader)"
    elif spec.check_type == "llm_judge":
        rubric, judge_status = build_judge(cfg, llm, spec)
        checker = Checker(kind="llm_judge", llm=llm, judge_model=cfg.judge.model,
                          task=spec.description, rubric=rubric)
    else:
        checker, judge_status = Checker(kind=spec.check_type), "n/a (not an LLM judge)"

    if dataset is None:
        dataset = generate_dataset(cfg, llm, spec, log=log)
    split = max(1, int(len(dataset) * cfg.data.dev_fraction))
    task = Task(spec=spec, checker=checker, dev=dataset[:split], test=dataset[split:],
                judge_status=judge_status)

    log(f"check    = {checker.kind}")
    if checker.kind == "llm_judge":
        log(f"judge    = {judge_status}")
    log(f"answers  = {', '.join(repr(e) for e in spec.answer_examples[:3])}")
    log(f"{len(dataset)} examples  ->  {len(task.dev)} dev / {len(task.test)} test")
    return task


def analyze_task(cfg, llm, seed_prompt: str, dataset=None) -> TaskSpec:
    """One structured call: infer the task description, the check, a judge rubric
    for free-form tasks, and what a correctly formatted answer looks like."""
    examples = ""
    if dataset:
        examples = "\n\nExamples (input -> answer):\n" + "".join(
            f"- {item['question']} -> {item['answer']}\n" for item in dataset[:5])
    prompt = prompts.render("analyze_task", seed_prompt=seed_prompt, examples=examples,
                            min_gold=int(cfg.judge.min_gold * 100),
                            max_empty=int(cfg.judge.max_empty * 100))
    return llm.parse(cfg.analysis_model, prompt, TaskSpec)


def build_judge(cfg, llm, spec: TaskSpec) -> tuple[str, str]:
    """Validate the rubric: it must score a gold answer HIGH and an empty answer
    LOW. If it doesn't discriminate, drop it for the generic judge — the numbers
    it produces would otherwise be meaningless."""
    golds = [e for e in spec.answer_examples if str(e).strip()][:2]
    if not golds:
        return spec.judge_rubric, "no golds to validate; using rubric as-is"

    judge = Checker(kind="llm_judge", llm=llm, judge_model=cfg.judge.model,
                    task=spec.description, rubric=spec.judge_rubric)
    high = sum(judge.judge(gold, gold) for gold in golds) / len(golds)   # gold vs itself -> high
    low = sum(judge.judge("", gold) for gold in golds) / len(golds)      # empty answer   -> low

    if high >= cfg.judge.min_gold and low <= cfg.judge.max_empty:
        return spec.judge_rubric, "ok"
    return "", f"rubric didn't discriminate (gold={high:.2f}, empty={low:.2f}); using generic judge"


def generate_dataset(cfg, llm, spec: TaskSpec, log=print) -> list[dict]:
    """Generate labeled examples in SMALL BATCHES — one giant structured-output
    call runs past max_output_tokens and comes back as truncated, invalid JSON.

    Each call is independent (the model has no memory of earlier batches), so
    cross-batch diversity is engineered three ways: point each batch at different
    case types, show it recent inputs to avoid, and dedup on a normalized key so
    paraphrases don't slip through.
    """
    free_form = spec.check_type == "llm_judge"
    answer_rule = (
        "Each 'answer' is an ideal reference output for that input — it may be "
        "multi-sentence / free-form; it will be graded by an LLM judge."
        if free_form else
        "Each 'answer' must be the correct final target ONLY — a bare value (the "
        "number or the label), with no explanation or units.")
    batch_size = cfg.data.judge_batch_size if free_form else cfg.data.batch_size

    case_types = _plan_case_types(cfg, llm, spec)
    log(f"generating ~{cfg.data.n_examples} examples across {len(case_types)} "
        f"case types (batches of {batch_size})...")

    data, seen, stalls, case_i = [], set(), 0, 0
    while len(data) < cfg.data.n_examples and stalls < cfg.data.max_stalls:
        chosen = [case_types[(case_i + j) % len(case_types)] for j in range(3)] if case_types else []
        case_i += 3
        recent = [item["question"][:80].replace("\n", " ") for item in data[-8:]]

        prompt = prompts.render(
            "generate_examples",
            k=min(batch_size, cfg.data.n_examples - len(data)),
            description=spec.description,
            answer_rule=answer_rule,
            case_hint=(" Cover these kinds of case specifically: " + "; ".join(chosen) + "."
                       if chosen else ""),
            avoid_hint=("\n\nMake them DIFFERENT from these already-generated inputs:\n- "
                        + "\n- ".join(recent) if recent else ""))
        try:
            batch = llm.parse(cfg.analysis_model, prompt, ExampleBatch).examples
        except Exception:
            batch = []                    # truncated / garbled batch -> skip, don't crash

        before = len(data)
        for item in batch:
            key = _normalize(item.question)
            if key and key not in seen:
                seen.add(key)
                # plain dicts, so generated and user-supplied data look the same
                data.append({"question": item.question, "answer": item.answer})
        stalls = stalls + 1 if len(data) == before else 0

    if len(data) < cfg.data.n_examples:
        log(f"(note: generated {len(data)}/{cfg.data.n_examples} unique examples)")
    return data[:cfg.data.n_examples]


def _plan_case_types(cfg, llm, spec: TaskSpec) -> list[str]:
    """Ask up front for the kinds of case a good test set spans, so batches can be
    pointed at different ones instead of all generating the "typical" example."""
    prompt = prompts.render("case_types", k=cfg.data.case_types, description=spec.description)
    try:
        return llm.parse(cfg.analysis_model, prompt, CaseTypes).case_types
    except Exception:
        return []


def _normalize(text: str) -> str:
    """Key for near-duplicate detection: lowercase, any run of non-alphanumerics
    collapsed to one space. Catches paraphrases exact-match would miss."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _load_dataset(path):
    if not path:
        return None
    return [json.loads(line) for line in open(path) if line.strip()]
