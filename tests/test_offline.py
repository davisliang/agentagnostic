"""Everything that can be checked without spending money.

The API is faked, so this covers the parts that are pure logic: pricing, grading,
the Pareto helpers, the metered runtime's guardrails, and what the design agent
is handed. Run with `uv run pytest`.
"""
import json
import os
import subprocess
import sys

import pytest

from workflow_optimizer import analysis, prompts
from workflow_optimizer.analysis import Benchmark, TaskAnalysis
from workflow_optimizer.config import load_config, load_resolved
from workflow_optimizer.designer import _round_prompt, _stage_agent_dir, summarize_archive
from workflow_optimizer.grading import Grader, as_number, extract_last_number
from workflow_optimizer.client import ApiResponse
from workflow_optimizer.models import ModelCatalog
from workflow_optimizer.optimizer import DEV, TEST, Candidate
from workflow_optimizer.pareto import best_under_budget, cheapest_above_accuracy, pareto_front
from workflow_optimizer.paths import ROOT, SKILLS_DIR
from workflow_optimizer.runtime import Evaluator, SplitScore, unwrap_answer
from workflow_optimizer.session import Session

SRC = ROOT / "src"

DATA = [{"question": "add 2 and 3", "answer": "5"},
        {"question": "add 10 and 5", "answer": "15"}]


@pytest.fixture
def cfg():
    return load_config("gsm8k")


@pytest.fixture
def catalog(cfg):
    return ModelCatalog.from_config(cfg)


class FakeClient:
    """Answers with the sum of the digits in the prompt, and bills a fixed usage."""

    def __init__(self, catalog):
        self.catalog = catalog

    def call(self, model, prompt, system=None, tools=None, effort=None, schema=None):
        answer = str(sum(int(t) for t in prompt.split() if t.isdigit()))
        return ApiResponse(text=json.dumps({"answer": answer}) if schema else answer,
                    usage={"input": 100, "output": 10, "cache_write": 0, "cache_read": 0})


def evaluator(cfg, catalog, grader=None):
    return Evaluator(FakeClient(catalog), grader or Grader(kind="numeric"), cfg.runtime)


# ---- catalog and pricing ----------------------------------------------------
def usage(**kw):
    return {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, **kw}


def test_catalog_is_ordered_cheapest_first(catalog):
    assert catalog.ids == ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-4-8"]
    assert catalog.default == "claude-haiku-4-5"


def test_unknown_model_falls_back_to_default(catalog):
    # model-written code routes by name and may invent one
    assert catalog.resolve("gpt-9") == catalog.default


def test_cache_changes_the_price(catalog):
    # the same 2,000 haiku input tokens, three ways: fresh, first send, resent
    assert catalog.cost_usd("claude-haiku-4-5", usage(input=2000)) == pytest.approx(0.0020)
    assert catalog.cost_usd("claude-haiku-4-5", usage(cache_write=2000)) == pytest.approx(0.0025)
    assert catalog.cost_usd("claude-haiku-4-5", usage(cache_read=2000)) == pytest.approx(0.0002)


# ---- grading ----------------------------------------------------------------
def test_a_numeric_answer_must_be_the_number():
    assert as_number("42") == 42.0
    assert as_number(" 1,024 ") == 1024.0          # thousands separators are fine
    assert as_number("42 apples") is None          # an answer must BE the number
    check = Grader(kind="numeric")
    assert check.score("42", {"answer": 42}) == 1.0
    assert check.score("42 apples", {"answer": "42"}) == 0.0


def test_exact_check_ignores_case():
    check = Grader(kind="exact")
    assert check.score("Positive", {"answer": "positive"}) == 1.0
    assert check.score("cat", {"answer": "dog"}) == 0.0


def test_extract_last_number_is_for_programs_not_grading():
    assert extract_last_number("Sold 75 of 84 apples, so 9 remain.") == 9.0
    assert extract_last_number("none here") is None


# ---- the answer contract ----------------------------------------------------
def test_unwrap_answer_reduces_what_solve_returned():
    assert unwrap_answer("42") == "42"
    assert unwrap_answer({"answer": " 42 ", "note": "kept"}) == "42"


def test_an_object_with_no_answer_key_is_a_contract_violation():
    # stringifying it would silently grade "{'result': 'positive'}" and score 0
    with pytest.raises(ValueError):
        unwrap_answer({"result": "positive"})


# ---- pareto -----------------------------------------------------------------
def results():
    return [SplitScore("cheap", 0.70, 0.0002), SplitScore("cot", 0.90, 0.0010),
            SplitScore("sc", 0.92, 0.0050), SplitScore("dominated", 0.80, 0.0020)]


def test_pareto_front_drops_dominated_results():
    # "dominated" goes: "cot" is both cheaper AND more accurate
    assert [r.name for r in pareto_front(results())] == ["cheap", "cot", "sc"]


def test_the_two_constrained_picks():
    assert best_under_budget(results(), 0.0015).name == "cot"
    assert cheapest_above_accuracy(results(), 0.90).name == "cot"
    assert best_under_budget(results(), 0.0) is None
    assert cheapest_above_accuracy(results(), 1.0) is None


def test_the_split_being_compared_on_is_the_callers_choice():
    # same candidates, opposite verdicts — which is exactly why the caller names
    # the split instead of the object picking one
    candidates = [
        Candidate("A", "", "a", dev=SplitScore("A", 0.9, 0.001), test=SplitScore("A", 0.5, 0.001)),
        Candidate("B", "", "b", dev=SplitScore("B", 0.5, 0.002), test=SplitScore("B", 0.9, 0.002)),
    ]
    assert [c.name for c in pareto_front(candidates, on=DEV)] == ["A"]
    assert [c.name for c in pareto_front(candidates, on=TEST)] == ["A", "B"]
    assert best_under_budget(candidates, 0.005, on=DEV).name == "A"
    assert best_under_budget(candidates, 0.005, on=TEST).name == "B"


# ---- the metered runtime ----------------------------------------------------
GOOD = {"name": "H", "code": (
    "def solve(question, call_model):\n"
    "    reply = call_model(question, schema=ANSWER_SCHEMA)\n"
    "    return reply.data['answer'] if reply.data else str(reply).strip()\n")}


def test_a_working_candidate_is_scored_and_metered(cfg, catalog):
    result = evaluator(cfg, catalog).run(GOOD, DATA)
    assert result.accuracy == 1.0
    assert result.cost > 0
    assert not result.errors
    assert len(result.records[0]["calls"]) == 1     # the call trace is kept


def test_the_sandbox_blocks_imports_outside_the_allowlist(cfg, catalog):
    sneaky = {"name": "bad", "code": "import os\ndef solve(q, c):\n    return os.getcwd()\n"}
    assert "compile" in evaluator(cfg, catalog).run(sneaky, DATA).error


def test_a_crash_scores_zero_instead_of_sinking_the_run(cfg, catalog):
    crasher = {"name": "boom", "code": "def solve(q, c):\n    raise ValueError('x')\n"}
    result = evaluator(cfg, catalog).run(crasher, DATA)
    assert result.accuracy == 0.0
    assert len(result.errors) == len(DATA)


def test_a_runaway_program_hits_its_call_budget(cfg, catalog):
    runaway = {"name": "loop", "code": (
        "def solve(q, call_model):\n"
        "    for _ in range(100):\n        call_model(q)\n    return '0'\n")}
    assert "budget" in evaluator(cfg, catalog).run(runaway, DATA).errors[0]


def test_a_program_without_solve_is_rejected(cfg, catalog):
    assert "solve" in evaluator(cfg, catalog).run({"name": "x", "code": "y = 1\n"}, DATA).error


# The sandbox has to admit ordinary Python. A name it refuses doesn't read as
# "blocked" in the results — it reads as "this strategy scores 0", and the search
# then avoids a whole family of workflow for a reason nothing reports.
ORDINARY = {
    "sampling": "import random\ndef solve(q, m):\n    return str(random.choice([5, 15]))\n",
    "a helper class": ("class V:\n    def __init__(self, x):\n        self.x = x\n"
                       "def solve(q, m):\n    return V('5').x\n"),
    "getattr on a reply": ("def solve(q, call_model):\n"
                           "    r = call_model(q, schema=ANSWER_SCHEMA)\n"
                           "    return (getattr(r, 'data', None) or {}).get('answer', '5')\n"),
    "catching its own budget error": ("def solve(q, call_model):\n"
                                      "    try:\n        return str(call_model(q))\n"
                                      "    except RuntimeError:\n        return '0'\n"),
    "typing hints": "from typing import Optional\ndef solve(q, m) -> Optional[str]:\n    return '5'\n",
}


@pytest.mark.parametrize("what", list(ORDINARY))
def test_the_sandbox_admits_ordinary_python(cfg, catalog, what):
    result = evaluator(cfg, catalog).run({"name": what, "code": ORDINARY[what]}, DATA)
    assert not result.error, f"{what}: {result.error}"
    assert not result.errors, f"{what}: {result.errors}"


def test_an_empty_dataset_raises_instead_of_scoring_zero(cfg, catalog):
    # accuracy 0.00 / $0.00000 is indistinguishable from a real measurement
    with pytest.raises(ValueError):
        evaluator(cfg, catalog).run(GOOD, [])


@pytest.mark.parametrize("reach_out", [
    "def solve(q, m):\n    return open('/etc/passwd').read()\n",
    "def solve(q, m):\n    return eval('1+1')\n",
    "import subprocess\ndef solve(q, m):\n    return '5'\n",
])
def test_the_sandbox_still_blocks_reaching_outside(cfg, catalog, reach_out):
    result = evaluator(cfg, catalog).run({"name": "x", "code": reach_out}, DATA)
    assert result.error or result.errors     # at compile time or at call time


def test_the_eval_skill_always_emits_one_json_line(tmp_path):
    # the agent parses this; a traceback would leave it guessing
    script = SKILLS_DIR / "workflow-eval" / "eval_candidate.py"
    for args in ([], ["nope.py"]):
        done = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path,
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(SRC)})
        assert json.loads(done.stdout.strip())["ok"] is False, done.stdout + done.stderr


# ---- config -----------------------------------------------------------------
def test_overrides_win_over_the_task_file():
    cfg = load_config("gsm8k", ["designer.rounds=1", "runtime.concurrency=2"])
    assert (cfg.designer.rounds, cfg.runtime.concurrency) == (1, 2)
    assert cfg.task.name == "gsm8k"                 # untouched keys survive


def test_an_unknown_key_is_rejected():
    with pytest.raises(Exception):
        load_config("gsm8k", ["designer.rouds=1"])


def test_a_session_wires_config_catalog_and_client_together(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    session = Session.load("gsm8k", ["designer.rounds=1"])
    assert session.cfg.designer.rounds == 1
    assert session.catalog.ids[0] == "claude-haiku-4-5"
    assert session.client.catalog is session.catalog          # one catalog, not two
    assert session.evaluator(Grader(kind="numeric")).default_model == session.catalog.default


# ---- prompts ----------------------------------------------------------------
def test_prompts_are_filled_from_files():
    text = prompts.render("judge", task="t", rubric="r", gold="g", prediction="p")
    assert "Task: t" in text and "$" not in text


def test_a_missing_prompt_variable_is_an_error_not_a_blank():
    with pytest.raises(KeyError):
        prompts.render("judge", task="t")


# ---- what the design agent is handed ----------------------------------------
def benchmark_fixture():
    analysis = TaskAnalysis(description="Add two numbers.", check_type="numeric",
                            judge_rubric="", answer_examples=["5", "15"])
    return Benchmark(analysis=analysis, grader=Grader(kind="numeric"), dev=DATA, test=DATA)


def test_the_agent_dir_has_everything_the_eval_skill_reads(cfg, tmp_path):
    _stage_agent_dir(cfg, benchmark_fixture(), tmp_path)
    assert json.loads((tmp_path / "task_spec.json").read_text())["check"]["type"] == "numeric"
    assert json.loads((tmp_path / "dev_task.json").read_text()) == DATA
    for skill in cfg.designer.skills:
        assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").exists()
    # the staged config round-trips, so the agent meters candidates as the search does
    assert load_resolved(tmp_path / "run_config.yaml").runtime.max_model_calls == cfg.runtime.max_model_calls


def test_round_one_asks_for_diversity_and_later_rounds_ask_for_cheaper(cfg):
    benchmark = benchmark_fixture()
    first = _round_prompt(cfg, benchmark, 1, "")
    assert "DIVERSE" in first and "claude-haiku-4-5" in first
    assert "must BE the number" in first          # the strict checker is spelled out

    archive = [Candidate("H", "", "code-a", dev=SplitScore("H", 0.8, 0.001))]
    later = _round_prompt(cfg, benchmark, 2, summarize_archive(archive))
    assert "cost LESS per query" in later and "code-a" in later


def test_the_archive_summary_shows_the_frontier_and_the_best_code():
    archive = [Candidate("H", "", "cheap-code", dev=SplitScore("H", 0.70, 0.0002)),
               Candidate("S", "", "good-code", dev=SplitScore("S", 0.90, 0.0010)),
               Candidate("D", "", "dud-code", dev=SplitScore("D", 0.60, 0.0050))]
    summary = summarize_archive(archive)
    assert "dud-code" not in summary              # dominated: not worth the agent's context
    assert "good-code" in summary                 # the most accurate, as a base to improve on


# ---- a broken grader must not read as a run of zeros ------------------------
def test_a_grader_that_cannot_read_the_data_fails_before_the_search(cfg):
    """A grader that raises is caught per example and scored 0.0, so a broken
    grader looks exactly like a workflow that answers wrongly. This once cost a
    full search — 9 candidates over 120 examples, every score 0 — to discover
    that the dataset lacked the `doc` field its grader reads."""
    def needs_doc(prediction, item):
        return float(item["doc"]["ok"])          # the field the data hasn't got

    grader = Grader(kind="custom", grade_fn=needs_doc)
    with pytest.raises(ValueError) as raised:
        analysis.check_grader(grader, {"question": "q", "answer": "a"})
    message = str(raised.value)
    assert "cannot read this dataset" in message
    assert "KeyError" in message
    assert "['answer', 'question']" in message   # says what the data actually has


def test_a_working_grader_passes_the_check():
    analysis.check_grader(Grader(kind="exact"), {"question": "q", "answer": "a"})
    analysis.check_grader(Grader(kind="numeric"), {"question": "q", "answer": "42"})


def test_the_judge_is_not_probed(monkeypatch):
    # probing a judge costs an API call, and it fails by scoring low, not raising
    judge = Grader(kind="llm_judge", client=None)
    analysis.check_grader(judge, {"question": "q", "answer": "a"})     # must not call out


# ---- n_examples has to mean the same thing for loaded and generated data -----
def test_n_examples_caps_a_loaded_dataset():
    """Asking for 40 against a 200-row benchmark used to run all 200 — and the
    cost estimate, which sizes itself from n_examples, understated it 5x."""
    from workflow_optimizer import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(200)]
    taken = datasets.take(data, 40, log=lambda *a: None)
    assert len(taken) == 40
    assert all(item in data for item in taken)


def test_taking_is_deterministic_so_two_runs_are_comparable():
    from workflow_optimizer import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(200)]
    first = datasets.take(data, 40, log=lambda *a: None)
    second = datasets.take(data, 40, log=lambda *a: None)
    assert [d["question"] for d in first] == [d["question"] for d in second]


def test_asking_for_more_than_exists_keeps_everything():
    from workflow_optimizer import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(10)]
    assert len(datasets.take(data, 40, log=lambda *a: None)) == 10
    assert len(datasets.take(data, 0, log=lambda *a: None)) == 10      # 0 means all
