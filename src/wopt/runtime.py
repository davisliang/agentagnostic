"""Running a candidate workflow program, metering and capping every model call.

A workflow is a Python program defining `solve(question, call_model) -> answer`.
Because it is code, it can express any inference-time paradigm; the harness
fixes only the contract, the metered call site, and the grader.
"""
import builtins
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict

from .grading import extract_last_number


class Reply(str):
    """What `call_model` returns: the reply text, with the whole response attached.

    It IS a string, so `return call_model(prompt)` works — but nothing is thrown
    away. `.blocks` holds every content block the call produced (tool calls, tool
    results, text), `.data` is the parsed JSON when the call passed a schema, and
    `.usage` / `.model` say what it cost.
    """
    def __new__(cls, text, blocks=(), usage=None, model="", data=None):
        reply = super().__new__(cls, text)
        reply.blocks = list(blocks)
        reply.usage = dict(usage or {})
        reply.model = model
        reply.data = data
        return reply


@dataclass
class Turn:
    """One metered model call, kept whole so the trace loses nothing."""
    model: str
    prompt: str
    reply: Reply
    cost: float


class Answer(BaseModel):
    """The shape of a final answer, handed to every program so the graded
    contract and the schema it asks the model for cannot drift apart."""
    model_config = ConfigDict(extra="forbid")
    answer: str


# Programs get it as a plain dict: model-written code runs in a sandbox with no
# pydantic, and call_model's schema= accepts either form.
ANSWER_SCHEMA = Answer.model_json_schema()


class Budget:
    """The per-query object a program calls for every model call.

    This is the one place cost is measured, and the one place a runaway program
    is stopped. Every call is recorded in `.turns`, so a run can be inspected
    afterwards.
    """
    def __init__(self, llm, catalog, default_model, max_calls, token_budget):
        self.llm = llm
        self.catalog = catalog
        self.default_model = default_model
        self.max_calls = max_calls
        self.token_budget = token_budget
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0
        self.turns: list[Turn] = []

    def call_model(self, prompt, model=None, system=None, tools=None,
                   effort=None, schema=None) -> Reply:
        if self.calls >= self.max_calls:
            raise RuntimeError("workflow exceeded its model-call budget")
        self.calls += 1
        # An unknown model name falls back to the default: model-written code
        # routes by name and may invent one.
        model = self.catalog.resolve(model) if model else self.default_model

        call = self.llm.call(model, str(prompt), system=system, tools=tools,
                             effort=effort, schema=schema)
        data = None
        if schema:
            try:
                data = json.loads(call.text)    # the reply is constrained to the schema
            except ValueError:
                data = None                     # a refusal, or a reply past the ceiling

        cost = self.catalog.cost_usd(model, call.usage)
        self.tokens += sum(call.usage.values())
        self.cost += cost

        reply = Reply(call.text, blocks=call.blocks, usage=call.usage, model=model, data=data)
        self.turns.append(Turn(model=model, prompt=str(prompt), reply=reply, cost=cost))

        if self.tokens > self.token_budget:
            raise RuntimeError("workflow exceeded its token budget")
        return reply


# ---- Compiling a candidate --------------------------------------------------
# Candidate code is model-written, so by default it runs with a restricted import
# list and a small builtins allowlist rather than full Python. Set
# runtime.sandbox=false for a plain exec (faster to debug, no guardrail).
_ALLOWED_IMPORTS = {"re", "json", "math", "statistics", "collections", "itertools",
                    "functools", "string"}
_ALLOWED_BUILTINS = (
    "range len min max sum sorted abs round divmod pow enumerate zip map filter "
    "list dict set tuple str int float bool any all isinstance reversed print "
    "Exception ValueError KeyError TypeError AttributeError ZeroDivisionError")


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"blocked import in candidate: {name}")
    return builtins.__import__(name, *args, **kwargs)


def compile_solve(code: str, catalog, sandbox: bool = True):
    """Run a program's source and return its `solve`. The program may use the
    names below without importing them (see the workflow-design skill)."""
    namespace = {
        "re": re, "json": json, "statistics": statistics, "Counter": Counter,
        "extract_last_number": extract_last_number,
        "MODELS": catalog.ids, "ANSWER_SCHEMA": ANSWER_SCHEMA,
    }
    if sandbox:
        allowed = {n: getattr(builtins, n) for n in _ALLOWED_BUILTINS.split()}
        allowed["__import__"] = _guarded_import
        namespace["__builtins__"] = allowed
    exec(code, namespace)
    if not callable(namespace.get("solve")):
        raise ValueError("program does not define solve(question, call_model)")
    return namespace["solve"]


def final_answer(returned) -> str:
    """What `solve()` handed back, reduced to the string we grade.

    A program returns its answer directly, a dict carrying the answer plus
    anything else it wants to keep, or a schema-constrained `Reply` as-is — all
    three unwrap to the answer. A structure with no "answer" in it is a contract
    violation, not an answer: stringifying it would grade `{'result': 'positive'}`
    as the prediction and silently score 0, so raise and let the record show why.
    """
    if isinstance(returned, Reply) and returned.data is not None:
        returned = returned.data
    if isinstance(returned, dict):
        if "answer" not in returned:
            raise ValueError(f"solve() returned an object with no 'answer' key: {sorted(returned)}")
        returned = returned["answer"]
    return str(returned).strip()


# ---- Evaluating a candidate over a dataset ----------------------------------
@dataclass
class Result:
    """How one candidate did on one split."""
    name: str
    accuracy: float = 0.0            # mean score in [0, 1]
    cost: float = 0.0                # mean USD per query
    cached_input_frac: float = 0.0
    records: list = field(default_factory=list)   # per example; kept, never graded
    error: str = ""                  # set only if the program didn't compile

    @property
    def errors(self) -> list[str]:
        return [r["error"] for r in self.records if r["error"]]


class Evaluator:
    """Runs candidates against a dataset: the same metered runtime and the same
    checker for dev, test, and the design agent's own self-tests."""

    def __init__(self, llm, catalog, checker, runtime_cfg, default_model=None):
        self.llm = llm
        self.catalog = catalog
        self.checker = checker
        self.cfg = runtime_cfg
        self.default_model = default_model or catalog.default

    def run(self, program: dict, dataset: list[dict]) -> Result:
        """Run the program once per example: grade the answer it returns, tally
        what its model calls cost, and keep the full trace of every call. A
        program that crashes or blows its budget scores 0 on that example rather
        than sinking the whole evaluation."""
        if not dataset:
            return Result(name=program["name"], error="empty dataset")
        try:
            solve = compile_solve(program["code"], self.catalog, self.cfg.sandbox)
        except Exception as error:
            return Result(name=program["name"], error=f"compile: {error}")

        workers = min(self.cfg.workers, len(dataset))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(lambda item: self._run_one(solve, item), dataset))
        else:
            records = [self._run_one(solve, item) for item in dataset]

        return Result(
            name=program["name"],
            accuracy=statistics.mean(r["score"] for r in records),
            cost=statistics.mean(r["cost"] for r in records),
            cached_input_frac=_cached_input_frac(records),
            records=records,
        )

    def _run_one(self, solve, item: dict) -> dict:
        # Each example gets its own Budget, so examples share nothing and can run
        # concurrently — the wall clock here is otherwise sequential API latency,
        # which dominates everything else in the loop.
        budget = Budget(self.llm, self.catalog, self.default_model,
                        self.cfg.max_calls, self.cfg.token_budget)
        answer, score, error = "", 0.0, None
        try:
            answer = final_answer(solve(item["question"], budget.call_model))
            score = self.checker.score(answer, item)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        return {"question": item["question"], "gold": item.get("answer", ""),
                "answer": answer, "score": score, "cost": budget.cost,
                "error": error, "turns": budget.turns}


def _cached_input_frac(records: list[dict]) -> float:
    """What share of input tokens came from cache.

    Prompt caching silently does nothing below a per-model size floor (~4k tokens
    on haiku/opus, ~1k on sonnet), so a workflow built on "resending is cheap" can
    be paying full price with no error to say so. Surfaced here rather than left
    to be discovered from the bill.
    """
    fresh = cached = 0
    for record in records:
        for turn in record["turns"]:
            fresh += turn.reply.usage["input"] + turn.reply.usage["cache_write"]
            cached += turn.reply.usage["cache_read"]
    total = fresh + cached
    return cached / total if total else 0.0
