#!/usr/bin/env python3
"""Score a candidate solve(question, call_model) program on the dev set.

Task-agnostic: rebuilds the task's checker from task_spec.json (numeric /
exact-match / llm_judge). The grading code below is kept identical to the
notebook's, so dev and test grade the same way. A program RETURNS its answer,
so there is nothing to extract. Reads task_spec.json and dev_task.json from
the current working directory.
Usage: python eval_candidate.py <candidate.py>
"""
import re, sys, json, statistics, builtins, signal
from collections import Counter
from dataclasses import dataclass
import anthropic
from pydantic import BaseModel, ConfigDict

@dataclass
class Model:
    id: str
    price_in: float       # USD per 1,000,000 input tokens
    price_out: float      # USD per 1,000,000 output tokens
    thinks: bool          # supports the effort / adaptive-thinking params

MODEL_SPECS = [
    Model("claude-haiku-4-5", 1.0,  5.0, thinks=False),
    Model("claude-sonnet-5",  3.0, 15.0, thinks=True),
    Model("claude-opus-4-8",  5.0, 25.0, thinks=True),
]
BY_ID = {m.id: m for m in MODEL_SPECS}      # id -> Model
MODELS = [m.id for m in MODEL_SPECS]        # ids, cheapest -> most expensive
DEFAULT_MODEL = MODELS[0]
client = anthropic.Anthropic()

CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.10

def cost_usd(m, usage):
    # cache writes bill 1.25x the input rate, reads only 0.10x (~90% off)
    spec = BY_ID[m]
    return (usage["input"] * spec.price_in
            + usage["cache_write"] * spec.price_in * CACHE_WRITE_MULT
            + usage["cache_read"] * spec.price_in * CACHE_READ_MULT
            + usage["output"] * spec.price_out) / 1_000_000

TOOL_DEFS = {
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
    # allowed_callers=["direct"] is REQUIRED for the cheap model: the _20260209 web
    # tools default to being called from inside code execution, and haiku can't do
    # programmatic tool calling — without this every web_search call 400s.
    "web_search":     {"type": "web_search_20260209", "name": "web_search",
                       "allowed_callers": ["direct"]},
}

MAX_TOOL_TURNS = 5   # cap on API calls while a server-side tool keeps pausing the turn

# One output ceiling for every call, set high enough to never be the reason an
# answer is short. NOT a cost knob — you are billed for the tokens a reply actually
# uses. It must be generous because THINKING COUNTS AGAINST IT: at effort="max" a
# 16k ceiling burns the whole budget on thinking and returns an empty answer.
MAX_OUTPUT_TOKENS = 64000

def _get_usage(msg):
    # Anthropic returns these token counts in the response's usage block.
    return {
        "input": msg.usage.input_tokens,
        "output": msg.usage.output_tokens,
        "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
    }

def _call_api(model, prompt, system=None, tools=None, effort=None, schema=None):
    # cache breakpoint on the prompt -> identical resends to this model are cheap
    content = [{"type": "text", "text": str(prompt),
                "cache_control": {"type": "ephemeral"}}]
    request = {"model": model, "max_tokens": MAX_OUTPUT_TOKENS,
               "messages": [{"role": "user", "content": content}]}
    if system:
        request["system"] = [{"type": "text", "text": system,
                              "cache_control": {"type": "ephemeral"}}]
    if tools:
        request["tools"] = []
        for name in tools:
            request["tools"].append(TOOL_DEFS[name])
    output_config = {}
    if effort and model in BY_ID and BY_ID[model].thinks:
        request["thinking"] = {"type": "adaptive"}
        output_config["effort"] = effort
    else:
        request["thinking"] = {"type": "disabled"}
    if output_config:
        request["output_config"] = output_config
    if schema is not None:
        # Constrains ONLY the text the model writes at the end — tool calls and
        # tool results in the same reply are untouched. Takes a Pydantic model
        # class (what this notebook uses) or a raw JSON Schema dict (what a
        # workflow program passes, since the sandbox has no pydantic).
        request["output_format"] = (
            schema if isinstance(schema, type) and issubclass(schema, BaseModel)
            else {"type": "json_schema", "schema": schema})
    turn_texts = []
    blocks = []
    usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    turns = 0
    while turns < MAX_TOOL_TURNS:
        turns += 1
        # Streamed because MAX_OUTPUT_TOKENS is large: the SDK refuses a
        # non-streaming request whose ceiling could outlive the HTTP timeout.
        with client.messages.stream(**request) as stream:
            msg = stream.get_final_message()
        got = _get_usage(msg)
        usage["input"] += got["input"]
        usage["output"] += got["output"]
        usage["cache_write"] += got["cache_write"]
        usage["cache_read"] += got["cache_read"]
        blocks.extend(msg.content)
        # Join text blocks WITHIN a response (citations split one message into
        # several); keep responses apart — the LAST one holds the answer.
        turn_texts.append("".join(b.text for b in msg.content if b.type == "text"))
        if msg.stop_reason != "pause_turn":
            break
        request["messages"].append({"role": "assistant", "content": msg.content})
    return (turn_texts[-1] if turn_texts else ""), blocks, usage

# ---- Grading an answer (mirrors the notebook) -------------------------------
# CHECK_TYPE / JUDGE_RUBRIC / JUDGE_TASK come from task_spec.json, set in main().
JUDGE_MODEL = "claude-haiku-4-5"
JUDGE_RUBRIC = ""
JUDGE_TASK = ""

class JudgeScore(BaseModel):
    """The judge replies with a number, not a sentence containing one."""
    model_config = ConfigDict(extra="forbid")
    score: int

def extract_last_number(text):
    # The last number in the text, or None. NOT used by grading — it is handed to
    # workflow programs, which may still want to pull a value out of a free-text
    # intermediate result mid-pipeline.
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None

def as_number(value):
    # A numeric answer must BE a number. A program returns its answer, so there is
    # nothing to search for: "42" parses, "42 apples" does not. Searching prose for
    # the last number instead would grade "42 out of 100" as 100.
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None

def judge_score(prediction, gold, rubric="", task=""):
    # Grade a free-form candidate from 0 to 1 against the task rubric (the gold is
    # an example of a good answer, not the only acceptable one). 0.0 if the judge
    # refuses or its reply doesn't parse.
    criteria = rubric.strip() or "Does the candidate correctly and completely satisfy the task?"
    prompt = (f"Task: {task}\n\nGrading rubric:\n{criteria}\n\n"
              f"Reference (an example of a good answer):\n{gold}\n\n"
              f"Candidate answer:\n{prediction}\n\n"
              "Score how well the candidate satisfies the rubric for the task, from 0 to 100 "
              "(100 = fully correct/complete, 0 = wrong or empty).")
    reply, _, _ = _call_api(JUDGE_MODEL, prompt, schema=JudgeScore)
    try:
        score = JudgeScore.model_validate_json(reply).score
    except ValueError:                         # refusal, or a reply past the ceiling
        return 0.0
    return max(0.0, min(1.0, score / 100.0))   # a schema can't bound a range, so clamp

def check_answer(prediction, gold, check_type):
    if check_type == "numeric":
        p, g = as_number(prediction), as_number(gold)
        return 1.0 if (p is not None and g is not None and abs(p - g) < 1e-6) else 0.0
    if check_type == "exact":
        return 1.0 if str(prediction).strip().casefold() == str(gold).strip().casefold() else 0.0
    # "llm_judge": a graded 0-1 quality score against the task rubric.
    return judge_score(prediction, gold, JUDGE_RUBRIC, JUDGE_TASK)

# ---- Running a candidate program -------------------------------------------
class Reply(str):
    """What call_model returns: the reply text, with the whole response attached.

    It IS a string, so `return call_model(prompt)` keeps working — but nothing is
    thrown away. `.blocks` holds every content block the call produced (tool calls,
    tool results, text), `.data` is the parsed JSON when the call passed a schema,
    and `.usage` / `.model` say what it cost.
    """
    def __new__(cls, text, blocks=(), usage=None, model="", data=None):
        reply = super().__new__(cls, text)
        reply.blocks = list(blocks)
        reply.usage = dict(usage or {})
        reply.model = model
        reply.data = data
        return reply

class Runtime:
    """A workflow program calls `runtime.call_model(...)` for every model call. This is
    the one place cost is measured, and it stops the program if it goes over budget."""
    def __init__(self, default_model, max_calls=24, token_budget=120_000):
        self.default_model = default_model
        self.max_calls = max_calls
        self.token_budget = token_budget
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0

    def call_model(self, prompt, model=None, system=None, tools=None,
                   effort=None, schema=None):
        if self.calls >= self.max_calls:
            raise RuntimeError("workflow exceeded its model-call budget")
        self.calls += 1
        if model not in BY_ID:
            model = self.default_model
        text, blocks, usage = _call_api(
            model, str(prompt), system=system, tools=tools, effort=effort, schema=schema)
        data = None
        if schema:
            try:
                data = json.loads(text)
            except ValueError:
                data = None
        self.tokens += usage["input"] + usage["output"] + usage["cache_write"] + usage["cache_read"]
        self.cost += cost_usd(model, usage)
        if self.tokens > self.token_budget:
            raise RuntimeError("workflow exceeded its token budget")
        return Reply(text, blocks=blocks, usage=usage, model=model, data=data)

# The shape of a final answer. Handed to every program so the graded contract and
# the schema it asks the model for cannot drift apart. This is the shape for the
# value solve() RETURNS — intermediate calls (a difficulty router, a decomposer)
# should use whatever schema fits them.
class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str

# Handed to programs as a plain dict: model-written code runs in a sandbox with no
# pydantic, and call_model's schema= accepts either form.
ANSWER_SCHEMA = Answer.model_json_schema()

# Candidate code is model-written, so it runs with a restricted import list and a
# small builtins allowlist rather than full Python.
_ALLOWED = {"re", "json", "math", "statistics", "collections", "itertools",
            "functools", "string"}
def _imp(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED:
        raise ImportError(name)
    return builtins.__import__(name, *a, **k)
_B = {n: getattr(builtins, n) for n in (
    "range len min max sum sorted abs round divmod pow enumerate zip map filter "
    "list dict set tuple str int float bool any all isinstance reversed print "
    "Exception ValueError ZeroDivisionError").split()}
_B["__import__"] = _imp

def compile_solve(code):
    namespace = {"__builtins__": _B, "re": re, "json": json, "statistics": statistics,
                 "Counter": Counter, "extract_last_number": extract_last_number,
                 "MODELS": MODELS, "ANSWER_SCHEMA": ANSWER_SCHEMA}
    exec(code, namespace)
    if not callable(namespace.get("solve")):
        raise ValueError("program does not define solve(question, call_model)")
    return namespace["solve"]

def final_answer(returned):
    # What solve() handed back, reduced to the string we grade. A program returns
    # its answer directly, a dict carrying the answer plus anything else it wants
    # to keep, or a schema-constrained Reply as-is — all three are unwrapped to the
    # answer. A structure with no "answer" in it is a contract violation, not an
    # answer: stringifying it would grade `{'result': 'positive'}` as the
    # prediction and silently score 0, so raise and let the record show why.
    if isinstance(returned, Reply) and returned.data is not None:
        returned = returned.data
    if isinstance(returned, dict):
        if "answer" not in returned:
            raise ValueError(
                f"solve() returned an object with no 'answer' key: {sorted(returned)}")
        returned = returned["answer"]
    return str(returned).strip()

class _Timeout(Exception):
    pass

def _with_timeout(fn, seconds):
    # Stop a program that hangs (an unbounded loop, a wedged call).
    if not hasattr(signal, "SIGALRM"):
        return fn()
    def _raise(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def main():
    global CHECK_TYPE, JUDGE_RUBRIC, JUDGE_TASK
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: eval_candidate.py <file.py>"}))
        return

    spec = json.load(open("task_spec.json"))
    dev = json.load(open("dev_task.json"))
    CHECK_TYPE = spec["check"].get("type", "exact")
    JUDGE_RUBRIC = spec["check"].get("rubric", "")
    JUDGE_TASK = spec["check"].get("task", "")

    try:
        solve = compile_solve(open(sys.argv[1]).read())
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"compile: {error}"}))
        return

    # Run the program once per example: grade the answer it RETURNS, and tally
    # what the model calls cost. A program that crashes, hangs, or blows its
    # budget scores 0 on that example instead of sinking the run.
    scores = []
    costs = []
    errors = []
    for item in dev:
        runtime = Runtime(DEFAULT_MODEL)
        try:
            returned = _with_timeout(lambda: solve(item["question"], runtime.call_model), 90)
            scores.append(check_answer(final_answer(returned), item["answer"], CHECK_TYPE))
        except Exception as error:
            scores.append(0.0)
            errors.append(str(error)[:80])
        costs.append(runtime.cost)

    print(json.dumps({"ok": True, "accuracy": statistics.mean(scores),
                      "cost_per_query": statistics.mean(costs), "n": len(dev),
                      "errors": errors[:3]}))

main()
