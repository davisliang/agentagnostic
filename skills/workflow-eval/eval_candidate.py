#!/usr/bin/env python3
"""Score a candidate solve(question, call_model) program on the dev set.

Task-agnostic: reconstructs the extractor + checker from task_spec.json
(numeric / exact-match / llm_judge). Reads task_spec.json and dev_task.json from
the current working directory. Usage: python eval_candidate.py <candidate.py>
"""
import re, sys, json, statistics, builtins, signal
from collections import Counter
from dataclasses import dataclass
import anthropic

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

def extract_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None

TOOL_DEFS = {
    "code_execution": {"type": "code_execution_20260521", "name": "code_execution"},
    "web_search":     {"type": "web_search_20260209", "name": "web_search"},
}

MAX_TOOL_TURNS = 5   # cap on API calls while a server-side tool keeps pausing the turn
def _get_usage(msg):
    # Anthropic returns these token counts in the response's usage block.
    return {
        "input": msg.usage.input_tokens,
        "output": msg.usage.output_tokens,
        "cache_write": getattr(msg.usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read": getattr(msg.usage, "cache_read_input_tokens", 0) or 0,
    }

def _call_api(model, prompt, max_tokens, system=None, tools=None, effort=None):
    # cache breakpoint on the prompt -> identical resends to this model are cheap
    content = [{"type": "text", "text": str(prompt),
                "cache_control": {"type": "ephemeral"}}]
    request = {"model": model, "max_tokens": max_tokens,
               "messages": [{"role": "user", "content": content}]}
    if system:
        request["system"] = [{"type": "text", "text": system,
                              "cache_control": {"type": "ephemeral"}}]
    if tools:
        request["tools"] = []
        for name in tools:
            request["tools"].append(TOOL_DEFS[name])
    if effort and model in BY_ID and BY_ID[model].thinks:
        request["thinking"] = {"type": "adaptive"}
        request["output_config"] = {"effort": effort}
        request["max_tokens"] = max(max_tokens, 8192)
    else:
        request["thinking"] = {"type": "disabled"}
    text = ""
    usage = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    turns = 0
    while turns < MAX_TOOL_TURNS:
        turns += 1
        msg = client.messages.create(**request)
        got = _get_usage(msg)
        usage["input"] += got["input"]
        usage["output"] += got["output"]
        usage["cache_write"] += got["cache_write"]
        usage["cache_read"] += got["cache_read"]
        for block in msg.content:
            if block.type == "text":
                text += block.text
        if msg.stop_reason != "pause_turn":
            break
        request["messages"].append({"role": "assistant", "content": msg.content})
    return text, usage

def make_extractor(spec):
    # Prefer the analyzer's task-specific extractor code (validated in the
    # notebook); fall back to the deterministic type if it's absent or errors.
    code = spec.get("code")
    if code:
        try:
            ns = {"re": re, "json": json, "extract_number": extract_number}
            exec(code, ns)
            extract = ns["extract"]
            extract("probe 0")          # smoke test: callable and doesn't crash
            return extract
        except Exception:
            pass
    t = spec.get("type", "full")
    if t == "last_number":
        def ex(text):
            n = extract_number(text)
            return "" if n is None else (str(int(n)) if n == int(n) else str(n))
        return ex
    if t == "last_line":
        def ex(text):
            lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
            return lines[-1].strip() if lines else ""
        return ex
    return lambda text: (text or "").strip()

def make_checker(spec, judge_call):
    # Returns a scorer ck(pred, gold) -> float in [0, 1]: 1.0/0.0 for numeric &
    # exact, a graded quality score for llm_judge (against the task rubric). Kept
    # identical to the notebook's check_answer so dev and test grade the same way.
    t = spec.get("type", "exact")
    if t == "numeric":
        tol = float(spec.get("tol", 1e-6))
        def _num(x):                                   # tolerate bare OR prose values
            s = str(x).replace(",", "").strip()
            try:
                return float(s)
            except ValueError:
                return extract_number(s)
        def ck(pred, gold):
            pn, gn = _num(pred), _num(gold)
            return 1.0 if (pn is not None and gn is not None and abs(pn - gn) <= tol) else 0.0
        return ck
    if t == "exact":
        def norm(x):
            s = str(x)
            if spec.get("strip", True):
                s = s.strip()
            if spec.get("casefold", True):
                s = s.casefold()
            return s
        def ck(pred, gold):
            return 1.0 if norm(pred) == norm(gold) else 0.0
        return ck
    jm = spec.get("model", "claude-haiku-4-5")
    rubric = str(spec.get("rubric", "")).strip() or "Does the candidate correctly and completely satisfy the task?"
    task = spec.get("task", "")
    def ck(pred, gold):
        q = (f"Task: {task}\n\nGrading rubric:\n{rubric}\n\n"
             f"Reference (an example of a good answer):\n{gold}\n\n"
             f"Candidate answer:\n{pred}\n\n"
             "Score how well the candidate satisfies the rubric for the task, from 0 to 100 "
             "(100 = fully correct/complete, 0 = wrong or empty). Reply with ONLY the number.")
        n = extract_number(judge_call(jm, q))
        return 0.0 if n is None else max(0.0, min(1.0, n / 100.0))
    return ck

class BudgetError(RuntimeError):
    pass

class Runtime:
    def __init__(self, default_model, max_calls=24, max_tokens=120000):
        self.default_model = default_model
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.tokens = 0
        self.cost = 0.0
    def call_model(self, prompt, max_tokens=256, model=None, system=None, tools=None, effort=None):
        if self.calls >= self.max_calls:
            raise BudgetError("call cap")
        self.calls += 1
        m = model if model in BY_ID else self.default_model
        text, usage = _call_api(m, str(prompt), int(max_tokens),
                                system=system, tools=tools, effort=effort)
        self.tokens += usage["input"] + usage["output"] + usage["cache_write"] + usage["cache_read"]
        self.cost += cost_usd(m, usage)
        if self.tokens > self.max_tokens:
            raise BudgetError("token cap")
        return text

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
    ns = {"__builtins__": _B, "re": re, "json": json, "statistics": statistics,
          "Counter": Counter, "extract_number": extract_number,
          "MODELS": MODELS}
    exec(code, ns)
    if not callable(ns.get("solve")):
        raise ValueError("program does not define solve(question, call_model)")
    return ns["solve"]

class _TO(Exception):
    pass
def _with_timeout(fn, sec):
    if not hasattr(signal, "SIGALRM"):
        return fn()
    def _h(s, f):
        raise _TO()
    old = signal.signal(signal.SIGALRM, _h)
    signal.setitimer(signal.ITIMER_REAL, sec)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: eval_candidate.py <file.py>"}))
        return
    spec = json.load(open("task_spec.json"))
    dev = json.load(open("dev_task.json"))
    extract = make_extractor(spec["extract"])
    check = make_checker(spec["check"], lambda m, p: _call_api(m, p, 16)[0])
    code = open(sys.argv[1]).read()
    try:
        solve = compile_solve(code)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"compile: {e}"}))
        return
    total, costs, errs = 0.0, [], []
    for item in dev:
        rt = Runtime(DEFAULT_MODEL)
        try:
            ans = _with_timeout(lambda: solve(item["question"], rt.call_model), 90)
            pred = extract(ans if isinstance(ans, str) else str(ans))
            score = float(check(pred, item["answer"]))       # in [0, 1]
        except Exception as e:
            score = 0.0
            errs.append(str(e)[:80])
        total += score
        costs.append(rt.cost)
    print(json.dumps({"ok": True, "accuracy": total / len(dev),
                      "cost_per_query": sum(costs) / len(costs), "n": len(dev),
                      "errors": errs[:3]}))

main()
