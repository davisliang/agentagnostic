#!/usr/bin/env python3
"""Score a candidate solve(question, llm) program on the dev set.

Task-agnostic: reconstructs the extractor + checker from task_spec.json
(numeric / exact-match / llm_judge). Reads task_spec.json and dev_task.json from
the current working directory. Usage: python eval_candidate.py <candidate.py>
"""
import re, sys, json, statistics, builtins, signal
from collections import Counter
import anthropic

PRICES = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (3.0, 15.0),
          "claude-opus-4-8": (5.0, 25.0)}
DEFAULT_MODEL = "claude-haiku-4-5"
client = anthropic.Anthropic()

def cost_usd(m, ti, to):
    pin, pout = PRICES[m]
    return (ti * pin + to * pout) / 1_000_000

def extract_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None

def _call_api(model, prompt, max_tokens):
    msg = client.messages.create(model=model, max_tokens=max_tokens,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens

def make_extractor(spec):
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
            return pn is not None and gn is not None and abs(pn - gn) <= tol
        return ck
    if t == "exact":
        def norm(x):
            s = str(x)
            if spec.get("strip", True):
                s = s.strip()
            if spec.get("casefold", True):
                s = s.casefold()
            return s
        return lambda pred, gold: norm(pred) == norm(gold)
    jm = spec.get("model", "claude-haiku-4-5")
    def ck(pred, gold):
        q = (f"Task: {spec.get('task', '')}\nReference answer: {gold}\n"
             f"Candidate answer: {pred}\nIs the candidate correct / equivalent to "
             "the reference? Answer 'yes' or 'no'.")
        return judge_call(jm, q).strip().lower().startswith("y")
    return ck

class BudgetError(RuntimeError):
    pass

class Runtime:
    def __init__(self, default_model, max_calls=24, max_tokens=120000):
        self.default_model = default_model
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.tokens = 0
        self.cost = 0.0
    def llm(self, prompt, max_tokens=256, model=None):
        if self.calls >= self.max_calls:
            raise BudgetError("call cap")
        self.calls += 1
        m = model if model in PRICES else self.default_model
        text, ti, to = _call_api(m, str(prompt), int(max_tokens))
        self.tokens += ti + to
        self.cost += cost_usd(m, ti, to)
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
          "MODELS": list(PRICES)}
    exec(code, ns)
    if not callable(ns.get("solve")):
        raise ValueError("program does not define solve(question, llm)")
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
    n_ok, costs, errs = 0, [], []
    for item in dev:
        rt = Runtime(DEFAULT_MODEL)
        try:
            ans = _with_timeout(lambda: solve(item["question"], rt.llm), 90)
            pred = extract(ans if isinstance(ans, str) else str(ans))
            ok = bool(check(pred, item["answer"]))
        except Exception as e:
            ok = False
            errs.append(str(e)[:80])
        n_ok += int(ok)
        costs.append(rt.cost)
    print(json.dumps({"ok": True, "accuracy": n_ok / len(dev),
                      "cost_per_query": sum(costs) / len(costs), "n": len(dev),
                      "errors": errs[:3]}))

main()
