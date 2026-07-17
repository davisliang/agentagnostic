---
name: workflow-design
description: Design and select a diverse set of inference-time LLM workflows for a task, as runnable Python programs that span the cost/accuracy tradeoff. Use when asked to propose, design, or optimize LLM workflows/strategies for a dataset.
---

# Design inference-time LLM workflows

Produce a diverse set of candidate workflows for the task described in the
prompt, each as a Python program, and keep only the ones that actually work.
Span the cost/accuracy tradeoff — from a cheap single call to elaborate
multi-call paradigms (chain-of-thought, self-consistency / majority vote,
decompose-then-solve, debate, difficulty routing, cheap-model-with-escalation).

## The program contract

Each candidate is a `.py` file defining exactly:

```python
def solve(question, llm):
    ...
    return answer
```

- `llm(prompt, max_tokens=256, model=None, system=None, tools=None, effort=None)`
  is the ONLY way to call a model; it returns the response text.
    - `model=<name>` — route to a specific model (see `MODELS`, cheap → expensive).
    - `system="..."` — set a system prompt for that call.
    - `tools=["code_execution"]` and/or `tools=["web_search"]` — let the model run
      Python or search the web (server-side; results come back in the reply). Use a
      large model (Sonnet 5 or Opus 4.8) with tools.
    - `effort="low"|"medium"|"high"|"xhigh"|"max"` — turn on the model's own
      step-by-step thinking at that depth (Sonnet 5 / Opus 4.8 only; ignored on the
      cheap model). Costs more tokens; the per-query budget still applies.
- Inside `solve` you may use, with no imports: `re`, `json`, `statistics`,
  `Counter`, `extract_number(text) -> float | None`, and the list `MODELS`.
- No file / network / system access inside `solve`.
- The runtime meters cost and enforces a per-query call/token budget, so keep the
  number of model calls modest.
- Format the returned answer to match how it will be scored — the prompt states
  the extract + check rules (e.g. end with the number for numeric tasks, or put a
  short label on its own last line for exact-match).

## Improving existing workflows

If the prompt gives you existing workflows with their accuracy and cost, your job
is to make them **cheaper without losing accuracy**. Good moves: use a cheaper
model, make fewer model calls, route easy inputs to the cheap model and only
escalate hard ones, or use `tools=["code_execution"]` so the model computes
exactly instead of sampling many times. Keep a new candidate only if it stays at
least as accurate as the best existing workflow while costing less per query.

Resending the **same prompt to the same model** is much cheaper on input: prompt
caching bills the repeat at ~10% of the input rate (the first send costs a bit more).
Output tokens are never cached, so this cuts self-consistency's *input* cost but not
its output cost, and a *different* model never shares the cache.

## Workflow

1. Optionally research inference-time techniques with WebSearch / WebFetch.
2. Write each candidate to its own `.py` file in the working directory.
3. Test candidates **one at a time, in the foreground**, with the **workflow-eval**
   skill. Do NOT launch background jobs (`&`), job control, or many evals at once —
   idle-waiting on background jobs makes the session hang and drop the connection.
   Run one eval, read its JSON result, then move to the next. Fix or drop
   candidates that error.
4. Keep **4–5** diverse, WORKING candidates spanning cheap → accurate. One dev run
   per candidate is enough — don't re-test.
5. **Your final action MUST be to write `programs.json`** in the working directory:
   a JSON list of objects with keys `name`, `description`, `code` (code = the full
   `solve` source). Include every candidate that passed; exclude any that errored.
   Do this even if a candidate was slow — never end the session without writing
   `programs.json`.
