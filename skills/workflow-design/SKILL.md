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

- `llm(prompt, max_tokens=256, model=None)` is the ONLY way to call a model; it
  returns the response text. Route between models by passing `model=<name>`.
- Inside `solve` you may use, with no imports: `re`, `json`, `statistics`,
  `Counter`, `extract_number(text) -> float | None`, and the list `MODELS`.
- No file / network / system access inside `solve`.
- The runtime meters cost and enforces a per-query call/token budget, so keep the
  number of model calls modest.
- Format the returned answer to match how it will be scored — the prompt states
  the extract + check rules (e.g. end with the number for numeric tasks, or put a
  short label on its own last line for exact-match).

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
