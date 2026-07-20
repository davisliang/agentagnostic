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
def solve(question, call_model):
    ...
    return answer
```

- **`solve` must RETURN the final answer itself** — nothing is parsed out of prose.
  Return the bare value (the number, the label, the text), not a sentence wrapped
  around it: return `"42"`, not `"The answer is 42."`. The prompt states the
  `check` rule that scores it.
    - You may instead return a dict `{"answer": <the answer>, ...}` if you want to
      keep extra context alongside it. Only `answer` is graded — a returned object
      with no `answer` key is a contract violation and scores 0 with an error.
- `call_model(prompt, model=None, system=None, tools=None, effort=None, schema=None)`
  is the ONLY way to call a model. It returns a `Reply`, which **is** a string (so
  `return call_model(p)` works), with the full response attached: `.blocks` (every
  content block, including tool calls and their results), `.data` (parsed JSON when
  you passed a `schema`), `.usage`, `.model`.
    - `model=<name>` — route to a specific model (see `MODELS`, cheap → expensive).
    - `system="..."` — set a system prompt for that call.
    - `tools=["code_execution"]` and/or `tools=["web_search"]` — let the model run
      Python or search the web (server-side; results come back in the same reply).
    - `effort="low"|"medium"|"high"|"xhigh"|"max"` — turn on the model's own
      step-by-step thinking at that depth (Sonnet 5 / Opus 4.8 only; ignored on the
      cheap model). Costs more tokens; the per-query budget still applies.
    - `schema=<JSON Schema>` — constrain the reply to JSON matching it, and read the
      parsed object off `reply.data`. This is the most reliable way to get a clean
      answer out of a model, and it composes with `tools=` (the schema shapes only
      the text the model writes at the end). If the model refuses, `reply.data`
      is `None` — fall back to the reply text.
      Use the provided **`ANSWER_SCHEMA`** for the call that produces the final
      answer; it is `{"answer": <string>}`, the shape `solve` is graded on. Give
      intermediate calls whatever schema fits them — a difficulty router wants
      `{"difficulty": ...}`, a decomposer `{"subquestions": [...]}`. Only the value
      you RETURN has to carry an `answer`.
- Inside `solve` you may use, with no imports: `re`, `json`, `statistics`,
  `Counter`, `extract_last_number(text) -> float | None`, the list `MODELS`, and
  `ANSWER_SCHEMA`.
- No file / network / system access inside `solve`.
- There is no output-length knob: every call gets one generous ceiling. You pay for
  the tokens a reply actually uses, so length is controlled by the prompt, not a cap.
- The runtime meters cost and enforces a per-query call/token budget, so keep the
  number of model calls modest.

A reliable shape for the last step of a workflow:

```python
def solve(question, call_model):
    reply = call_model(question, schema=ANSWER_SCHEMA)
    return reply.data["answer"] if reply.data else str(reply).strip()
```

Returning the reply itself also works — `return call_model(question, schema=ANSWER_SCHEMA)`
is unwrapped to its `answer` for you.

## Improving existing workflows

If the prompt gives you existing workflows with their accuracy and cost, your job
is to make them **cheaper without losing accuracy**. Good moves: use a cheaper
model, make fewer model calls, route easy inputs to the cheap model and only
escalate hard ones, or use `tools=["code_execution"]` so the model computes
exactly instead of sampling many times. Keep a new candidate only if it stays at
least as accurate as the best existing workflow while costing less per query.

**Prompt caching only engages above a per-model size floor — check before relying
on it.** Resending the same prompt to the same model bills the repeat at ~10% of the
input rate, but ONLY if the shared prefix is long enough. Below the floor nothing is
cached and there is no error, just a silently full-price call:

| model | shared prefix must exceed |
|---|---|
| `claude-haiku-4-5` | ~4,096 tokens |
| `claude-opus-4-8` | ~4,096 tokens |
| `claude-sonnet-5` | ~1,024 tokens |

Short-prompt tasks are the common case and they are all far below this — a one-line
question with a paragraph of system prompt is ~100 tokens, so caching cannot help at
all. Do NOT choose a workflow shape on the theory that repeating a prompt is cheap
unless the prefix is genuinely long (a big system prompt, few-shot examples, a
document). Verify rather than assume: `reply.usage["cache_read"]` is 0 when nothing
cached, and the runtime reports the cached share of input tokens for the whole run.

Output tokens are never cached, so even when caching does engage it cuts
self-consistency's *input* cost but not its output cost, and a *different* model
never shares the cache.

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
5. **Name each candidate by its structure**, using the **workflow-naming** skill —
   `H→S`, `S×5→vote`, `H→{self: stop|S^}`. Names describe what the program does, not
   what you hoped it would achieve, so the results table can be compared by name.
6. **Your final action MUST be to write `programs.json`** in the working directory:
   a JSON list of objects with keys `name`, `description`, `code` (code = the full
   `solve` source). Include every candidate that passed; exclude any that errored.
   Do this even if a candidate was slow — never end the session without writing
   `programs.json`.
