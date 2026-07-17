# Workflow Optimizer — V0

Automatically find the best **LLM workflow** for a task under a budget. Instead of
hand-writing prompting strategies, it has the model **design workflow programs** and
**iterate on them to cut cost without losing accuracy**, runs them on your data, and
lets you pick the point you want on the accuracy/cost **Pareto frontier**.

## The idea in one paragraph

A "workflow" is just an arbitrary Python function `solve(question, llm) -> answer`.
Because it's *code*, it can express **any** inference-time paradigm — a single call,
chain-of-thought, self-consistency, decomposition, debate, a cheap→expensive router —
without the harness needing a special case for each. The harness fixes only three
things, and all the generality rides on them:

1. **Contract** — every workflow is `solve(question, llm) -> answer`.
2. **Metered call site** — `llm(prompt, max_tokens, model)` is the *only* way a
   workflow can call a model. It's instrumented (counts tokens → USD) and
   budget-capped, so cost is measured at one chokepoint no matter what the code does.
3. **Task-inferred scoring** — an `extract` (pull the answer out of text) + `check`
   (numeric tolerance / exact-match / LLM-judge), so the evaluator never needs to
   know the paradigm.

## Pipeline (the notebook, top to bottom)

1. **Define the problem** — the *only* per-task input: a `SEED_PROMPT` and an
   optional `DATASET` (a list of `{"question", "answer"}`). Leave `DATASET = None`
   to have one generated.
2. **Profile the task** — one structured LLM call infers a task *description*, the
   grading *check* (numeric / exact-match / LLM-judge), and — rather than picking from a
   fixed menu — **writes an `extract(text)` function for the task**, which is validated
   against gold probes and falls back to a deterministic extractor if it doesn't round-trip
   (`build_extractor`). It also generates a labeled dataset if you didn't supply one. The
   data is split into **dev** (the designer may tune on this) and a held-out **test** set.
3. **Optimize (a loop)** — a **Claude Agent SDK** agent (web search + Bash + file
   tools), driven by two skills, runs for `N_ROUNDS`:
   - `workflow-design` — the methodology and the program contract.
   - `workflow-eval` — a bundled `eval_candidate.py` that scores a candidate on dev.

   Round 1 designs a diverse initial set. Each later round is shown the best workflows
   so far and asked for **cheaper** ones that keep accuracy (a cheaper model, fewer
   calls, difficulty routing, code execution instead of many samples). Every candidate
   is scored on the **dev** split and added to an `archive`. The agent runs in a
   **clean subprocess** (`run_proposer.py`) to stay isolated from the notebook kernel's
   async state; if it doesn't write `programs.json`, the notebook salvages the
   candidate files from disk.
4. **Rank finalists** — take the dev-set Pareto frontier of the archive and
   re-evaluate just those on the **held-out test split** for honest accuracy/cost
   numbers.
5. **Choose** — compute the accuracy/cost **Pareto frontier**, answer the two
   constrained questions — *best workflow under a $ budget* and *cheapest workflow
   above an accuracy floor* — plot it, then print each finalist's code so you can pick
   the methodology you want (`CHOICE`).

## The metered runtime

Generated programs are model-written code, so each one runs through a small
`Runtime` (`evaluate_program`):

- **Metered** — every `llm()` call adds to a per-query token/cost tally.
- **Capped** — hard limits on model calls and tokens per query; a program that
  blows the budget or crashes just scores 0 — it can't run away.

The program's source is run with plain `exec` (it's the model designing workflows
for *your* task in a research notebook). For untrusted code, run it in a container.

## The two skills

Standard `SKILL.md` skills under `skills/`, copied into the agent's `.claude/skills/`
at runtime so the SDK discovers them:

- `skills/workflow-design/SKILL.md` — how to design + test candidates, and the `solve`
  contract.
- `skills/workflow-eval/{SKILL.md, eval_candidate.py}` — the dev evaluator; it mirrors
  the notebook's runtime and reconstructs the extractor/checker from `task_spec.json`.

## Running it

```sh
uv run jupyter notebook          # then open workflow_optimizer_v0.ipynb
```

- Set `ANTHROPIC_API_KEY` in your shell first — **every cell makes real API calls**
  (the profiler, the design agent plus its self-tests, and the search).
- To run a different task, edit `SEED_PROMPT` (and optionally `DATASET`) in the
  "Define the problem" cell, then **Kernel → Restart & Run All**.
- If you edit the notebook outside Jupyter, do **File → Reload Notebook from Disk**
  before running, or a stale in-memory copy will overwrite the change on save.

## Repo layout

```
workflow_optimizer_v0.ipynb        the pipeline
run_proposer.py                    drives the design agent in a clean subprocess
skills/
  workflow-design/SKILL.md         design methodology + program contract
  workflow-eval/SKILL.md           dev-evaluator skill
  workflow-eval/eval_candidate.py  the bundled dev evaluator
pyproject.toml, uv.lock            deps (anthropic, claude-agent-sdk, jupyter, ...)
```

## Notes

- The `llm_judge` checker makes its own cheap API calls; that cost is the
  *evaluator's* and is deliberately **not** counted as workflow cost.
- Costs are **cache-aware**: every `llm()` call sets a prompt-cache breakpoint, so the
  same prompt resent to the same model bills cache reads (~90% off the input rate),
  while a different model never shares the cache (always a fresh, uncached call).
  `cost_usd` splits `usage` into input / output / cache-write / cache-read and prices
  each — writes at 1.25×, reads at 0.10× the input rate.
- The dataset generator and the agent's dev self-tests use a small slice of data for
  speed; the final Pareto ranking is always on the held-out test split, so programs
  aren't scored on data the designer tuned against.
