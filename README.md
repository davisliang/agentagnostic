# Workflow Optimizer

Automatically find the best **LLM workflow** for a task under a budget. Instead of
hand-writing prompting strategies, a design agent **writes workflow programs** and
**iterates on them to cut cost without losing accuracy**, runs them on your data, and
hands you the accuracy/cost **Pareto frontier** to pick from.

```sh
export ANTHROPIC_API_KEY=...
uv sync
uv run workflow-optimizer --task gsm8k                     # a file under config/task/
uv run workflow-optimizer --task gsm8k designer.rounds=1   # override any config key
```

In Python, `Session` is the entry point — it holds the config, the model catalog
derived from it, and the client:

```python
from workflow_optimizer import Session, analysis, optimize, report

session = Session.load("gsm8k", ["designer.rounds=1"])
benchmark = analysis.build_benchmark(session.cfg, session.client)
search = optimize(session.cfg, benchmark, session.evaluator(benchmark.grader))
report.summarize(search, session.cfg)
```

## Vocabulary

These words mean one thing each, everywhere — in the code, the config, and below.

| Term | Meaning |
| --- | --- |
| **workflow** | A Python program `solve(question, call_model) -> answer`. The thing being optimized. |
| **candidate** | One workflow the design agent proposed, plus how it scored. |
| **benchmark** | A task made measurable: its analysis, a grader, and dev/test splits. |
| **task analysis** | What the task is and how an answer should be graded, inferred from your description. |
| **grader** | Scores one answer in [0, 1] — numeric, exact, LLM-judge, or a task's own metric. |
| **split score** | One candidate's accuracy and cost on one split. |
| **search** | One optimization: every candidate tried, and the finalists. |
| **dev / test** | Dev guides the search; test is held out and only the final ranking touches it. |
| **frontier** | The non-dominated candidates — nothing else is both cheaper and more accurate. |
| **call meter** | The per-query object whose `call_model` a workflow calls. Measures and caps. |

## The idea in one paragraph

A workflow is an arbitrary Python function `solve(question, call_model) -> answer`.
Because it's *code*, it can express **any** inference-time paradigm — a single call,
chain-of-thought, self-consistency, decomposition, debate, a cheap→expensive router —
without the harness needing a special case for each. The harness fixes only three
things, and all the generality rides on them:

1. **Contract** — every workflow is `solve(question, call_model) -> answer`. It
   *returns* its answer, so nothing has to be parsed back out of prose.
2. **Metered call site** — `call_model(prompt, model=...)` is the *only* way a
   workflow can reach a model. It counts tokens, prices them, and enforces a
   per-query budget, so cost is measured at one chokepoint no matter what the code
   does. It returns a `Reply` — a string carrying the full response, so a run can be
   inspected afterwards without anything being discarded.
3. **Task-inferred grading** — a `Grader` (numeric tolerance / exact match /
   LLM-judge / the task's own metric), so the evaluator never needs to know the
   paradigm.

## The pipeline

1. **Analyze the task** (`analysis`) — one structured call infers a task description,
   the grading rule, and — rather than picking from a fixed menu — a **judge rubric**
   for free-form tasks, which is then calibrated against example answers and dropped
   for a generic judge if it doesn't discriminate. `dataset` generates labeled
   examples if you didn't supply any, split into **dev** and held-out **test**. A task
   that already knows its own shape sets `task.description` in config and skips this.
2. **Design and optimize** (`designer`, `optimizer`) — a **Claude Agent SDK** agent
   runs once per round, driven by the three skills below. Round 1 designs a diverse
   initial set; each later round is shown the best workflows so far and asked for
   **cheaper** ones that hold accuracy (a cheaper model, fewer calls, difficulty
   routing, code execution instead of many samples). Every candidate is scored on dev
   and added to the archive.
3. **Rank finalists** — the dev frontier is re-scored on the **held-out test split**,
   for numbers nothing was tuned against.
4. **Choose** (`report`) — the frontier, the two constrained picks (*the best workflow
   I can afford*, *the cheapest one that's good enough*), a plot, and each finalist's
   code.

## Layout

```
config/                 every knob (OmegaConf)
  config.yaml           models + prices, call, runtime, judge, data, designer, report
  task/*.yaml           one file per task: the seed prompt, optional data + grader
prompts/*.md            every prompt sent to a model, as text (${placeholders})
skills/                 what the design agent is taught
  workflow-design/      the methodology and the program contract
  workflow-eval/        the dev evaluator (a wrapper over the same runtime)
  workflow-naming/      naming workflows by structure, so results tables compare
src/workflow_optimizer/
  config.py             typed config schema + loading/overrides
  session.py            Session: config + catalog + client, wired once
  models.py             ModelCatalog: ids, prices, capabilities
  client.py             ModelClient — the one place anything reaches a model
  prompts.py            fills prompts/*.md
  grading.py            Grader: numeric / exact / llm_judge / custom
  runtime.py            Reply, CallMeter, compile_solve, Evaluator, SplitScore
  analysis.py           what is this task, and how is an answer graded
  dataset.py            load the task's examples, or generate diverse ones
  designer.py           stage and run one design round
  proposer.py           the agent subprocess entry point
  optimizer.py          Candidate, Search — the round loop and the archive
  pareto.py             frontier + the two constrained picks
  report.py             frontier table, plot, search JSON
  cli.py                `workflow-optimizer`
notebooks/optimize.ipynb  the same pipeline, interactively
experiments/            benchmark comparisons (see routerllm_ifeval/)
tests/test_offline.py   everything checkable without spending money
```

## Configuration

Everything tunable lives in `config/`, nothing in the code. A task is one small file:

```yaml
# config/task/gsm8k.yaml
task:
  name: gsm8k
  seed_prompt: >-
    Grade-school math word problems. Each answer is a single integer.
data:
  n_examples: 120
```

Override any key from the command line — `uv run workflow-optimizer --task gsm8k
designer.rounds=1 runtime.concurrency=4 report.max_cost_per_query=0.001` — or in
Python via `Session.load("gsm8k", [...])`. The schema in
`src/workflow_optimizer/config.py` is typed, so a misspelled key fails at load time
instead of being silently ignored.

Optional task fields: `dataset` (a `.jsonl` of `{"question", "answer"}`), `grader`
(a `.py` exposing `grade(prediction, item) -> float`, to score with an external
benchmark's own metric), and `description` (skip the analyzer).

## The metered runtime

Generated programs are model-written code, so each one runs through
`workflow_optimizer.runtime`:

- **Metered** — every `call_model()` call adds to a per-query token and cost tally.
- **Capped** — `runtime.max_model_calls` and `runtime.max_tokens` per query; a program
  that blows its budget or crashes scores 0 on that example rather than sinking the run.
- **Sandboxed** — candidates run with a restricted import list and builtins allowlist
  (see `compile_solve`). The allowlist is deliberately generous: a name it refuses
  doesn't read as "blocked" in the results, it reads as "this strategy scores 0". This
  raises the bar; it is **not** a security boundary — run genuinely untrusted code in a
  container.

The design agent's dev evaluator (`skills/workflow-eval/eval_candidate.py`) is a thin
**wrapper over this same code**, handed the same resolved config, so a dev number means
what it will mean in the final ranking.

## Notes

- Grading returns a **score in [0, 1]**; a candidate's accuracy is the mean over the
  dataset. Nothing is parsed out of prose: `numeric` requires the answer to *be* a
  number and the judge replies under a schema. `numeric`/`exact` are 1/0; `llm_judge`
  is a **graded** score from a cheap model against a task-specific rubric, so for
  free-form tasks "accuracy" is mean quality. The judge's API calls are the
  *evaluator's* cost, deliberately **not** counted as workflow cost.
- Costs are **cache-aware**: every call sets a prompt-cache breakpoint, so the same
  prompt resent to the same model bills cache reads (~90% off the input rate), while a
  different model never shares the cache. Caching only engages above a per-model size
  floor (~1–4k tokens), so `SplitScore.cached_input_frac` reports the share of input
  tokens that actually came from cache — a workflow built on "resending is cheap" can
  otherwise pay full price with nothing to say so.
- The final ranking is always on the held-out test split, so candidates aren't scored
  on data the design agent tuned against.

## Tests

```sh
uv run pytest
```

The API is faked, so the suite costs nothing and covers what is pure logic: pricing,
grading, the answer contract, the Pareto helpers, the runtime's guardrails (sandbox,
call budget, crash isolation), config overrides, and what the design agent is handed.
