---
name: workflow-skills
description: Read and write a working set of reusable skill notes for the current run. Use to record a technique or task-specific gotcha you discover, so later design rounds of this run benefit from it instead of rediscovering it.
---

# Working skills for this run

You have a `working_skills/` directory in your working directory: short `SKILL.md`
notes on what works for THIS task. They persist across the design rounds of this
run and are discarded when the run ends, so a lesson you record in round 1 is
available to you in round 2.

## Read them first

Before you design anything, read every `working_skills/*/SKILL.md` and build on
what they say. On round 1 the directory may be empty — that is expected.

## Write what you learn

When you discover something worth reusing, record it so the next round doesn't
have to rediscover it. Good things to capture:

- a prompt or schema that reliably gets the answer into the graded format,
- a routing / escalation rule that works for this task,
- a grader quirk or answer-format gotcha you hit,
- a workflow shape that clearly beat the others, and why.

To add one, create `working_skills/<short-name>/SKILL.md` with YAML frontmatter
(`name:` and a one-line `description:`) followed by the note — concrete, specific
to THIS task, a few lines is plenty.

Keep them **few and high-signal**. Do not restate the built-in skills, and do not
record the obvious — record only what you actually learned about this task that a
fresh round would otherwise have to work out again.

## Reusable operators your workflows can CALL

Beyond notes, you can write reusable Python functions in `working_skills/helpers.py`.
Every function defined there is injected into your workflow's namespace, so your
`solve` code can call it **by name without defining or importing it** — exactly
like `extract_last_number`. Write an operator once, reuse it in every workflow this
run, and refine it across rounds.

Rules:

- `helpers.py` runs in the **same sandbox** as `solve` — same allowed imports and
  builtins. It may use `re`, `json`, `statistics`, `Counter`, `extract_last_number`,
  `MODELS`, and `ANSWER_SCHEMA` without importing them. A syntax error or a blocked
  import there fails every candidate that round, so test after editing it.
- Operators that call the model take `call_model` as a parameter — the workflow
  passes it in.
- Only **add** functions across rounds; do not rename or remove ones that earlier
  workflows call, or those workflows fail when re-scored on test.

Example `working_skills/helpers.py`:

    def self_consistency(question, call_model, n=5, model=None):
        votes = [call_model(question, model=model, schema=ANSWER_SCHEMA) for _ in range(n)]
        answers = [(v.data or {}).get("answer", str(v)).strip() for v in votes]
        return Counter(answers).most_common(1)[0][0]

    def verify_and_repair(question, draft, call_model, model=None):
        critique = call_model(f"Is this answer correct? If not, give the fix.\n"
                              f"Q: {question}\nA: {draft}", model=model)
        fixed = call_model(f"{question}\nA draft said: {draft}\nA reviewer said: "
                           f"{critique}\nGive the final answer.", model=model, schema=ANSWER_SCHEMA)
        return (fixed.data or {}).get("answer", str(fixed)).strip()

Then a workflow just calls them:

    def solve(question, call_model):
        return self_consistency(question, call_model, n=5)
