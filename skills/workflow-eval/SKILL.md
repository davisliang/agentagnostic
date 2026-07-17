---
name: workflow-eval
description: Score a candidate solve(question, call_model) workflow program on the dev set. Use to test any candidate before keeping it — it reports accuracy and cost per query, or the error if the program is broken.
---

# Evaluate a workflow candidate

Test a candidate workflow program (a `.py` file defining `solve(question, call_model)`;
see the **workflow-design** skill for the contract) against the dev set.

## How to run

Run from the working directory — it must contain `task_spec.json` and
`dev_task.json` (the task's answer-format spec and the dev examples). Invoke the
bundled script on the candidate file:

```bash
python .claude/skills/workflow-eval/eval_candidate.py <candidate_file.py>
```

The script reconstructs the task's extractor + checker from `task_spec.json`
(numeric / exact-match / LLM-judge), runs the candidate on every dev example
through the same metered, sandboxed runtime the final search uses, and reports
the result.

## Output

A single JSON line:

- Success: `{"ok": true, "accuracy": <0..1>, "cost_per_query": <usd>, "n": <count>, "errors": [...]}`
- Failure: `{"ok": false, "error": "<message>"}`

Read it as: `accuracy` is the fraction correct on the dev set, `cost_per_query`
is the measured mean USD cost. `ok: false` means the program didn't compile or
didn't define `solve`. A non-empty `errors` array with `ok: true` means some
examples raised — investigate and fix the candidate. Keep candidates that run
cleanly; fix or drop the rest.
