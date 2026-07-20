# routerllm ifeval — workflow optimization vs. a trained router

Optimizes a workflow for the `ifeval` task of routerllm's 14-dataset holdout
benchmark, and reports it against that repo's recorded baselines.

## Why this task

`ifeval` scores `prompt_level_strict_acc`: an answer counts only if it satisfies
EVERY formatting constraint stated in the prompt. On routerllm's numbers the
trained router scores 0.848 there — identical to `haiku_only`, i.e. it never
escalates — while the routing oracle reaches 0.957. Opus is *worse* than Haiku
on the train slice (0.810 vs 0.840), so capability isn't the lever; compliance
is. And because the constraints are stated in the prompt, a workflow can verify
its own answer before returning it, which is not something routing can express.

## Comparability

Grading uses lm-eval's own `test_instruction_following_strict`, the same checker
behind routerllm's `summary.json`. Validated: it reproduces the recorded
`correct{haiku,opus}` labels on all 146 ifeval examples (292/292 checks).

`grader.py` is deliberately kept OUT of the workflow sandbox. A program able to
call the checker would be grading itself against the metric.

## Data

    selection : 100 train examples  (split.json train partition, seed 0)
    reporting : 46 test examples    (holdout14 <=100/task subset)

Zero overlap between them; the June-23 split's test partition is an exact set
match to `holdout14_examples.jsonl`.

## Run

    ANTHROPIC_API_KEY=... python -u run_ifeval.py --rounds 3
    python -u run_ifeval.py --smoke        # 8 examples, 1 round, proves the wiring

Use `-u`: the design agent's subprocess output is block-buffered otherwise and
the run looks hung when it isn't.

Paths to the routerllm checkout are absolute in `grader.py` — adjust if the
sibling repo moves.
