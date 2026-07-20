# ifeval — results

3 design rounds, 13 candidates, 100 train / 46 held-out. **$36.97, 110 min.**

All figures are the held-out 46, never seen during selection. Recorded baselines
come from routerllm's `exp1_frontier`, which is a **batch** cost basis; the
workflows ran non-batch, so recorded costs are doubled to make the column
comparable.

| policy | acc | $/query | vs opus | x opus $ |
|---|---|---|---|---|
| `haiku_only` (recorded) | 0.848 | 0.00161 | -0.043 | 0.21 |
| **`haiku_single`** (workflow) | **0.891** | 0.00163 | +0.000 | 0.21 |
| `router@0.5` (recorded) | 0.848 | 0.00209 | -0.043 | 0.27 |
| `oracle_gain` (recorded) | 0.957 | 0.00243 | +0.066 | 0.32 |
| **`sonnet_single`** (workflow) | **0.935** | 0.00699 | +0.044 | 0.92 |
| `opus_only` (recorded) | 0.891 | 0.00761 | +0.000 | 1.00 |
| **`router_haiku_selfcheck_escalate`** | **0.957** | 0.01756 | +0.066 | 2.31 |

`haiku_single`'s cost landing on `haiku_only`'s doubled cost (0.00163 vs 0.00161)
is a useful check that the cost accounting lines up across the two harnesses.

## Findings

**The gain is mostly prompting, not workflow structure.** One Haiku call with a
constraint-focused system prompt reaches 0.891 — Opus's accuracy at a fifth of
its price, and +4.3 over the trained router. No workflow structure involved.
Any comparison against the recorded 0.848 conflates prompt quality with workflow
design; the honest control for judging a workflow is `haiku_single`, not the
recorded baseline.

**Sonnet is missing from the existing comparison.** `exp1_frontier` routes only
Haiku<->Opus. `sonnet_single` gets 0.935 at 0.92x Opus cost — a policy the
current table cannot express, beating everything in it except the oracle.

**Oracle accuracy is reachable without oracle knowledge, at 2.3x Opus cost.**
`router_haiku_selfcheck_escalate` matches `oracle_gain`'s 0.957. The oracle is
not an implementable policy (it needs to know which model is right); the
workflow is.

**Workflows can exceed the routing ceiling — demonstrably, but barely.** The
best workflow solved 1 example that neither Haiku nor Opus solved one-shot,
which no routing policy can do by construction. It also missed 1 that a single
model got. Net zero on 46 examples: the mechanism is real, the magnitude is not
measurable at this n.

**Code-execution verification is actively harmful here** — 0.522 / 0.674 /
0.783, all below plain Haiku at 10-14x the cost. A model-written constraint
checker rejects valid answers more often than it catches bad ones. Do NOT
generalise this to `humaneval_plus` / `mbpp_plus`, where the tests are given
rather than generated.

## Why the dev numbers should be distrusted

Dev->test rank correlation was only **+0.65** (Spearman, n=13).

- dev champion `haiku_draft_sonnet_higheffort_audit`: 0.983 dev -> **0.935** test,
  dominated by a plain Sonnet call at a quarter the cost.
- `haiku_draft_selfcheck`: +11.6 over plain Haiku on dev -> **below** it on test.
- `haiku_single` scored 0.850 and 0.817 on two runs of identical code.

Selecting the best of 13 on 60 examples bought several points of illusion.

**The four-way tie at 0.957 is coincidence, not a ceiling.** They fail four
different examples between them; only one example is failed by all four. An
ensemble would likely clear 0.957.

## Prompt caching never engaged

Every call in this run paid full price on input. ifeval's prompt plus system is
**96 tokens**; caching does not engage below ~4,096 on haiku/opus (~1,024 on
sonnet), and below that floor it fails silently — no error, `cache_read` simply
stays 0.

That invalidates one candidate's stated rationale. `sonnet_draft_sonnet_audit`
justified using the same model twice as "same-model repeat benefits from input
prompt caching". At 96 tokens that saving was unavailable, so the workflow was
chosen for a mechanism that could not fire.

The design skill was the source: its cost-reduction section advised that
resending the same prompt is ~10% of input price, unconditionally. That is true
for long prefixes and false for every short-prompt benchmark, which is most of
the 14 tasks. The skill now states the per-model floors, and
`evaluate_program` reports `cached_input_frac` so a run says plainly that
nothing cached instead of leaving it to be inferred from the bill.

On this task the real cost lever was making FEWER calls, not cheaper ones —
which is what `router_haiku_selfcheck_escalate` found.

## Caveats

- n=46: one example is 2.2 points. Differences under ~4 points are not resolvable.
- `haiku_single` here is not `haiku_only` there: same model, different prompt and
  no lm-eval chat template. For a number that drops into routerllm's own table,
  regrade `results/ifeval_answers.jsonl.gz` through a replay backend.
- ifeval grading is nondeterministic across processes (see README). Measured
  spread on this 46-example slice is 0.0000, so these numbers are unaffected.

## Artifacts

    results/ifeval_summary.json      metrics + full source of all 13 workflows
    results/ifeval_answers.jsonl.gz  598 per-example answers, for offline regrading
