A user wants to build an LLM workflow for this task:
${seed_prompt}
${examples}

Reply with:

- description: one clear paragraph describing the task for a workflow designer.
- check_type: how to score a prediction against the gold answer — 'numeric' for
  numbers, 'exact' for short labels/strings, 'llm_judge' for free-form/semantic
  answers.
- judge_rubric: ONLY for check_type 'llm_judge' — a short rubric (2-5 concrete
  criteria) defining what makes a candidate output correct / high-quality for
  THIS task; it grades candidates from 0 to 100. It is calibration-checked
  before use: scored under it, an ideal reference answer must reach at least
  ${min_gold_score}, and an EMPTY answer must stay at or below ${max_empty_score} — a rubric
  that fails either is thrown away for a generic one. So write criteria that
  AWARD points for the qualities a good answer has, not only ones that deduct
  for faults: a rubric phrased purely as penalties gives an empty answer full
  marks (it contains no errors) and fails. Return an empty string for other
  check_types.
- answer_examples: 2-4 examples of CORRECT final answers for this task, in
  exactly the form a workflow should return them — a BARE value for
  numeric/exact tasks (the number or the label alone, no words or units around
  it), a full reference answer for free-form ones. These are shown to the
  workflow designer as the target format, and are used to sanity-check the
  judge.
