"""Estimating what a search will cost, before committing to it.

A search spends money in four places, and they are not equally predictable:

1. **Analysis** — a handful of structured calls, plus dataset generation when the
   task brings no data. Small and steady.
2. **The design agent** — an Opus session per round. This is usually the largest
   single item and the least predictable: it depends on how much the agent reads,
   searches and re-tests. It bills through the SDK, not through our meter, so it
   is only knowable from what past runs actually recorded.
3. **Scoring candidates on dev** — rounds × candidates × dev examples × the cost
   of one query. That last term varies by two orders of magnitude between a
   single cheap call and a five-sample vote, which is most of the uncertainty.
4. **Scoring the frontier on test**, plus a judge call per graded answer when the
   task is judged.

So the estimate is a range, not a number, and every figure it rests on is
reported alongside it. Where past runs on this machine measured something, that
measurement is used and labelled `measured`; otherwise a documented default is
used and labelled `default`. An estimate that hides which is which invites more
trust than it has earned.
"""
from dataclasses import dataclass, field
from statistics import median

# Defaults used only until this machine has measured the real thing. Each is a
# rough central value; the range below widens generously around them.
DEFAULT_AGENT_COST_PER_ROUND = 1.20   # USD, an Opus design session
DEFAULT_CANDIDATES_PER_ROUND = 4.5    # the skill asks for 4-5
DEFAULT_COST_PER_QUERY = 0.0015       # a mid-range workflow: a call or two
DEFAULT_ANALYSIS_COST = 0.05          # analyzer + case types
COST_PER_GENERATED_EXAMPLE = 0.004    # Opus writing labelled examples
JUDGE_COST_PER_GRADE = 0.0004         # one cheap judged answer

# How far the range spreads around the central estimate. Workflow cost per query
# is the dominant unknown — a vote-of-five costs ~10x a single call — so the
# spread is wide on purpose. Narrow bounds here would be false precision.
LOW_FACTOR, HIGH_FACTOR = 0.45, 2.6


@dataclass
class Estimate:
    """What a search is expected to cost, and what that rests on.

    Attributes:
        low: Low end of the plausible range, USD.
        expected: Central estimate, USD.
        high: High end, USD.
        breakdown: Named parts of the central estimate, USD each.
        assumptions: Human-readable statements of every figure used, each marked
            "measured" (from past runs here) or "default".
        based_on_runs: How many past runs the measured figures came from.
    """
    low: float = 0.0
    expected: float = 0.0
    high: float = 0.0
    breakdown: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    based_on_runs: int = 0


def observed(runs: list) -> dict:
    """Measure what past runs on this machine actually cost, per task.

    Cost per query is a property of the TASK, not of the machine: one measured
    ARC workflow cost $1.52 a query while an ifeval one cost $0.0013 — a factor
    of a thousand. Pooling those into a single median makes both estimates
    meaningless, so figures are kept per task and only fall back to the pool
    when a task has no history of its own.

    Args:
        runs: `(task, events)` pairs, events as `runstore.read_events` returns.

    Returns:
        `{"tasks": {task: figures}, "pooled": figures, "n_runs": n}`. Each
        `figures` holds whichever of `agent_cost_per_round`,
        `candidates_per_round`, `cost_per_query`, `cost_per_query_low` and
        `cost_per_query_high` the runs evidence.
    """
    by_task: dict[str, dict] = {}
    contributing = 0

    for task, events in runs:
        bucket = by_task.setdefault(task, {"agent": [], "per_round": [], "query": []})
        rounds, candidates, saw = set(), 0, False
        for event in events:
            kind = event.get("event")
            if kind == "agent_cost":
                bucket["agent"].append(event["usd"])
                saw = True
            elif kind == "round_start":
                rounds.add(event["round"])
            elif kind == "candidate":
                candidates += 1
                bucket["query"].append(event["dev_cost"])
                saw = True
        if rounds and candidates:
            bucket["per_round"].append(candidates / len(rounds))
        contributing += 1 if saw else 0

    def figures(bucket: dict) -> dict:
        """Reduce one bucket of raw observations to the figures the estimator uses."""
        out = {}
        if bucket["agent"]:
            out["agent_cost_per_round"] = median(bucket["agent"])
        if bucket["per_round"]:
            out["candidates_per_round"] = median(bucket["per_round"])
        if bucket["query"]:
            out["cost_per_query"] = median(bucket["query"])
            # The spread within a task is the real uncertainty: the designer will
            # try both a single cheap call and a five-sample vote.
            out["cost_per_query_low"] = min(bucket["query"])
            out["cost_per_query_high"] = max(bucket["query"])
        return out

    pooled = {"agent": [], "per_round": [], "query": []}
    for bucket in by_task.values():
        for key in pooled:
            pooled[key].extend(bucket[key])

    return {"tasks": {task: figures(b) for task, b in by_task.items()},
            "pooled": figures(pooled), "n_runs": contributing}


def estimate(cfg, history: dict = None, generates_data: bool = None,
             judged: bool = None) -> Estimate:
    """Estimate the cost of running a search with this config.

    Args:
        cfg: The resolved config the search would use.
        history: `observed(...)` output. Anything it provides is preferred over
            the defaults, and labelled as measured.
        generates_data: Whether examples will be generated rather than loaded.
            Inferred from `cfg.task.dataset` when omitted.
        judged: Whether grading calls a model per answer. Inferred from
            `cfg.task.check_type` when omitted; unknowable before analysis for a
            free-text task, where it is assumed False and said so.

    Returns:
        An Estimate. Its `assumptions` list is the point: every figure it used,
        and whether that figure was measured here or is a default.
    """
    history = history or {}
    notes, breakdown = [], {}
    task_name = cfg.task.name
    mine = (history.get("tasks") or {}).get(task_name, {})
    pooled = history.get("pooled") or {}

    def figure(key, fallback, describe, transfers=True):
        """Take the most specific measurement available, and say which it was.

        Args:
            key: Which figure to look up.
            fallback: The documented default.
            describe: Renders the chosen value as a human-readable clause.
            transfers: Whether a measurement from OTHER tasks is evidence for
                this one. True for properties of the search process — how many
                candidates a round produces, what a design session costs. False
                for cost per query, which is a property of the task: ARC measured
                $0.74 a query against ifeval's $0.0018, so borrowing across tasks
                is worse than the default, not better.
        """
        if key in mine:
            notes.append(f"{describe(mine[key])} — measured on {task_name}")
            return mine[key]
        if transfers and key in pooled:
            notes.append(f"{describe(pooled[key])} — measured on other tasks; "
                         f"nothing on {task_name} yet")
            return pooled[key]
        if key in pooled:
            notes.append(f"{describe(fallback)} — default. Other tasks have "
                         f"measurements, but cost per query is task-specific "
                         f"(ARC ran ~400x ifeval), so they are not used here")
        else:
            notes.append(f"{describe(fallback)} — default, nothing measured yet")
        return fallback

    rounds = int(cfg.designer.rounds)
    n_examples = int(cfg.data.n_examples)
    dev = max(1, int(n_examples * float(cfg.data.dev_fraction)))
    test = max(1, n_examples - dev)

    if generates_data is None:
        generates_data = not cfg.task.dataset
    if judged is None:
        judged = (cfg.task.check_type or "") == "llm_judge"

    # 1. analysis, and generating examples when the task brings none
    breakdown["analysis"] = DEFAULT_ANALYSIS_COST
    notes.append(f"analysis ~${DEFAULT_ANALYSIS_COST:.2f} — default")
    if generates_data:
        breakdown["generate examples"] = n_examples * COST_PER_GENERATED_EXAMPLE
        notes.append(f"generating {n_examples} examples at "
                     f"~${COST_PER_GENERATED_EXAMPLE:.3f} each — default")
    else:
        notes.append("dataset supplied, so nothing is generated")

    # 2. the design agent — usually the largest item
    agent = figure("agent_cost_per_round", DEFAULT_AGENT_COST_PER_ROUND,
                   lambda v: f"design agent ~${v:.2f} per round × {rounds}")
    breakdown["design agent"] = agent * rounds

    # 3. scoring every candidate on dev
    candidates = figure("candidates_per_round", DEFAULT_CANDIDATES_PER_ROUND,
                        lambda v: f"{v:.1f} candidates per round")
    per_query = figure("cost_per_query", DEFAULT_COST_PER_QUERY,
                       lambda v: f"${v:.5f} per query for a typical workflow",
                       transfers=False)
    total_candidates = candidates * rounds
    breakdown["score on dev"] = total_candidates * dev * per_query
    notes.append(f"dev split {dev} of {n_examples} examples "
                 f"(dev_fraction {cfg.data.dev_fraction})")

    # the agent tests its own candidates against a sample while iterating
    breakdown["agent self-tests"] = (total_candidates * int(cfg.designer.dev_sample_size)
                                     * per_query)

    # 4. the frontier on test — only the non-dominated candidates get there
    finalists = max(1.0, min(total_candidates, 3.0))
    breakdown["score on test"] = finalists * test * per_query
    notes.append(f"assuming ~{finalists:.0f} finalists reach the held-out test split")

    # 5. judged grading is an extra call per graded answer
    if judged:
        graded = total_candidates * dev + finalists * test
        breakdown["llm judge"] = graded * JUDGE_COST_PER_GRADE
        notes.append(f"judged grading adds ~${JUDGE_COST_PER_GRADE:.4f} per answer")
    elif (cfg.task.check_type or "") == "" and not cfg.task.description:
        notes.append("grading rule not known until the task is analyzed; assuming "
                     "it is not model-judged, which would cost more if it is")

    expected = sum(breakdown.values())

    # Where this task's own spread is known, scale the range by it rather than by
    # a made-up factor: the cheapest and dearest workflow actually seen bound the
    # query-cost term far better than a guess does.
    scored_queries = total_candidates * dev + finalists * test
    if "cost_per_query_low" in mine and per_query > 0:
        fixed = expected - scored_queries * per_query
        low = fixed + scored_queries * mine["cost_per_query_low"]
        high = fixed + scored_queries * mine["cost_per_query_high"]
        notes.append(f"range spans the cheapest and dearest workflow measured on "
                     f"{task_name}: ${mine['cost_per_query_low']:.5f} to "
                     f"${mine['cost_per_query_high']:.5f} per query")
    else:
        low, high = expected * LOW_FACTOR, expected * HIGH_FACTOR
        notes.append(f"range is a flat {LOW_FACTOR}x-{HIGH_FACTOR}x — no measured "
                     f"spread for {task_name} to derive it from")

    return Estimate(low=max(0.0, low), expected=expected, high=high,
                    breakdown={k: round(v, 4) for k, v in breakdown.items()},
                    assumptions=notes, based_on_runs=history.get("n_runs", 0))
