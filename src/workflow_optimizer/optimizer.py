"""Step 3 — loop the design agent, then rank the survivors on held-out test.

Each round the agent is shown the best workflows so far and asked for cheaper
ones that hold accuracy; every candidate is scored on dev and added to the
archive. Only the dev Pareto frontier is re-scored on test — those are the
candidates actually worth choosing between, and test calls cost money.
"""
from dataclasses import dataclass, field
from typing import Optional

from . import designer
from .analysis import Benchmark
from .pareto import pareto_front
from .runtime import Evaluator, SplitScore
from .session import Session

# Which split to compare candidates on. Pass one to the pareto helpers so every
# comparison says out loud which numbers it is using.
DEV = lambda candidate: candidate.dev      # noqa: E731
TEST = lambda candidate: candidate.test    # noqa: E731


@dataclass
class Candidate:
    """One workflow program the design agent proposed, and how it scored.

    Attributes:
        name: Structural name, e.g. "H×3→vote→?S^". Unique within a search.
        description: The agent's one-line summary of what the workflow does.
        code: The program's source — a `solve(question, call_model)` definition.
        dev: Its SplitScore on the dev split, set once scored.
        test: Its SplitScore on held-out test. Only finalists have one.
    """
    name: str
    description: str
    code: str
    dev: Optional[SplitScore] = None
    test: Optional[SplitScore] = None

    @property
    def program(self) -> dict:
        """The `{"name", "code"}` dict `Evaluator.run` takes."""
        return {"name": self.name, "code": self.code}


@dataclass
class Search:
    """The result of one optimization: everything tried, and the finalists.

    Attributes:
        archive: Every distinct candidate seen, in the order proposed. All are
            dev-scored.
        finalists: The dev Pareto frontier, re-scored on held-out test. These are
            the workflows worth choosing between.
    """
    archive: list[Candidate] = field(default_factory=list)
    finalists: list[Candidate] = field(default_factory=list)


def optimize(cfg, benchmark: Benchmark, evaluator: Evaluator = None, log=print) -> Search:
    """Design candidates, score them on dev, then rank the frontier on test.

    Args:
        cfg: The run config; `cfg.designer.rounds` sets how many rounds run.
        benchmark: What is being optimized — task, grader, and the two splits.
        evaluator: Scores candidates. Built from `cfg` and the benchmark's grader
            if omitted.
        log: Where progress goes.

    Returns:
        A Search. `finalists` is empty when no candidate survived.
    """
    evaluator = evaluator or Session.from_config(cfg).evaluator(benchmark.grader)
    search = Search()

    for round_num in range(1, cfg.designer.rounds + 1):
        log(f"\n===== design round {round_num} / {cfg.designer.rounds} =====")
        context = designer.summarize_archive(search.archive)      # empty on round 1
        for program in designer.run_design_round(cfg, benchmark, round_num, context, log=log):
            if any(program["code"] == c.code for c in search.archive):   # skip exact repeats
                continue
            candidate = Candidate(name=_unique_name(program["name"], search.archive),
                                  description=program.get("description", ""),
                                  code=program["code"])
            candidate.dev = evaluator.run(candidate.program, benchmark.dev)
            search.archive.append(candidate)
            log(f"  + {candidate.name:24s} dev acc {candidate.dev.accuracy:.2f}  "
                f"${candidate.dev.cost:.5f}/query  "
                f"cached {candidate.dev.cached_input_frac:.0%}")

    log(f"\n{len(search.archive)} workflows in the archive after {cfg.designer.rounds} rounds.")
    if not search.archive:
        return search

    log("scoring the dev frontier on the held-out test split...")
    for candidate in pareto_front(search.archive, on=DEV):
        candidate.test = evaluator.run(candidate.program, benchmark.test)
        search.finalists.append(candidate)
        log(f"  {candidate.name:24s} test acc {candidate.test.accuracy:.2f}  "
            f"${candidate.test.cost:.5f}/query")
    return search


def _unique_name(name: str, archive: list[Candidate]) -> str:
    """Make a proposed name unique within the archive.

    Two rounds can propose the same structural name for different code; keeping
    them distinguishable stops a results table from merging them.

    Args:
        name: The name the agent gave the program.
        archive: Candidates already accepted.

    Returns:
        `name`, or `name#2`, `name#3`, ... if it is already taken.
    """
    taken = {c.name for c in archive}
    if name not in taken:
        return name
    suffix = 2
    while f"{name}#{suffix}" in taken:
        suffix += 1
    return f"{name}#{suffix}"
