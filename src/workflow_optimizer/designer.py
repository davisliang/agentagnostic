"""Step 2 — the design agent that writes candidate workflows.

Each round runs a Claude Agent SDK session in a scratch directory, driven by the
three skills under `skills/`. On round 1 it designs a diverse initial set; on
later rounds it is shown the best workflows so far and asked for cheaper ones
that hold accuracy.

The agent runs in a SEPARATE PROCESS (`workflow_optimizer.proposer`). A notebook
kernel that has called `nest_asyncio.apply()` monkeypatches `asyncio` for the
whole session and breaks `asyncio.run`, even from a worker thread; a subprocess
is immune.
"""
import json
import os
import re
import pathlib
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from . import prompts
from .pareto import pareto_front
from .paths import ROOT, SKILLS_DIR, resolve

# The candidates the agent sees are only ever dev-scored; test is held out.
_ON_DEV = lambda candidate: candidate.dev      # noqa: E731


def summarize_archive(candidates: list) -> str:
    """Describe the workflows found so far, for the next round's prompt.

    Args:
        candidates: Every Candidate scored so far. Each needs `.dev` set.

    Returns:
        The current dev Pareto frontier as one line per workflow, plus the full
        code of the most accurate one as a base to improve on. Empty string when
        there are no candidates yet, i.e. on round 1.
    """
    if not candidates:
        return ""
    lines = [f"- {c.name}: accuracy {c.dev.accuracy:.2f}, ${c.dev.cost:.5f}/query"
             for c in pareto_front(candidates, on=_ON_DEV)]
    best = max(candidates, key=lambda c: c.dev.accuracy)
    lines.append(f"\nMost accurate so far is '{best.name}' (accuracy {best.dev.accuracy:.2f}, "
                 f"${best.dev.cost:.5f}/query). Its code:\n{best.code}")
    return "\n".join(lines)


AGENT_COST = re.compile(r"\[agent cost: \$([0-9.]+) over (\d+) turns\]")


def run_design_round(cfg, benchmark, round_num: int, context: str, log=print,
                     on_cost=None) -> list[dict]:
    """Run one design round and collect the workflows it proposed.

    Stages a scratch directory, runs the agent there as a subprocess, and streams
    its output to `log` as it goes.

    Args:
        cfg: The run config.
        benchmark: The Benchmark being optimized — supplies the task description,
            grading rule and dev examples the agent may test against.
        round_num: 1-based round number. Round 1 asks for a diverse initial set;
            later rounds ask for cheaper workflows that hold accuracy.
        context: `summarize_archive(...)` output. Ignored on round 1.
        log: Where the agent's output goes, line by line.
        on_cost: Optional `on_cost(usd, turns)`, called with what the agent spent
            on itself. That spend goes through the SDK rather than our meter, so
            this is the only place it can be seen.

    Returns:
        The proposed programs as `{"name", "description", "code"}` dicts. Possibly
        empty if the agent produced nothing usable.
    """
    agent_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"workflow_design_r{round_num}_"))
    _stage_agent_dir(cfg, benchmark, agent_dir)

    (agent_dir / "proposer_config.json").write_text(json.dumps({
        "model": cfg.designer.model,
        "skills": list(cfg.designer.skills),
        "allowed_tools": list(cfg.designer.allowed_tools),
        "prompt": _round_prompt(cfg, benchmark, round_num, context),
    }))

    # The agent shells out to the eval skill, which imports workflow_optimizer;
    # both variables keep that working even if `python` on its PATH is not this
    # interpreter.
    src = str(ROOT / "src")
    env = {**os.environ, "PYTHONPATH": src, "WORKFLOW_OPTIMIZER_SRC": src}
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "workflow_optimizer.proposer"],
        cwd=agent_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        line = line.rstrip()
        log(line)
        found = AGENT_COST.search(line)
        if found and on_cost:
            on_cost(float(found.group(1)), int(found.group(2)))
    process.wait()

    return _collect_programs(agent_dir)


def _stage_agent_dir(cfg, benchmark, agent_dir: pathlib.Path) -> None:
    """Write everything the agent and its dev evaluator read from their cwd.

    Args:
        cfg: The run config, written out whole so the agent's evaluator meters
            candidates exactly as the search does — same models, same prices,
            same per-query limits.
        benchmark: Supplies the task description, grading rule and dev sample.
        agent_dir: The scratch directory to populate.
    """
    (agent_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (agent_dir / "task_spec.json").write_text(json.dumps({
        "description": benchmark.description,
        "check": {"type": benchmark.grader.kind, "task": benchmark.description,
                  "rubric": benchmark.grader.rubric},
        # absolute: the agent reads this from its own scratch directory
        "grader": str(resolve(cfg.task.grader)) if cfg.task.grader else None,
    }))
    (agent_dir / "dev_task.json").write_text(
        json.dumps(benchmark.dev[:cfg.designer.dev_sample_size]))

    skills_dir = agent_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    for name in cfg.designer.skills:
        shutil.copytree(SKILLS_DIR / name, skills_dir / name)


def _round_prompt(cfg, benchmark, round_num: int, context: str) -> str:
    """Build the prompt for one design round.

    Args:
        cfg: The run config, for the model list.
        benchmark: The task being optimized.
        round_num: 1-based round number; round 1 gets the "design a diverse set"
            goal, later rounds the "make it cheaper" one.
        context: `summarize_archive(...)` output, used from round 2 on.

    Returns:
        The full prompt for the agent.
    """
    analysis = benchmark.analysis
    # What the agent needs to return an answer in the right SHAPE, not just with
    # the right value — the grader is strict, so "42 apples" scores 0 on a numeric
    # task. The examples come from the analyzer.
    facts = ("Available models, cheap -> expensive: "
             + ", ".join(m.id for m in cfg.models) + ".\n"
             f"solve() must RETURN its final answer, scored by check="
             f"'{benchmark.grader.kind}'. Nothing is parsed out of prose.")
    if benchmark.grader.kind == "numeric":
        facts += " A numeric answer must BE the number, with no words or units around it."
    elif benchmark.grader.kind == "exact":
        facts += " It is compared against the gold answer exactly, ignoring case."
    if analysis.answer_examples:
        facts += ("\nCorrectly formatted answers for this task look exactly like:\n- "
                  + "\n- ".join(str(e) for e in analysis.answer_examples[:4]))

    goal = (prompts.render("design_goal_initial", description=benchmark.description)
            if round_num == 1 else
            prompts.render("design_goal_improve", description=benchmark.description,
                           archive=context))
    return prompts.render("design_round", goal=goal, facts=facts)


def _collect_programs(agent_dir: pathlib.Path) -> list[dict]:
    """Read the agent's chosen workflows out of its scratch directory.

    Args:
        agent_dir: The directory the agent ran in.

    Returns:
        `{"name", "description", "code"}` dicts from `programs.json`. If the agent
        never wrote that file — it ran out of turns, or errored — any candidate
        `.py` files it left behind are salvaged instead, rather than throwing the
        whole round away.
    """
    programs_file = agent_dir / "programs.json"
    if programs_file.exists():
        data = json.loads(programs_file.read_text())
        return data["programs"] if isinstance(data, dict) else data

    salvaged = []
    for path in sorted(agent_dir.glob("*.py")):
        code = path.read_text()
        if "def solve" in code:
            salvaged.append({"name": path.stem, "description": "(recovered)", "code": code})
    return salvaged
