"""Step 2 — the design agent that writes candidate workflows.

Each round runs a Claude Agent SDK session in a scratch directory, driven by the
three skills under `skills/`. On round 1 it designs a diverse initial set; on
later rounds it is shown the best workflows so far and asked for cheaper ones
that hold accuracy.

The agent runs in a SEPARATE PROCESS (`wopt.proposer`). A notebook kernel that
has called `nest_asyncio.apply()` monkeypatches `asyncio` for the whole session
and breaks `asyncio.run`, even from a worker thread; a subprocess is immune.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from . import prompts
from .paths import ROOT, SKILLS_DIR


def summarize_archive(candidates: list) -> str:
    """The context the agent sees each round: the current dev Pareto frontier,
    plus the code of the most accurate workflow as a base to improve on."""
    from .pareto import pareto_front       # local: pareto imports nothing from here

    if not candidates:
        return ""
    lines = [f"- {c.name}: accuracy {c.accuracy:.2f}, ${c.cost:.5f}/query"
             for c in pareto_front(candidates)]
    best = max(candidates, key=lambda c: c.accuracy)
    lines.append(f"\nMost accurate so far is '{best.name}' (accuracy {best.accuracy:.2f}, "
                 f"${best.cost:.5f}/query). Its code:\n{best.code}")
    return "\n".join(lines)


def run_round(cfg, task, round_num: int, context: str, log=print) -> list[dict]:
    """Run one design round; return the programs it proposed as
    `{name, description, code}` dicts."""
    agent_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"wopt_r{round_num}_"))
    _stage(cfg, task, agent_dir)

    (agent_dir / "proposer_config.json").write_text(json.dumps({
        "model": cfg.designer.model,
        "skills": list(cfg.designer.skills),
        "allowed_tools": list(cfg.designer.allowed_tools),
        "prompt": _round_prompt(cfg, task, round_num, context),
    }))

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "wopt.proposer"], cwd=agent_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        log(line.rstrip())
    process.wait()

    return _collect(agent_dir)


def _stage(cfg, task, agent_dir: pathlib.Path) -> None:
    """Everything the agent and its dev evaluator read from their working dir."""
    # The resolved config, so the evaluator meters candidates exactly as the
    # search does — same models, same prices, same per-query budget.
    (agent_dir / "wopt_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (agent_dir / "task_spec.json").write_text(json.dumps({
        "description": task.description,
        "check": {"type": task.checker.kind, "task": task.description,
                  "rubric": task.checker.rubric},
        "grader": cfg.task.grader,
    }))
    (agent_dir / "dev_task.json").write_text(
        json.dumps(task.dev[:cfg.designer.dev_sample]))

    skills_dir = agent_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    for name in cfg.designer.skills:
        shutil.copytree(SKILLS_DIR / name, skills_dir / name)


def _round_prompt(cfg, task, round_num: int, context: str) -> str:
    spec = task.spec
    # What the agent needs to return an answer in the right SHAPE, not just with
    # the right value — the checker is strict, so "42 apples" scores 0 on a
    # numeric task. The examples come from the analyzer.
    facts = ("Available models, cheap -> expensive: " + ", ".join(_model_ids(cfg)) + ".\n"
             f"solve() must RETURN its final answer, scored by check='{task.checker.kind}'. "
             "Nothing is parsed out of prose.")
    if task.checker.kind == "numeric":
        facts += " A numeric answer must BE the number, with no words or units around it."
    elif task.checker.kind == "exact":
        facts += " It is compared against the gold answer exactly, ignoring case."
    if spec.answer_examples:
        facts += ("\nCorrectly formatted answers for this task look exactly like:\n- "
                  + "\n- ".join(str(e) for e in spec.answer_examples[:4]))

    goal = (prompts.render("design_goal_initial", description=task.description)
            if round_num == 1 else
            prompts.render("design_goal_improve", description=task.description, archive=context))
    return prompts.render("design_round", goal=goal, facts=facts)


def _model_ids(cfg) -> list[str]:
    return [m.id for m in cfg.models]


def _collect(agent_dir: pathlib.Path) -> list[dict]:
    """Read the agent's picks. If it never wrote `programs.json` — it ran out of
    turns, or errored — salvage the candidate files it left on disk rather than
    throwing the round away."""
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
