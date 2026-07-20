#!/usr/bin/env python3
"""Import routerllm's benchmark exports into `benchmarks/`, one folder per task.

Each becomes a self-contained "prompt + dataset" pair the optimizer can run:

    benchmarks/<name>/
      benchmark.yaml   what the task is, how it grades, routerllm's baselines
      data.jsonl       {"question", "answer"} per line

and a thin `config/task/<name>.yaml` so `--task <name>` and the UI pick it up.

Baselines are RECOMPUTED here from routerllm's `joined_14.jsonl` rather than
copied from a table, so they cannot drift from their source. The router rule is
that repo's: escalate to Opus when p < 0.5; the oracle is "either model got it".

    uv run python scripts/import_routerllm_benchmarks.py            # default paths
    uv run python scripts/import_routerllm_benchmarks.py --limit 300
"""
import argparse
import json
import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# How routerllm labels a task's grader -> how we grade it. The three "native"
# tasks are graded by machinery that needs more than the prompt and the answer;
# ifeval has a grader in experiments/, the two code tasks would need a sandboxed
# test harness we do not have, so they are imported but marked unsupported.
GRADERS = {
    "exact":    {"check_type": "exact", "grader": None},
    "contains": {"check_type": "exact", "grader": "benchmarks/_graders/contains.py"},
    "grid":     {"check_type": "exact", "grader": "benchmarks/_graders/grid.py"},
    "judge":    {"check_type": "llm_judge", "grader": None},
}
NATIVE = {
    "ifeval": {"check_type": "exact",
               "grader": "experiments/routerllm_ifeval/grader.py",
               "note": "lm-eval's own constraint checker; needs the routerllm checkout"},
    "humaneval_plus_gen": {"check_type": None, "grader": None,
                           "note": "pass@1 by executing tests — no sandboxed harness here"},
    "mbpp_plus": {"check_type": None, "grader": None,
                  "note": "pass@1 by executing tests — no sandboxed harness here"},
}

# What the workflow designer is told each task is. routerllm's export carries no
# prose description, and the rendered prompts alone don't say what "correct"
# means, so each is stated here.
DESCRIPTIONS = {
    "aa_omniscience": "Answer a short factual question from the AA-Omniscience set. Answers are "
        "brief factual statements; an answer is correct if it states the same fact as the "
        "reference, in any wording.",
    "aime24_gen": "Solve an AIME 2024 competition maths problem. The answer is a single integer.",
    "aime25_gen": "Solve an AIME 2025 competition maths problem. The answer is a single integer.",
    "arc_agi_2": "Solve an ARC-AGI-2 puzzle. The prompt shows example input/output grids of "
        "digits; infer the transformation and produce the output grid for the final input. "
        "Answer with the grid as rows of space-separated digits.",
    "bbeh_gen": "Answer a BIG-Bench Extra Hard reasoning question. The answer is a short exact "
        "string — a word, number, or label — matched exactly.",
    "gpqa_diamond_gen": "Answer a GPQA Diamond graduate-level science multiple-choice question. "
        "Answer with the option letter in parentheses, e.g. (C).",
    "gsm_plus_mini": "Solve a GSM-Plus grade-school maths word problem, including adversarial "
        "perturbations of the original GSM8K wording. The answer is a single number.",
    "hle": "Answer a Humanity's Last Exam question — expert-level, across many fields. Answers "
        "are short and specific; an answer is correct if it matches the reference.",
    "humaneval_plus_gen": "Write a Python function that passes the HumanEval+ test suite for the "
        "given signature and docstring.",
    "ifeval": "Follow the instructions in the prompt exactly. Each prompt states one or more "
        "verifiable formatting constraints. An answer is correct ONLY if it satisfies EVERY "
        "constraint; content quality is not judged, only compliance. The constraints are stated "
        "in the prompt, so a response can be checked against them before being returned.",
    "mbpp_plus": "Write a Python function that passes the MBPP+ test suite for the given task.",
    "minerva_math": "Solve a MATH competition problem. The answer is a mathematical expression "
        "or number, compared for mathematical equivalence.",
    "mmlu_pro": "Answer an MMLU-Pro multiple-choice question across academic and professional "
        "subjects. Answer with the option letter alone, e.g. I.",
    "nq_open_gen": "Answer an open-domain Natural Questions query. The answer is a short factual "
        "span; several phrasings are accepted.",
    "simpleqa": "Answer a SimpleQA short-form factual question. The answer is a brief fact — a "
        "name, date, or number — correct if it matches the reference.",
}


def compute_baselines(joined: pathlib.Path, threshold: float = 0.5) -> dict:
    """Recompute routerllm's four reference accuracies per task.

    Args:
        joined: Path to `joined_14.jsonl` — one row per holdout example with
            `task`, `p` (the router's escalate score), and `h_correct` /
            `o_correct`.
        threshold: Escalate to Opus when `p` is below this.

    Returns:
        `{task: {"n", "haiku", "opus", "router", "oracle"}}`, empty if the file
        is missing.
    """
    if not joined.exists():
        return {}
    rows = [json.loads(line) for line in joined.read_text().splitlines() if line.strip()]
    by_task: dict[str, list] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)

    baselines = {}
    for task, items in by_task.items():
        haiku = [float(bool(r["h_correct"])) for r in items]
        opus = [float(bool(r["o_correct"])) for r in items]
        router = [float(bool(r["o_correct"] if r["p"] < threshold else r["h_correct"]))
                  for r in items]
        oracle = [float(bool(r["h_correct"]) or bool(r["o_correct"])) for r in items]
        baselines[task] = {"n": len(items),
                           "haiku": round(statistics.mean(haiku), 4),
                           "opus": round(statistics.mean(opus), 4),
                           "router": round(statistics.mean(router), 4),
                           "oracle": round(statistics.mean(oracle), 4)}
    return baselines


def main() -> int:
    """Import every exported benchmark. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routerllm", default="/Users/davis/Documents/code/routerllm",
                    help="checkout holding router_data/benchmarks/ and router_runs/")
    ap.add_argument("--limit", type=int, default=200,
                    help="max examples per benchmark (deterministic sample); 0 keeps all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source = pathlib.Path(args.routerllm) / "router_data" / "benchmarks"
    if not source.exists():
        print(f"no benchmark export at {source}", file=sys.stderr)
        return 1

    baselines = compute_baselines(pathlib.Path(args.routerllm) / "router_runs" / "joined_14.jsonl")
    out_root = ROOT / "benchmarks"
    out_root.mkdir(exist_ok=True)
    written = []

    for folder in sorted(source.iterdir()):
        if not folder.is_dir() or not (folder / "dataset.jsonl").exists():
            continue
        name = folder.name
        meta = _read_yaml_ish(folder / "benchmark.yaml")
        grader_kind = meta.get("grader", "exact")

        rows = [json.loads(line) for line in
                (folder / "dataset.jsonl").read_text().splitlines() if line.strip()]
        total = len(rows)
        mapping = NATIVE[name] if name in NATIVE else GRADERS.get(grader_kind, GRADERS["exact"])
        supported = mapping["check_type"] is not None

        # A benchmark we cannot grade is kept only as a sample, for reference —
        # humaneval's answers are whole test harnesses and run to megabytes.
        limit = args.limit if supported else min(args.limit or 20, 20)
        if limit and total > limit:
            rows = random.Random(args.seed).sample(rows, limit)

        target = out_root / name
        target.mkdir(exist_ok=True)
        (target / "data.jsonl").write_text("".join(
            json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n" for r in rows))
        (target / "benchmark.yaml").write_text(_benchmark_yaml(
            name=name, meta=meta, kept=len(rows), total=total, seed=args.seed,
            grader_kind=grader_kind, mapping=mapping, supported=supported,
            baseline=baselines.get(name)))

        # Never overwrite a task config that already exists — several are tuned
        # by hand (ifeval carries a tighter per-query budget and its own answer
        # examples) and regenerating them would quietly undo that.
        kept_existing = False
        if supported:
            task_file = ROOT / "config" / "task" / f"{name}.yaml"
            if task_file.exists() and "import_routerllm_benchmarks" not in task_file.read_text():
                kept_existing = True
            else:
                task_file.write_text(_task_yaml(name, mapping))
        written.append((name, len(rows), total, grader_kind, supported, kept_existing))

    print(f"{len(written)} benchmarks -> {out_root}")
    for name, kept, total, grader_kind, supported, kept_existing in written:
        mark = " " if supported else "!"
        note = ("" if supported else "  (data only — grading unsupported)")
        if kept_existing:
            note = "  (kept the existing hand-written config/task/%s.yaml)" % name
        print(f" {mark} {name:22s} {kept:>5}/{total:<5} {grader_kind}{note}")
    return 0


def _read_yaml_ish(path: pathlib.Path) -> dict:
    """Read routerllm's flat `key: value` benchmark.yaml without a YAML dependency.

    Args:
        path: The file to read.

    Returns:
        Its top-level keys as strings. Missing file gives {}.
    """
    data = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        data[key.strip()] = value.strip()
    return data


def _benchmark_yaml(name, meta, kept, total, seed, grader_kind, mapping, supported, baseline) -> str:
    """Render one benchmark's metadata file. See `main` for the arguments."""
    lines = [
        f"# {name} — imported from routerllm by scripts/import_routerllm_benchmarks.py",
        f"name: {name}",
        "description: >-",
        "  " + DESCRIPTIONS.get(name, "").replace("\n", " "),
        f"source_dataset: {meta.get('source_dataset', 'unknown')}",
        f"num_fewshot: {meta.get('num_fewshot', 'unknown')}",
        f"examples: {kept}",
        f"sampled_from: {total}" + ("" if kept == total else f"   # random.Random({seed}).sample"),
        f"routerllm_grader: {grader_kind}",
        f"grading_supported: {str(supported).lower()}",
    ]
    if mapping.get("note"):
        lines.append(f"grading_note: {mapping['note']}")
    if mapping.get("check_type"):
        lines.append(f"check_type: {mapping['check_type']}")
    if mapping.get("grader"):
        lines.append(f"grader: {mapping['grader']}")
    if baseline:
        lines += ["# routerllm's holdout numbers, recomputed from joined_14.jsonl.",
                  "# Comparable only if graded the same way — see grading_note.",
                  "baselines:"]
        lines += [f"  {k}: {v}" for k, v in baseline.items()]
    return "\n".join(lines) + "\n"


def _task_yaml(name, mapping) -> str:
    """Render the thin `config/task/<name>.yaml` that points at the benchmark."""
    lines = [f"# Generated by scripts/import_routerllm_benchmarks.py from benchmarks/{name}/",
             "task:", f"  name: {name}",
             "  description: >-", "    " + DESCRIPTIONS.get(name, "").replace("\n", " "),
             f"  check_type: {mapping['check_type']}",
             f"  dataset: benchmarks/{name}/data.jsonl"]
    if mapping.get("grader"):
        lines.append(f"  grader: {mapping['grader']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
