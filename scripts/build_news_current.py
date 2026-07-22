"""Build the `news_current` benchmark from frozen, sourced, timestamped facts.

The point of this task: its questions are about events from the week it was built
(see `as_of` in sources.json), which POST-DATE the models' training. A workflow
therefore cannot answer from memory — it must retrieve. That closes the
"knowledge bypass" that let single closed-book calls win on fanoutqa, and makes
this a clean test of whether the search will actually use web_search when it has
no choice (and whether decompose/verify structure then helps).

Source of truth: the newest `benchmarks/news_current*/sources.json` — one record
per fact with its question, answer, aliases, event date, category, and the URL it
was verified against, all stamped with the gather date. Editing that file and
re-running this regenerates everything. Answers are resultative (a completed
event's outcome), so they do not drift after the fact.

The benchmark's NAME carries the freeze date (`news_current_<as_of>`), so the
name itself says when the data goes stale and should be recycled. Refreshing
sources.json with a new `as_of` and re-running writes a NEW stamped benchmark
beside the old one rather than silently replacing it.

    uv run python scripts/build_news_current.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def newest_sources(pattern: str) -> pathlib.Path:
    """Find the most recently frozen sources.json for a benchmark family.

    Args:
        pattern: A glob over benchmark folder names, e.g. "news_current*".

    Returns:
        The sources.json with the latest `as_of` — normally the only one, but a
        refresh leaves the old freeze in place, so pick by date, not by luck.
    """
    found = list(ROOT.glob(f"benchmarks/{pattern}/sources.json"))
    if not found:
        raise SystemExit(f"no sources.json under benchmarks/{pattern}/")
    return max(found, key=lambda p: json.loads(p.read_text())["as_of"])

# The description the design agent sees (config/task/*.yaml -> the design prompt).
# Deliberately NEUTRAL: it states only the answer format and reveals nothing about
# recency or that retrieval is needed. Whether the search discovers it must use
# web_search is the experiment, so the task must not spoon-feed it.
TASK_DESCRIPTION = ("Answer the question with a single short, specific answer — a name, "
                    "number, place, or date — and nothing else.")

# Human-only blurb for the UI benchmark picker; never reaches the model.
BENCH_DESCRIPTION = ("Short-answer QA over recent (post-cutoff) news events. The "
                     "model-facing task description is kept neutral so the search must "
                     "discover for itself that the answers require web retrieval.")

ANSWER_EXAMPLES = ["Spain", "Kimi Antonelli", "Ryan Fox", "$12.56 billion"]


def main() -> None:
    """Write data.jsonl, the benchmark descriptor, and the task config."""
    src_path = newest_sources("news_current*")
    src = json.loads(src_path.read_text())
    as_of, window, items = src["as_of"], src["window"], src["items"]
    name = f"news_current_{as_of.replace('-', '')}"   # the name says when it goes stale

    rows = []
    for item in items:
        golds = [item["answer"], *[a for a in item.get("aliases", []) if a]]
        # Only question + answer reach the model. Provenance (as_of, event_date,
        # source_url, category) stays in sources.json, so the sampled examples the
        # design agent is shown reveal neither recency nor the news source.
        rows.append({"question": item["question"], "answer": golds})

    bench_dir = ROOT / "benchmarks" / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    if src_path.parent != bench_dir:       # a refreshed freeze lands in its own dir
        (bench_dir / "sources.json").write_text(src_path.read_text())
    (bench_dir / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (bench_dir / "benchmark.yaml").write_text(
        f"# {name} — recent-news QA, built by scripts/build_news_current.py\n"
        f"# Timestamped: answers verified as of {as_of} (events from {window}).\n"
        f"name: {name}\n"
        f"description: >-\n  {BENCH_DESCRIPTION}\n"
        "source_dataset: news\n"
        f"as_of: {as_of}\n"
        f"window: {window!r}\n"
        f"examples: {len(rows)}\n"
        "grading_supported: true\n"
        "check_type: exact\n"
        "routerllm_grader: contains\n"
        "grader: benchmarks/_graders/contains.py\n")

    examples_yaml = "\n".join(f"    - {json.dumps(e)}" for e in ANSWER_EXAMPLES)
    (ROOT / "config" / "task" / f"{name}.yaml").write_text(
        "# Recent-news QA — post-cutoff facts, so the workflow must retrieve, not recall.\n"
        "# NOTE (human context only; these comments never reach the model): the task\n"
        "# description below is kept NEUTRAL on purpose — it must not tell the design\n"
        "# agent that the answers are recent or that web search is needed, so that the\n"
        "# search discovering retrieval on its own is a real result, not a hint.\n"
        f"# Timestamped: answers as of {as_of} (events from {window}).\n"
        "task:\n"
        f"  name: {name}\n"
        f"  description: >-\n    {TASK_DESCRIPTION}\n"
        "  check_type: exact\n"
        f"  dataset: benchmarks/{name}/data.jsonl\n"
        "  grader: benchmarks/_graders/contains.py\n"
        "  answer_examples:\n"
        f"{examples_yaml}\n"
        "runtime:\n"
        "  tools: [web_search, web_fetch]   # post-cutoff facts: retrieval is mandatory\n"
        "data:\n"
        f"  n_examples: {len(rows)}     # use all of them ({len(rows)} facts)\n")

    print(f"wrote {len(rows)} facts (as of {as_of}) to {bench_dir/'data.jsonl'}")
    print(f"wrote {bench_dir/'benchmark.yaml'} and config/task/{name}.yaml")


if __name__ == "__main__":
    main()
