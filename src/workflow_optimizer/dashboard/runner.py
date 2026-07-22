"""Execute one search into a run directory. Entry point for the UI's subprocess.

The server never runs a search in its own process: a search takes minutes,
spawns the design agent, and must survive the request that started it. This
module is what the server spawns, and everything it learns it writes to the run
directory — status, milestones, raw log, final result — so the UI reads state
from disk rather than from server memory.

    python -m workflow_optimizer.dashboard.runner <run_id>
"""
import signal
import sys
import time
import traceback

from .. import analysis, report, runstore
from ..config import load_resolved
from ..optimizer import Search, optimize
from ..session import Session


def main(run_id: str) -> int:
    """Run the full pipeline for an already-created run directory.

    Args:
        run_id: The run to execute. Its directory must already hold `config.yaml`
            (see `runstore.create_run`).

    Returns:
        0 on success, 1 if the search failed. The failure message is recorded in
        the run's status either way, so the UI can show it.
    """
    directory = runstore.run_dir(run_id)
    log_file = open(directory / "log.txt", "a", buffering=1)

    def log(*parts) -> None:
        """Write one line to the run's raw log, and to stdout for `journalctl`-style tailing."""
        line = " ".join(str(p) for p in parts)
        log_file.write(line + "\n")
        print(line, flush=True)

    def emit(event: dict) -> None:
        """Record a milestone and reflect it in the run's status header."""
        runstore.append_event(run_id, event)
        kind = event.get("event")
        if kind == "researching":
            runstore.update_status(run_id, phase="researching")
        elif kind == "round_start":
            runstore.update_status(run_id, phase="designing", round=event["round"])
        elif kind == "candidate":
            status = runstore.read_status(run_id)
            runstore.update_status(run_id, n_candidates=(status.n_candidates + 1 if status else 1))
        elif kind == "ranking":
            runstore.update_status(run_id, phase="ranking")

    try:
        cfg = load_resolved(directory / "config.yaml")
        session = Session.from_config(cfg)

        runstore.update_status(run_id, phase="analyzing")
        runstore.append_event(run_id, {"event": "analyzing"})
        benchmark = analysis.build_benchmark(cfg, session.client, log=log)
        runstore.update_status(run_id, n_dev=len(benchmark.dev), n_test=len(benchmark.test))
        runstore.append_event(run_id, {
            "event": "analyzed", "check": benchmark.grader.kind,
            "description": benchmark.description, "judge_status": benchmark.judge_status,
            "rubric": benchmark.grader.rubric,
            "answer_examples": list(benchmark.analysis.answer_examples),
            "dev_sample": benchmark.dev[:3], "test_sample": benchmark.test[:3],
            "n_dev": len(benchmark.dev), "n_test": len(benchmark.test)})

        def keep_trace(candidate, split, score) -> None:
            """Persist every model call the candidate made, for the verbose view."""
            runstore.write_trace(run_id, candidate.name, split, score.records)

        def keep_research(notes) -> None:
            """Persist the research phase's notes, so the UI can show them."""
            runstore.write_research(run_id, notes)

        # The caller owns the Search, so the archive survives an interrupt.
        search = Search()
        _save_on_exit(run_id, cfg, search, log)
        optimize(cfg, benchmark, session.evaluator(benchmark.grader),
                 log=log, on_event=emit, on_scored=keep_trace,
                 on_research=keep_research,
                 skills_dir=directory / "skills",   # keep the run's working skills for inspection
                 search=search)

        report.summarize(search, cfg, log=log)
        _write_result(run_id, cfg, search)

        runstore.update_status(run_id, phase="done", state="done", ended_at=time.time())
        runstore.append_event(run_id, {"event": "done", "n_finalists": len(search.finalists)})
        log(f"\nrun complete: {len(search.finalists)} finalists")
        return 0

    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        log("\n" + traceback.format_exc())
        # Keep whatever was already scored — a search that fails in round 3 still
        # spent real money on rounds 1 and 2.
        saved = _write_result(run_id, cfg, locals().get("search"))
        runstore.update_status(run_id, phase="failed", state="failed",
                               ended_at=time.time(), error=message)
        runstore.append_event(run_id, {"event": "failed", "error": message,
                                       "kept_candidates": saved})
        return 1
    finally:
        log_file.close()


def _write_result(run_id: str, cfg, search) -> int:
    """Write the run's result file from whatever has been scored so far.

    Args:
        run_id: The run to write for.
        cfg: Its config.
        search: The Search, possibly partial. None or empty writes nothing.

    Returns:
        How many candidates were saved.
    """
    if search is None or not search.archive:
        return 0
    directory = runstore.run_dir(run_id)
    report.save(search, cfg, out_dir=directory)
    # report.save names the file after the task; the UI wants one known name.
    saved = directory / f"{cfg.task.name}.json"
    if saved.exists():
        saved.replace(directory / "result.json")
    return len(search.archive)


def _save_on_exit(run_id: str, cfg, search, log) -> None:
    """Save partial results if the run is stopped from the UI.

    Stop sends SIGTERM to the process group. Left to the default handler the
    process dies where it stands and everything scored so far is lost — one
    stopped ARC run threw away six scored candidates and about $260 of work.

    Args:
        run_id: The run being executed.
        cfg: Its config.
        search: The Search being filled in.
        log: Where to note what was kept.
    """
    def handler(signum, frame):
        kept = _write_result(run_id, cfg, search)
        log(f"\nstopped — kept {kept} scored candidate(s)")
        runstore.append_event(run_id, {"event": "stopped", "kept_candidates": kept})
        runstore.update_status(run_id, phase="stopped", state="stopped",
                               ended_at=time.time())
        sys.exit(143)

    signal.signal(signal.SIGTERM, handler)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m workflow_optimizer.dashboard.runner <run_id>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
