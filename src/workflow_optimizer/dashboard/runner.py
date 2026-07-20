"""Execute one search into a run directory. Entry point for the UI's subprocess.

The server never runs a search in its own process: a search takes minutes,
spawns the design agent, and must survive the request that started it. This
module is what the server spawns, and everything it learns it writes to the run
directory — status, milestones, raw log, final result — so the UI reads state
from disk rather than from server memory.

    python -m workflow_optimizer.dashboard.runner <run_id>
"""
import sys
import time
import traceback

from .. import analysis, report, runstore
from ..config import load_resolved
from ..optimizer import optimize
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
        if kind == "round_start":
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
            "n_dev": len(benchmark.dev), "n_test": len(benchmark.test)})

        search = optimize(cfg, benchmark, session.evaluator(benchmark.grader),
                          log=log, on_event=emit)

        report.summarize(search, cfg, log=log)
        report.save(search, cfg, out_dir=directory)
        # report.save names the file after the task; the UI wants one known name.
        saved = directory / f"{cfg.task.name}.json"
        if saved.exists():
            saved.replace(directory / "result.json")

        runstore.update_status(run_id, phase="done", state="done", ended_at=time.time())
        runstore.append_event(run_id, {"event": "done", "n_finalists": len(search.finalists)})
        log(f"\nrun complete: {len(search.finalists)} finalists")
        return 0

    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        log("\n" + traceback.format_exc())
        runstore.update_status(run_id, phase="failed", state="failed",
                               ended_at=time.time(), error=message)
        runstore.append_event(run_id, {"event": "failed", "error": message})
        return 1
    finally:
        log_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m workflow_optimizer.dashboard.runner <run_id>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
