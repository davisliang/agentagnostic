"""Where a run's state lives on disk.

Everything the UI shows is read from these files, never from server memory. A
search takes minutes and spawns subprocesses, so the state has to outlive both
the request that started it and the server process itself — restart the server
mid-run and the page still shows the run progressing.

One directory per run:

    runs/<run_id>/
      config.yaml     the fully resolved config the run was started with
      status.json     phase, counts, timings — the whole header in one read
      events.jsonl    one JSON object per milestone, append-only
      log.txt         raw stdout of the pipeline, including the design agent's
      result.json     the finished search (report.save output)
"""
import json
import os
import re
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from .paths import CONFIG_DIR, ROOT

RUNS_DIR = ROOT / "runs"

# Phases a run moves through, in order. The UI renders these as pills.
PHASES = ["queued", "analyzing", "designing", "ranking", "done"]


@dataclass
class RunStatus:
    """The header of one run — everything the run list needs in a single read.

    Attributes:
        run_id: Directory name, e.g. "gsm8k-20260720-143012".
        task: Task config name the run was started with.
        phase: One of PHASES, or "failed" / "stopped".
        state: "running", "done", "failed" or "stopped".
        started_at: Unix timestamp when the run began.
        ended_at: Unix timestamp when it finished, or None while running.
        pid: OS process id of the pipeline subprocess, for stopping it.
        round: Design round in progress, 1-based.
        rounds: Total rounds configured.
        n_candidates: Candidates scored on dev so far.
        n_dev: Dev split size, once known.
        n_test: Test split size, once known.
        error: Failure message, when state is "failed".
    """
    run_id: str
    task: str
    phase: str = "queued"
    state: str = "running"
    started_at: float = 0.0
    ended_at: Optional[float] = None
    pid: Optional[int] = None
    round: int = 0
    rounds: int = 0
    n_candidates: int = 0
    n_dev: int = 0
    n_test: int = 0
    error: str = ""


def list_tasks() -> list[str]:
    """List the task configs a run can be started from.

    Returns:
        Sorted names of `config/task/*.yaml`, without the extension.
    """
    return sorted(p.stem for p in (CONFIG_DIR / "task").glob("*.yaml"))


def new_run_id(task: str) -> str:
    """Build a fresh run id.

    Args:
        task: The task config name.

    Returns:
        "<task>-<YYYYMMDD>-<HHMMSS>", unique to the second. A suffix is added if
        that directory somehow already exists.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    run_id = f"{task}-{stamp}"
    suffix = 2
    while (RUNS_DIR / run_id).exists():
        run_id = f"{task}-{stamp}-{suffix}"
        suffix += 1
    return run_id


def is_valid_run_id(run_id: str) -> bool:
    """Check a run id is a plain directory name, not a path.

    The id arrives from HTTP, so this is what stops `../../etc` reaching the
    filesystem.

    Args:
        run_id: The candidate id.

    Returns:
        True if it is safe to use as a directory name.
    """
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]{1,120}", run_id)) and ".." not in run_id


def run_dir(run_id: str) -> Path:
    """The directory holding one run's files.

    Args:
        run_id: A validated run id.

    Returns:
        Its path under `runs/`.

    Raises:
        ValueError: The id is not a safe directory name.
    """
    if not is_valid_run_id(run_id):
        raise ValueError(f"unsafe run id: {run_id!r}")
    return RUNS_DIR / run_id


def create_run(task: str, cfg) -> RunStatus:
    """Lay out a new run's directory and write its resolved config.

    Args:
        task: The task config name.
        cfg: The fully resolved config for the run.

    Returns:
        The initial RunStatus, already written to disk.
    """
    status = RunStatus(run_id=new_run_id(task), task=task, started_at=time.time(),
                       rounds=int(cfg.designer.rounds))
    directory = run_dir(status.run_id)
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (directory / "log.txt").write_text("")
    (directory / "events.jsonl").write_text("")
    write_status(status)
    return status


def write_status(status: RunStatus) -> None:
    """Persist a run's status, replacing what was there.

    Args:
        status: The status to write.
    """
    path = run_dir(status.run_id) / "status.json"
    path.write_text(json.dumps(asdict(status), indent=1))


def read_status(run_id: str) -> Optional[RunStatus]:
    """Read one run's status.

    Args:
        run_id: The run to read.

    Returns:
        Its RunStatus, or None if the run or its status file is missing or
        unreadable.
    """
    path = run_dir(run_id) / "status.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    known = {f for f in RunStatus.__dataclass_fields__}
    return RunStatus(**{k: v for k, v in data.items() if k in known})


def update_status(run_id: str, **fields) -> Optional[RunStatus]:
    """Change some fields of a run's status.

    Args:
        run_id: The run to update.
        **fields: RunStatus attributes to set.

    Returns:
        The updated RunStatus, or None if the run has no status file.
    """
    status = read_status(run_id)
    if status is None:
        return None
    for key, value in fields.items():
        setattr(status, key, value)
    write_status(status)
    return status


def append_event(run_id: str, event: dict) -> None:
    """Append one milestone to a run's event log.

    Args:
        run_id: The run to append to.
        event: A JSON-serializable milestone. A "t" timestamp is added.
    """
    event = {"t": time.time(), **event}
    with open(run_dir(run_id) / "events.jsonl", "a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def read_events(run_id: str) -> list[dict]:
    """Read a run's milestones.

    Args:
        run_id: The run to read.

    Returns:
        Every event in order. Malformed trailing lines — possible while the file
        is being appended to — are skipped.
    """
    path = run_dir(run_id) / "events.jsonl"
    if not path.exists():
        return []
    events = []
    for line in path.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def read_log(run_id: str, max_lines: int = 400) -> str:
    """Read the tail of a run's raw output.

    Args:
        run_id: The run to read.
        max_lines: How many trailing lines to return.

    Returns:
        The last `max_lines` lines, or "" if there is no log yet.
    """
    path = run_dir(run_id) / "log.txt"
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def read_result(run_id: str) -> Optional[dict]:
    """Read a finished run's saved search.

    Args:
        run_id: The run to read.

    Returns:
        The parsed `result.json`, or None if the run hasn't produced one.
    """
    path = run_dir(run_id) / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_runs() -> list[RunStatus]:
    """List every run on disk, newest first.

    Also reconciles state: a run marked "running" whose process is gone — the
    machine restarted, the process was killed — is corrected to "failed", so the
    UI never shows a run that stopped existing as still going.

    Returns:
        RunStatus for each run directory, newest first.
    """
    if not RUNS_DIR.exists():
        return []
    statuses = []
    for directory in RUNS_DIR.iterdir():
        if not directory.is_dir() or not is_valid_run_id(directory.name):
            continue
        status = read_status(directory.name)
        if status is None:
            continue
        if status.state == "running" and not _process_alive(status.pid):
            status = update_status(status.run_id, state="failed", phase="failed",
                                   ended_at=time.time(),
                                   error=status.error or "process is no longer running")
        statuses.append(status)
    return sorted(statuses, key=lambda s: s.started_at, reverse=True)


def stop_run(run_id: str) -> dict:
    """Stop a running search.

    Signals the whole process group, so the design agent's own subprocess goes
    down with the pipeline rather than being orphaned.

    Args:
        run_id: The run to stop.

    Returns:
        `{"ok": True, "state": "stopped"}`, or `{"ok": False, "error": ...}` if
        the run is unknown or already finished.
    """
    status = read_status(run_id)
    if status is None:
        return {"ok": False, "error": "unknown run"}
    if status.state != "running":
        return {"ok": False, "error": f"run is already {status.state}"}
    if not status.pid:
        return {"ok": False, "error": "no process to stop"}
    try:
        os.killpg(os.getpgid(status.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as error:
        return {"ok": False, "error": f"could not stop: {error}"}
    update_status(run_id, state="stopped", phase="stopped", ended_at=time.time())
    append_event(run_id, {"event": "stopped"})
    return {"ok": True, "state": "stopped"}


def _process_alive(pid: Optional[int]) -> bool:
    """Check whether a process is still running.

    Args:
        pid: Process id, or None.

    Returns:
        True if a process with that id exists.
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    return True
