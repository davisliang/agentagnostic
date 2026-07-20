#!/usr/bin/env python3
"""The workflow-optimizer UI: a stdlib HTTP server over the run directory.

Serves one static page plus a small JSON API. Every read comes off disk, and
every search runs in its own subprocess, so the server keeps no state: restart
it mid-search and the page picks up exactly where it was.

Usage:
    uv run workflow-optimizer-ui                 # http://127.0.0.1:8770
    uv run workflow-optimizer-ui --port 9000

Binds to localhost by default. Starting a search spends real money, so anything
that can reach this port can spend it.
"""
import argparse
import json
import os
import subprocess
import sys
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .. import runstore
from ..config import load_config
from ..paths import ROOT

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"

# Overrides the New Search form can set, and how to read each one. Anything not
# on this list is rejected: values reach OmegaConf, and the form is not a shell.
FORM_FIELDS = {
    "designer.rounds": int,
    "data.n_examples": int,
    "runtime.concurrency": int,
    "runtime.max_model_calls": int,
    "report.max_cost_per_query": float,
    "report.min_accuracy": float,
}


def start_run(task: str, overrides: dict) -> dict:
    """Create a run directory and launch the pipeline against it.

    Args:
        task: A task config name. Must be one of `runstore.list_tasks()` — it
            names a file path, so an unknown value is rejected rather than
            resolved.
        overrides: Config overrides keyed by dotted path. Only keys in
            FORM_FIELDS are accepted, each coerced to that field's type.

    Returns:
        `{"ok": True, "run_id": ...}`, or `{"ok": False, "error": ...}` if the
        task is unknown, an override is invalid, or the config fails to load.
    """
    if task not in runstore.list_tasks():
        return {"ok": False, "error": f"unknown task: {task}"}

    dotlist = []
    for key, raw in (overrides or {}).items():
        if key not in FORM_FIELDS:
            return {"ok": False, "error": f"unknown setting: {key}"}
        if raw is None or raw == "":
            continue
        try:
            dotlist.append(f"{key}={FORM_FIELDS[key](raw)}")
        except (TypeError, ValueError):
            return {"ok": False, "error": f"bad value for {key}: {raw!r}"}

    try:
        cfg = load_config(task, dotlist)
    except Exception as error:
        return {"ok": False, "error": f"config: {error}"}

    status = runstore.create_run(task, cfg)
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "workflow_optimizer.dashboard.runner", status.run_id],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=True,       # its own process group, so Stop takes the agent too
    )
    runstore.update_status(status.run_id, pid=process.pid)
    return {"ok": True, "run_id": status.run_id}


def run_detail(run_id: str) -> dict:
    """Assemble everything the detail pane shows for one run.

    Args:
        run_id: The run to describe.

    Returns:
        Its status, milestones, candidates (merged from live events and the saved
        result so a running search and a finished one render the same way), the
        log tail, and the frontier. `{"error": "not_found"}` if unknown.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"error": "not_found"}

    events = runstore.read_events(run_id)
    result = runstore.read_result(run_id)

    # While running, candidates come from the event stream; once finished, the
    # saved result carries the code and the test scores too.
    candidates: dict[str, dict] = {}
    for event in events:
        if event.get("event") == "candidate":
            candidates[event["name"]] = {
                "name": event["name"], "description": event.get("description", ""),
                "round": event.get("round"),
                "dev": {"accuracy": event["dev_accuracy"], "cost_per_query": event["dev_cost"],
                        "cached_input_frac": event.get("cached_input_frac", 0.0),
                        "errors": event.get("errors", [])},
                "test": None, "code": "",
            }
        elif event.get("event") == "test_scored" and event["name"] in candidates:
            candidates[event["name"]]["test"] = {
                "accuracy": event["test_accuracy"], "cost_per_query": event["test_cost"]}
    for saved in (result or {}).get("candidates", []):
        merged = candidates.setdefault(saved["name"], {"name": saved["name"]})
        merged.update({k: saved[k] for k in ("description", "code") if k in saved})
        for split in ("dev", "test"):
            if saved.get(split):
                merged[split] = saved[split]

    analyzed = next((e for e in events if e.get("event") == "analyzed"), {})
    return {
        "status": _status_dict(status),
        "analysis": {"check": analyzed.get("check", ""),
                     "description": analyzed.get("description", ""),
                     "judge_status": analyzed.get("judge_status", "")},
        "candidates": list(candidates.values()),
        "frontier": (result or {}).get("frontier", []),
        "log": runstore.read_log(run_id),
        "events": events[-60:],
    }


def _status_dict(status) -> dict:
    """Render a RunStatus as JSON-safe fields.

    Args:
        status: The RunStatus to convert.

    Returns:
        Its fields as a plain dict.
    """
    return asdict(status)


class Handler(BaseHTTPRequestHandler):
    """Routes the UI's requests. One instance per request, as BaseHTTPRequestHandler wants."""

    def _json(self, obj, code: int = 200) -> None:
        """Send a JSON response.

        Args:
            obj: Any JSON-serializable object.
            code: HTTP status code.
        """
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        """Send an HTML response.

        Args:
            html: The page source.
        """
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        """Read and parse the request's JSON body.

        Returns:
            The parsed object, or {} if the body is empty or malformed.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        """Serve the page, the task list, the run list, or one run's detail."""
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                if not STATIC_INDEX.exists():
                    return self._html(f"<h1>500</h1><p>missing {STATIC_INDEX}</p>")
                return self._html(STATIC_INDEX.read_text(encoding="utf-8"))
            if path == "/api/tasks":
                return self._json({"tasks": runstore.list_tasks(),
                                   "fields": sorted(FORM_FIELDS)})
            if path == "/api/runs":
                return self._json({"runs": [_status_dict(s) for s in runstore.list_runs()]})
            if path.startswith("/api/run/"):
                run_id = path[len("/api/run/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                detail = run_detail(run_id)
                return self._json(detail, code=404 if detail.get("error") else 200)
            self.send_error(404, "Not Found")
        except Exception as error:
            self._json({"error": str(error)}, code=500)

    def do_POST(self):
        """Start a search, or stop a running one."""
        path = urlparse(self.path).path
        try:
            if path == "/api/runs":
                body = self._body()
                result = start_run(body.get("task", ""), body.get("overrides", {}))
                return self._json(result, code=200 if result.get("ok") else 400)
            if path.startswith("/api/run/") and path.endswith("/stop"):
                run_id = path[len("/api/run/"):-len("/stop")]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                result = runstore.stop_run(run_id)
                return self._json(result, code=200 if result.get("ok") else 400)
            self.send_error(404, "Not Found")
        except Exception as error:
            self._json({"error": str(error)}, code=500)

    def log_message(self, fmt, *args):
        """Silence the default per-request logging."""


def main() -> None:
    """Serve the UI until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="open a browser on start")
    args = parser.parse_args()

    runstore.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"workflow-optimizer UI → {url}")
    print("Starting a search spends real money. Ctrl-C to stop the server.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
