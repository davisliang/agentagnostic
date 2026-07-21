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
import signal
import subprocess
import sys
import webbrowser
from dataclasses import asdict
from typing import Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import costs, runstore
from ..config import load_config
from .. import paths
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


def parse_dataset(text: str) -> tuple[list, str]:
    """Read an uploaded dataset into examples.

    Accepts JSONL or a JSON array, with the answer under "answer", "target" or
    "gold" — the three spellings the exports around here use.

    Args:
        text: The uploaded file's contents.

    Returns:
        `(examples, "")` on success, or `([], reason)` if it can't be read.
    """
    text = (text or "").strip()
    if not text:
        return [], "the dataset is empty"

    rows = []
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as error:
            return [], f"not valid JSON: {error}"
    else:
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                return [], f"line {number} is not valid JSON: {error}"

    examples = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            return [], f"row {number} is not an object"
        question = row.get("question") or row.get("input") or row.get("prompt")
        answer = row.get("answer", row.get("target", row.get("gold")))
        if question is None or answer is None:
            return [], (f"row {number} needs a question and an answer — saw keys "
                        f"{sorted(row)[:6]}")
        examples.append({**row, "question": str(question), "answer": answer})

    if len(examples) < 2:
        return [], "at least 2 examples are needed, to split dev from test"
    return examples, ""


ALLOWED_WORKFLOW_TOOLS = ["code_execution", "web_search"]


def start_run(task: str, overrides: dict, prompt: str = "", dataset_text: str = "",
              tools: list = None) -> dict:
    """Create a run directory and launch the pipeline against it.

    Args:
        task: A task config name. Must be one of `runstore.list_tasks()` — it
            names a file path, so an unknown value is rejected rather than
            resolved. Ignored when `prompt` is given.
        overrides: Config overrides keyed by dotted path. Only keys in
            FORM_FIELDS are accepted, each coerced to that field's type.
        prompt: A free-text task description. Starts an ad-hoc search instead of
            using a task file; the analyzer infers the grading rule from it.
        dataset_text: An uploaded JSONL or JSON array of examples. Optional — a
            free-text task with no data generates its own.
        tools: Server-side tools workflows may use, a subset of
            ALLOWED_WORKFLOW_TOOLS. None leaves the task's config default; a list
            (including []) overrides it, so [] forbids all tools.

    Returns:
        `{"ok": True, "run_id": ...}`, or `{"ok": False, "error": ...}`.
    """
    if tools is not None:
        bad = [x for x in tools if x not in ALLOWED_WORKFLOW_TOOLS]
        if bad:
            return {"ok": False, "error": f"unknown tool(s): {', '.join(bad)}"}
    freetext = bool(prompt and prompt.strip())
    if not freetext and task not in runstore.list_tasks():
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

    examples = []
    if dataset_text:
        examples, reason = parse_dataset(dataset_text)
        if reason:
            return {"ok": False, "error": f"dataset: {reason}"}

    try:
        if freetext:
            # Nothing from the form names a file: the task is built in memory,
            # and any uploaded data is written inside the run's own directory.
            cfg = load_config("", dotlist)
            cfg.task.name = "custom"
            cfg.task.seed_prompt = prompt.strip()
        else:
            cfg = load_config(task, dotlist)
    except Exception as error:
        return {"ok": False, "error": f"config: {error}"}

    if tools is not None:
        cfg.runtime.tools = list(tools)

    status = runstore.create_run(cfg.task.name, cfg)
    if examples:
        data_file = runstore.run_dir(status.run_id) / "dataset.jsonl"
        data_file.write_text("".join(json.dumps(e) + "\n" for e in examples))
        cfg.task.dataset = str(data_file)
    if examples or tools is not None:
        runstore.write_config(status.run_id, cfg)
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "workflow_optimizer.dashboard.runner", status.run_id],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=True,       # its own process group, so Stop takes the agent too
    )
    runstore.update_status(status.run_id, pid=process.pid)
    return {"ok": True, "run_id": status.run_id}


def compare_runs() -> dict:
    """Gather every scored candidate across every run, for the comparison chart.

    One point per candidate per run, so searches on the same benchmark — or on
    different ones — can be read on the same accuracy/cost axes.

    Returns:
        `{"points": [...], "baselines": {task: {...}}}`. Each point carries the
        run it came from, its task, the split its numbers are from, accuracy,
        cost, whether it was on that run's frontier, and its description.
    """
    points, tasks = [], set()
    for status in runstore.list_runs():
        detail_events = runstore.read_events(status.run_id)
        result = runstore.read_result(status.run_id) or {}
        frontier = set(result.get("frontier", []))
        seen = {}
        for event in detail_events:
            if event.get("event") == "candidate":
                seen[event["name"]] = {"dev": (event["dev_accuracy"], event["dev_cost"]),
                                       "description": event.get("description", "")}
            elif event.get("event") == "test_scored" and event["name"] in seen:
                seen[event["name"]]["test"] = (event["test_accuracy"], event["test_cost"])
        for name, entry in seen.items():
            split = "test" if "test" in entry else "dev"
            accuracy, cost = entry.get("test") or entry["dev"]
            points.append({"run_id": status.run_id, "task": status.task, "name": name,
                           "split": split, "accuracy": accuracy, "cost": cost,
                           "frontier": name in frontier,
                           "description": entry.get("description", "")})
        if seen:
            tasks.add(status.task)

    return {"points": points,
            "baselines": {task: runstore.baselines_for(task) for task in sorted(tasks)}}


def estimate_cost(task: str, overrides: dict, freetext: bool = False,
                  has_dataset: bool = False) -> dict:
    """Estimate what the form's current settings would cost to run.

    Args:
        task: The selected task config name, or "" for a free-text task.
        overrides: The form's settings, same keys as `start_run` accepts.
        freetext: Whether this is a described-in-prose task.
        has_dataset: Whether a dataset was uploaded, so none is generated.

    Returns:
        The Estimate as a dict, or `{"error": ...}` if the settings don't load —
        the same rejections `start_run` would give, surfaced before spending.
    """
    dotlist = []
    for key, raw in (overrides or {}).items():
        if key not in FORM_FIELDS or raw in (None, ""):
            continue
        try:
            dotlist.append(f"{key}={FORM_FIELDS[key](raw)}")
        except (TypeError, ValueError):
            return {"error": f"bad value for {key}: {raw!r}"}
    if not freetext and task not in runstore.list_tasks():
        return {"error": f"unknown task: {task}"}

    try:
        cfg = load_config("" if freetext else task, dotlist)
    except Exception as error:
        return {"error": f"config: {error}"}

    history = costs.observed([(s.task, runstore.read_events(s.run_id))
                              for s in runstore.list_runs()])
    generates = not (has_dataset or (not freetext and bool(cfg.task.dataset)))
    guess = costs.estimate(cfg, history, generates_data=generates,
                           judged=None if not freetext else False,
                           available=_dataset_size(cfg))
    return {"low": guess.low, "expected": guess.expected, "high": guess.high,
            "breakdown": guess.breakdown, "assumptions": guess.assumptions,
            "based_on_runs": guess.based_on_runs}


def probe_and_estimate(task: str, overrides: dict) -> dict:
    """Measure this task with a few real calls, then estimate from that.

    Costs a few cents and takes seconds. Unlike history, it works on a task
    nobody has ever run, and it measures the two things that actually drive cost:
    how many tokens a call on this task takes, and whether the cheap model can do
    the work at all.

    Args:
        task: A task config name. Free-text tasks cannot be probed — there are no
            examples until the dataset is generated.
        overrides: The form's settings.

    Returns:
        The estimate, with a "probe" block describing the measurement, or
        `{"error": ...}`.
    """
    from .. import analysis
    from ..session import Session

    base = estimate_cost(task, overrides)
    if base.get("error"):
        return base

    dotlist = [f"{k}={FORM_FIELDS[k](v)}" for k, v in (overrides or {}).items()
               if k in FORM_FIELDS and v not in (None, "")]
    cfg = load_config(task, dotlist)
    if not cfg.task.dataset:
        return {**base, "probe": {"skipped": "this task generates its own examples, "
                                  "so there is nothing to probe until it runs"}}

    session = Session.from_config(cfg)
    try:
        benchmark = analysis.build_benchmark(cfg, session.client, log=lambda *a: None)
    except Exception as error:
        return {"error": f"could not load the task: {error}"}

    measured = costs.run_probe(cfg, session.client, benchmark.grader, benchmark.dev, n=3)
    if not measured.n:
        return {**base, "probe": {"skipped": "every probe call failed — the API may be "
                                  "busy; the estimate below is from defaults"}}

    history = costs.observed([(s.task, runstore.read_events(s.run_id))
                              for s in runstore.list_runs()])
    guess = costs.estimate(cfg, history, generates_data=False, probe=measured,
                           available=_dataset_size(cfg))
    return {"low": guess.low, "expected": guess.expected, "high": guess.high,
            "breakdown": guess.breakdown, "assumptions": guess.assumptions,
            "based_on_runs": guess.based_on_runs,
            "probe": {"n": measured.n, "model": measured.model,
                      "input_tokens": measured.input_tokens,
                      "output_tokens": measured.output_tokens,
                      "accuracy": measured.accuracy, "cost": measured.cost}}


def compare_examples(run_id: str, split: str = "dev", limit: int = 200) -> dict:
    """Line every workflow's answer to the same example up side by side.

    A per-candidate accuracy says which workflow won; it doesn't say where they
    differed, which is the thing worth reading. Aligning answers by example shows
    exactly which questions separate a cheap workflow from an expensive one — and
    whether the expensive one is right for a reason or just lucky.

    Args:
        run_id: The run to read.
        split: "dev" or "test".
        limit: Most rows to return.

    Returns:
        `{"candidates": [names], "rows": [...], "split": ...}`. Each row carries
        the question, the gold answer, one cell per candidate, and `spread` —
        the gap between the best and worst score on that example, so the rows
        that discriminate can be found first.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"error": "not_found"}

    names = [e["name"] for e in runstore.read_events(run_id)
             if e.get("event") == "candidate"]
    traces = [(name, runstore.read_trace(run_id, name, split)) for name in names]
    traces = [(name, trace) for name, trace in traces if trace]
    if not traces:
        return {"candidates": [], "rows": [], "split": split,
                "note": f"no {split} traces recorded for this run"}

    # Align on the question text. Candidates are scored over the same split in the
    # same order, but matching on content survives a reordering.
    rows: dict[str, dict] = {}
    for name, trace in traces:
        for record in trace["records"]:
            question = record["question"]["text"]
            row = rows.setdefault(question, {
                "question": question, "gold": record["gold"]["text"], "cells": {}})
            row["cells"][name] = {
                "answer": record["answer"]["text"], "clipped": record["answer"]["clipped"],
                "score": record["score"], "error": record["error"],
                "cost": record["cost"], "calls": len(record["calls"])}

    candidates = [name for name, _ in traces]
    out = []
    for row in rows.values():
        scores = [c["score"] for c in row["cells"].values()]
        out.append({**row,
                    "cells": [row["cells"].get(name) for name in candidates],
                    "spread": (max(scores) - min(scores)) if scores else 0.0})
    # Most-disagreed first: a row every workflow got right teaches nothing.
    out.sort(key=lambda r: (-r["spread"], r["question"]))
    return {"candidates": candidates, "rows": out[:limit], "split": split,
            "n_rows": len(out)}


def run_detail(run_id: str, log_lines: int = 400) -> dict:
    """Assemble everything the detail pane shows for one run.

    Args:
        run_id: The run to describe.
        log_lines: How many trailing log lines to include. The UI asks for more
            when the reader opens the full log.

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
                     "judge_status": analyzed.get("judge_status", ""),
                     "rubric": analyzed.get("rubric", ""),
                     "answer_examples": analyzed.get("answer_examples", []),
                     "dev_sample": analyzed.get("dev_sample", []),
                     "test_sample": analyzed.get("test_sample", [])},
        "candidates": list(candidates.values()),
        "frontier": (result or {}).get("frontier", []),
        "log": runstore.read_log(run_id, max_lines=log_lines),
        "config": runstore.read_config_text(run_id),
        "events": events,
    }


def _dataset_size(cfg) -> Optional[int]:
    """Count the examples a task's dataset actually holds.

    The run scores `min(n_examples, this)`, so an estimate that assumes
    `n_examples` can be out by the ratio — a request for 40 against a
    200-example benchmark understated a run fivefold before `n_examples` was
    applied to loaded data.

    Args:
        cfg: The run config.

    Returns:
        The row count, or None if the task generates its own examples or the file
        cannot be read.
    """
    if not cfg.task.dataset:
        return None
    try:
        with open(paths.resolve(cfg.task.dataset)) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


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
                                   "benchmarks": runstore.list_benchmarks(),
                                   "fields": sorted(FORM_FIELDS),
                                   "workflow_tools": ALLOWED_WORKFLOW_TOOLS,
                                   "default_tools": list(load_config().runtime.tools)})
            if path == "/api/compare":
                return self._json(compare_runs())
            if path == "/api/estimate":
                params = parse_qs(urlparse(self.path).query)
                overrides = {k: v[0] for k, v in params.items()
                             if k in FORM_FIELDS and v and v[0] != ""}
                return self._json(estimate_cost(
                    (params.get("task") or [""])[0],
                    overrides,
                    freetext=(params.get("freetext") or ["0"])[0] == "1",
                    has_dataset=(params.get("has_dataset") or ["0"])[0] == "1"))
            if path == "/api/runs":
                return self._json({"runs": [_status_dict(s) for s in runstore.list_runs()]})
            if path.startswith("/api/answers/"):
                run_id = path[len("/api/answers/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                split = (params.get("split") or ["dev"])[0]
                if split not in ("dev", "test"):
                    return self._json({"error": "split must be dev or test"}, code=400)
                result = compare_examples(run_id, split)
                return self._json(result, code=404 if result.get("error") else 200)
            if path.startswith("/api/trace/"):
                run_id = path[len("/api/trace/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                name = (params.get("name") or [""])[0]
                split = (params.get("split") or ["dev"])[0]
                if split not in ("dev", "test"):
                    return self._json({"error": "split must be dev or test"}, code=400)
                trace = runstore.read_trace(run_id, name, split)
                return self._json(trace or {"error": "no trace recorded"},
                                  code=200 if trace else 404)
            if path.startswith("/api/run/"):
                run_id = path[len("/api/run/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                lines = min(int((params.get("log_lines") or ["400"])[0] or 400), 20000)
                detail = run_detail(run_id, log_lines=lines)
                return self._json(detail, code=404 if detail.get("error") else 200)
            self.send_error(404, "Not Found")
        except Exception as error:
            self._json({"error": str(error)}, code=500)

    def do_POST(self):
        """Start a search, or stop a running one."""
        path = urlparse(self.path).path
        try:
            if path == "/api/probe":
                body = self._body()
                return self._json(probe_and_estimate(body.get("task", ""),
                                                     body.get("overrides", {})))
            if path == "/api/runs":
                body = self._body()
                result = start_run(body.get("task", ""), body.get("overrides", {}),
                                   prompt=body.get("prompt", ""),
                                   dataset_text=body.get("dataset", ""),
                                   tools=body.get("tools"))
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

    # Reap finished searches automatically. Without this a completed run stays in
    # the process table as a zombie, which still answers `kill(pid, 0)` — so the
    # run list would report a finished run as still running. The server never
    # calls wait() itself, so nothing here depends on collecting exit statuses.
    if hasattr(signal, "SIGCHLD"):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)

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
