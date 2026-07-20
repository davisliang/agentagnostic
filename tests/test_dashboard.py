"""The UI's logic, without a browser or an API key.

Covers what the run store promises (state survives on disk, ids can't escape
`runs/`) and what the server rejects, since the form's values reach OmegaConf and
starting a search spends money. Run with `uv run pytest`.
"""
import json
import sys

import pytest

from workflow_optimizer import runstore
from workflow_optimizer.config import load_config, load_resolved
from workflow_optimizer.dashboard import server


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point the run store at a temporary directory for the duration of a test."""
    monkeypatch.setattr(runstore, "RUNS_DIR", tmp_path / "runs")
    runstore.RUNS_DIR.mkdir(parents=True)
    return runstore.RUNS_DIR


@pytest.fixture
def a_run(runs_dir):
    """Create one run directory and return its status."""
    return runstore.create_run("gsm8k", load_config("gsm8k", ["designer.rounds=1"]))


# ---- run ids are filesystem paths, so they are validated ---------------------
@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "a/b", "..", "", "x" * 200, "has space", "semi;colon",
])
def test_unsafe_run_ids_are_rejected(bad):
    assert not runstore.is_valid_run_id(bad)
    with pytest.raises(ValueError):
        runstore.run_dir(bad)


def test_ordinary_run_ids_are_accepted():
    assert runstore.is_valid_run_id("gsm8k-20260720-143012")
    assert runstore.is_valid_run_id("ifeval-20260720-143012-2")


# ---- a run's state lives on disk --------------------------------------------
def test_creating_a_run_writes_config_and_status(a_run, runs_dir):
    directory = runs_dir / a_run.run_id
    assert (directory / "config.yaml").exists()
    assert (directory / "status.json").exists()
    assert a_run.task == "gsm8k"
    assert a_run.rounds == 1
    assert a_run.state == "running"


def test_status_survives_a_round_trip(a_run):
    runstore.update_status(a_run.run_id, phase="designing", round=2, n_candidates=3)
    reloaded = runstore.read_status(a_run.run_id)
    assert (reloaded.phase, reloaded.round, reloaded.n_candidates) == ("designing", 2, 3)


def test_events_append_in_order(a_run):
    runstore.append_event(a_run.run_id, {"event": "analyzing"})
    runstore.append_event(a_run.run_id, {"event": "candidate", "name": "H"})
    events = runstore.read_events(a_run.run_id)
    assert [e["event"] for e in events] == ["analyzing", "candidate"]
    assert all("t" in e for e in events)          # every event is timestamped


def test_a_half_written_event_line_is_skipped(a_run):
    # the UI polls while the runner is appending, so it can read a torn last line
    runstore.append_event(a_run.run_id, {"event": "analyzing"})
    with open(runstore.run_dir(a_run.run_id) / "events.jsonl", "a") as f:
        f.write('{"event": "candi')
    assert [e["event"] for e in runstore.read_events(a_run.run_id)] == ["analyzing"]


def test_a_run_whose_process_died_is_not_still_running(a_run):
    # a pid that cannot exist: the machine restarted, or the process was killed
    runstore.update_status(a_run.run_id, pid=2 ** 30)
    listed = runstore.list_runs()
    assert listed[0].state == "failed"
    assert "no longer running" in listed[0].error


def test_reading_an_unknown_run_returns_none(runs_dir):
    assert runstore.read_status("nope-20260101-000000") is None
    assert runstore.read_events("nope-20260101-000000") == []
    assert runstore.read_result("nope-20260101-000000") is None


# ---- what the server accepts from the form ----------------------------------
def test_an_unknown_task_cannot_name_a_file(runs_dir):
    result = server.start_run("../../etc/passwd", {})
    assert result["ok"] is False and "unknown task" in result["error"]


def test_only_listed_settings_can_be_overridden(runs_dir):
    result = server.start_run("gsm8k", {"task.grader": "/etc/passwd"})
    assert result["ok"] is False and "unknown setting" in result["error"]


def test_a_setting_that_is_not_a_number_is_rejected(runs_dir):
    result = server.start_run("gsm8k", {"designer.rounds": "; rm -rf /"})
    assert result["ok"] is False and "bad value" in result["error"]


def test_blank_settings_fall_back_to_the_config_default(runs_dir, monkeypatch):
    # the form submits "" for a field the user left alone
    started = {}
    real_create_run = runstore.create_run

    def spy(task, cfg):
        started["cfg"] = cfg
        return real_create_run(task, cfg)

    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    monkeypatch.setattr(server.runstore, "create_run", spy)

    result = server.start_run("gsm8k", {"designer.rounds": "", "data.n_examples": 12})
    assert result["ok"] is True
    assert started["cfg"].data.n_examples == 12                      # the value given
    assert started["cfg"].designer.rounds == load_config("gsm8k").designer.rounds  # the default


# ---- what the detail endpoint returns ---------------------------------------
def test_detail_merges_live_events_into_candidates(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "one call",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    runstore.append_event(a_run.run_id, {
        "event": "test_scored", "name": "H", "test_accuracy": 0.75, "test_cost": 0.0011})

    detail = server.run_detail(a_run.run_id)
    assert detail["status"]["run_id"] == a_run.run_id
    candidate = detail["candidates"][0]
    assert candidate["dev"]["accuracy"] == 0.8
    assert candidate["test"]["accuracy"] == 0.75
    assert candidate["code"] == ""          # code only lands when the search finishes


def test_detail_prefers_the_saved_result_once_it_exists(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    (runstore.run_dir(a_run.run_id) / "result.json").write_text(json.dumps({
        "frontier": ["H"],
        "candidates": [{"name": "H", "description": "one call", "code": "def solve(): ...",
                        "dev": {"accuracy": 0.8, "cost_per_query": 0.001},
                        "test": {"accuracy": 0.9, "cost_per_query": 0.0012}}]}))

    detail = server.run_detail(a_run.run_id)
    candidate = detail["candidates"][0]
    assert candidate["code"] == "def solve(): ..."
    assert candidate["test"]["accuracy"] == 0.9
    assert detail["frontier"] == ["H"]


def test_detail_of_an_unknown_run_is_not_found(runs_dir):
    assert server.run_detail("nope-20260101-000000") == {"error": "not_found"}


def test_stopping_a_finished_run_says_so(a_run):
    runstore.update_status(a_run.run_id, state="done")
    assert runstore.stop_run(a_run.run_id)["ok"] is False


def test_a_zombie_process_does_not_count_as_running(a_run):
    """A finished-but-unreaped child still answers `kill(pid, 0)`.

    Taken as alive, a crashed run would sit in the list as "running" forever —
    which is exactly what a real run did before `_process_alive` learned to reap.
    """
    import os
    import subprocess as sp
    import time

    child = sp.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 5
    while time.time() < deadline:       # wait for it to actually become a zombie
        state = sp.run(["ps", "-o", "stat=", "-p", str(child.pid)],
                       capture_output=True, text=True).stdout.strip()
        if state.startswith("Z"):
            break
        time.sleep(0.05)
    else:
        pytest.skip("could not produce a zombie on this platform")

    os.kill(child.pid, 0)               # the zombie is still signallable...
    runstore.update_status(a_run.run_id, pid=child.pid)
    assert runstore.list_runs()[0].state == "failed"   # ...but it is not running


# ---- free-text tasks and uploaded datasets ----------------------------------
def test_an_uploaded_jsonl_is_read(runs_dir):
    text = '{"question": "a", "answer": "1"}\n{"question": "b", "answer": "2"}\n'
    examples, reason = server.parse_dataset(text)
    assert reason == ""
    assert [e["question"] for e in examples] == ["a", "b"]


def test_a_json_array_and_alternate_key_names_are_read(runs_dir):
    # exports around here spell the gold "target" or "gold", and the input "prompt"
    text = '[{"prompt": "a", "target": "1"}, {"input": "b", "gold": "2"}]'
    examples, reason = server.parse_dataset(text)
    assert reason == "" and len(examples) == 2
    assert examples[0]["answer"] == "1" and examples[1]["question"] == "b"


def test_extra_columns_survive_for_a_custom_grader(runs_dir):
    text = '{"question": "a", "answer": "1", "doc": {"k": 2}}\n{"question": "b", "answer": "2"}\n'
    examples, _ = server.parse_dataset(text)
    assert examples[0]["doc"] == {"k": 2}


@pytest.mark.parametrize("text,expected", [
    ("", "empty"),
    ("not json", "not valid JSON"),
    ('{"question": "a"}\n{"question": "b", "answer": "x"}', "needs a question and an answer"),
    ('{"question": "a", "answer": "1"}', "at least 2 examples"),
])
def test_a_bad_dataset_is_refused_with_a_reason(runs_dir, text, expected):
    examples, reason = server.parse_dataset(text)
    assert examples == [] and expected in reason


def test_a_freetext_task_needs_no_task_file(runs_dir, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    result = server.start_run("", {}, prompt="Classify sentiment as positive or negative.")
    assert result["ok"] is True
    cfg = load_resolved(runstore.run_dir(result["run_id"]) / "config.yaml")
    assert cfg.task.name == "custom"
    assert "Classify sentiment" in cfg.task.seed_prompt


def test_an_uploaded_dataset_lands_in_the_run_and_is_pointed_at(runs_dir, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    result = server.start_run("", {}, prompt="Label it.",
                              dataset_text='{"question": "a", "answer": "1"}\n'
                                           '{"question": "b", "answer": "2"}\n')
    assert result["ok"] is True
    directory = runstore.run_dir(result["run_id"])
    assert (directory / "dataset.jsonl").exists()
    cfg = load_resolved(directory / "config.yaml")
    assert cfg.task.dataset == str(directory / "dataset.jsonl")


def test_a_freetext_run_still_rejects_unlisted_settings(runs_dir):
    result = server.start_run("", {"task.grader": "/etc/passwd"}, prompt="Label it.")
    assert result["ok"] is False and "unknown setting" in result["error"]


# ---- benchmarks and the comparison view -------------------------------------
def test_benchmarks_are_listed_with_their_metadata():
    found = {b["name"]: b for b in runstore.list_benchmarks()}
    if not found:
        pytest.skip("benchmarks/ not imported in this checkout")
    assert "ifeval" in found and "arc_agi_2" in found
    assert found["ifeval"]["baselines"]["haiku"] == pytest.approx(0.8478, abs=1e-4)
    # the code tasks cannot be graded here, and say so rather than scoring wrongly
    for name in ("humaneval_plus_gen", "mbpp_plus"):
        if name in found:
            assert found[name]["supported"] is False and found[name]["note"]


def test_compare_puts_every_run_on_the_same_axes(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "one call",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    runstore.append_event(a_run.run_id, {
        "event": "test_scored", "name": "H", "test_accuracy": 0.7, "test_cost": 0.0012})

    compared = server.compare_runs()
    point = compared["points"][0]
    assert point["run_id"] == a_run.run_id and point["task"] == "gsm8k"
    assert point["split"] == "test"            # test scores win where they exist
    assert point["accuracy"] == 0.7
