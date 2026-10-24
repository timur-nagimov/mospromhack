"""Safety of the smoke CLI: no model/runtime loads in help or refusal paths."""

import json
import queue
import signal
import subprocess
import sys
import time

import pytest

from tools import cpu_smoke


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "61"])
def test_timeout_is_bounded(value):
    with pytest.raises(SystemExit) as error:
        cpu_smoke.parser().parse_args(["--timeout", value])
    assert error.value.code == 2


def test_existing_result_directory_is_never_reused(tmp_path):
    evidence = tmp_path / "report.json"
    evidence.write_text("preserved")
    with pytest.raises(SystemExit) as error:
        cpu_smoke.main(["--output-dir", str(tmp_path), "--json"])
    assert error.value.code == 2
    assert evidence.read_text() == "preserved"


def test_ws_wait_has_a_shared_deadline_even_with_unrelated_messages():
    inbox = queue.Queue()
    for _ in range(3):
        inbox.put({"kind": "metrics"})
    history = []
    started = time.monotonic()
    with pytest.raises(cpu_smoke.SmokeFailure, match="срок ожидания"):
        cpu_smoke._wait_ws(inbox, history, lambda item: item.get("kind") == "event", started + 0.02)
    assert time.monotonic() - started < 0.5
    assert len(history) == 3


def test_ws_disconnect_fails_instead_of_waiting_for_missing_kind():
    inbox = queue.Queue()
    inbox.put({"kind": "smoke_ws_error", "error": "WebSocketDisconnect"})
    with pytest.raises(cpu_smoke.SmokeFailure, match="WS закрылся"):
        cpu_smoke._wait_ws(inbox, [], lambda item: item.get("kind") == "event", time.monotonic() + 10)


def test_global_watchdog_kills_process_group_and_reports_failure(monkeypatch, tmp_path, capsys):
    class Process:
        pid = 123456789
        returncode = None
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls < 3:
                raise subprocess.TimeoutExpired("smoke", timeout)
            self.returncode = -signal.SIGKILL

    process, signals = Process(), []
    monkeypatch.setattr(cpu_smoke.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(cpu_smoke.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / "new-output"
    assert cpu_smoke.main(["--output-dir", str(output), "--timeout", "1", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and report["ok"] is False
    assert [sig for _, sig in signals] == [signal.SIGTERM, signal.SIGKILL]
    assert json.loads((output / "report.json").read_text()) == report
    assert report["versions"]["python"]


def test_missing_assets_return_json_without_starting_inference(tmp_path):
    code = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    blocked = ("cv2", "onnxruntime", "guard.pipeline.engine", "guard.sources", "fastapi")
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError("runtime import before successful preflight: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools import cpu_smoke
args = cpu_smoke.parser().parse_args(["--_run", "--_deadline", str(cpu_smoke.time.monotonic() + 10),
    "--data-dir", sys.argv[1], "--models-dir", sys.argv[2], "--output-dir", sys.argv[3]])
raise SystemExit(cpu_smoke._run(args))
'''
    output = tmp_path / "output"
    output.mkdir()
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "missing-data"),
                             str(tmp_path / "missing-models"), str(output)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert not report["ok"] and report["preflight"]["ok"] is False
    assert "runtime import" not in report["error"]
    assert not (output / "guard.db").exists() and not (output / "media").exists()


def test_cli_failure_has_one_json_stdout_and_exit_one(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-m", "tools.cpu_smoke", "--json", "--output-dir", str(output),
                             "--data-dir", str(tmp_path / "missing"), "--models-dir", str(tmp_path / "missing-models")],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False and report["status"] == "failed"
    assert report["runtime_returncode"] == 1 and report["denominators"]["events_compared"] == 0
    assert (output / "runtime.log").is_file()


@pytest.mark.parametrize("module", [True, False])
def test_help_does_not_load_runtime(module):
    command = [sys.executable, "-m", "tools.cpu_smoke"] if module else [sys.executable, str(cpu_smoke.ROOT / "tools/cpu_smoke.py")]
    result = subprocess.run([*command, "--help"], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--timeout" in result.stdout and "--models-dir" in result.stdout
