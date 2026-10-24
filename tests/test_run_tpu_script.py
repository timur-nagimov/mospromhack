"""Launch routing with stub forwards/VPN/stand, without network or inference."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,forward,expected,vpn_calls", [
    ("external", "ready", 0, []),
    ("external", "down", 1, []),
    ("auto", "ready", 0, []),
    ("auto", "after-up", 0, ["vpn:status", "vpn:up"]),
    ("container", "ready", 0, ["vpn:status", "vpn:up"]),
    ("invalid", "ready", 2, []),
])
def test_launch_preserves_existing_forwards_and_never_purges(tmp_path, mode, forward, expected, vpn_calls):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_tpu.sh").write_text((ROOT / "scripts/run_tpu.sh").read_text())
    log = tmp_path / "calls.log"
    python = tmp_path / "stub-python"
    python.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
log = Path(os.environ["TEST_LOG"])
if sys.argv[1:] == ["-"]:
    previous = log.read_text() if log.exists() else ""
    with log.open("a") as stream:
        stream.write("forward-check\\n")
    ready = os.environ["TEST_FORWARD"] == "ready" or (os.environ["TEST_FORWARD"] == "after-up" and "vpn:up" in previous)
    raise SystemExit(0 if ready else 1)
with log.open("a") as stream:
    stream.write("python:" + " ".join(sys.argv[1:]) + "\\n")
raise SystemExit(0)
''')
    python.chmod(0o755)
    vpn = tmp_path / "infra/vpn/vpnctl.sh"
    vpn.parent.mkdir(parents=True)
    vpn.write_text('''#!/bin/sh
printf 'vpn:%s\n' "$1" >> "$TEST_LOG"
if [ "$1" = status ]; then exit 1; fi
exit 0
''')
    vpn.chmod(0o755)
    dev = scripts / "run_dev.sh"
    dev.write_text('''#!/bin/sh
printf 'dev:%s:%s:%s\n' "$GUARD_MODE" "$GUARD_EDGE_ID" "$GUARD_BACKEND_FALLBACK" >> "$TEST_LOG"
''')
    dev.chmod(0o755)
    environment = {
        **os.environ, "GUARD_PYTHON": str(python), "TEST_LOG": str(log), "TEST_FORWARD": forward,
        "GUARD_VPN_MODE": mode, "GUARD_EDGE_ID": "test-edge", "GUARD_BACKEND_FALLBACK": "onnx-cpu",
        "GUARD_RABBIT_URL": "amqp://127.0.0.1:5672/", "JUPYTER_LOCAL_URL": "http://127.0.0.1:8888/", "SYNC": "0",
    }
    result = subprocess.run(["bash", str(scripts / "run_tpu.sh")], env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode == expected, result.stderr
    calls = log.read_text().splitlines() if log.exists() else []
    assert [line for line in calls if line.startswith("vpn:")] == vpn_calls
    assert not any("purge" in line or "edge_probe" in line for line in calls)
    if expected == 0:
        assert calls[-1] == "dev:gateway:test-edge:none"
        assert any("tools/stand.py" in line and "status" in line for line in calls)
    else:
        assert not any(line.startswith("dev:") or "tools/stand.py" in line for line in calls)
