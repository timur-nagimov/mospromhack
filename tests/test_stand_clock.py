import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import stand_clock  # noqa: E402


def test_offset_is_laptop_minus_stand_from_both_ntp_offsets():
    # offset_ms — сколько прибавить к своим часам, чтобы получить NTP. Стенд отстаёт: его offset больше.
    stand = {"offset_ms": 4400.0, "uncertainty_ms": 3.0}
    laptop = {"offset_ms": -100.0, "uncertainty_ms": 12.0}
    assert stand_clock.laptop_minus_stand(stand, laptop) == {"laptop_minus_stand_ms": 4500.0, "uncertainty_ms": 15.0}


def test_missing_ntp_side_is_an_error():
    with pytest.raises(ValueError):
        stand_clock.laptop_minus_stand({"error": "time.google.com: timeout"}, {"offset_ms": 1.0, "uncertainty_ms": 1.0})


def test_measure_runs_ntp_on_stand_kernel_and_locally():
    calls = []

    def execute(code, timeout):
        calls.append(code)
        return SimpleNamespace(ok=True, stdout=json.dumps({"offset_ms": 4300.0, "uncertainty_ms": 2.0}) + "\n")

    client = SimpleNamespace(execute=execute)
    result = stand_clock.measure(client, local_ntp=lambda: {"offset_ms": 0.0, "uncertainty_ms": 5.0})
    assert result["laptop_minus_stand_ms"] == 4300.0
    assert "ntp_best" in calls[0] and "work/mosprom" in calls[0]
