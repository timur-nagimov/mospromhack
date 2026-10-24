"""Проверка канала конфигурации gateway → edge на работающем шлюзе (docs/EDGE.md §8).

Повторно отправляет текущую конфигурацию камеры без изменений. Edge подтверждает новую версию
и камеру не перезапускает: работе камер и демо проверка не мешает. Годится после выкатки.

    python tools/config_sync_check.py --url http://127.0.0.1:8000 --camera cam_s1 [--out check.json]

Код выхода 0 — edge применил и подтвердил версию, 1 — нет (причина в выводе), 2 — неверный вызов.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def call(method: str, url: str, body: Any = None, timeout: float = 10.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 — адрес шлюза задаёт оператор
        text = response.read().decode("utf-8")
    return json.loads(text) if text else None


def check(base: str, camera_id: str | None, timeout: float) -> dict[str, Any]:
    report: dict[str, Any] = {"gateway": base, "ok": False}
    health = call("GET", f"{base}/health")
    report["mode"] = health.get("mode")
    report["config_sync"] = health.get("config_sync")
    if health.get("mode") != "gateway" or "config_sync" not in health:
        report["error"] = "gateway without config sync: old code or GUARD_CONFIG_SYNC=0"
        return report
    cameras = call("GET", f"{base}/api/cameras")
    camera = next((item for item in cameras if camera_id in (None, item["camera_id"])), None)
    if camera is None:
        report["error"] = f"camera {camera_id or '(any)'} not in /api/cameras: no edge catalogue yet?"
        return report
    report["camera_id"] = camera["camera_id"]
    sync = (camera.get("status") or {}).get("config_sync") or {}
    report["before"] = sync
    if sync.get("edge_version") is None:
        report["error"] = "edge does not report config_version: old edge code, or the catalogue has not arrived"
        return report
    body = {key: value for key, value in camera.items() if key != "status"}
    started = time.time()
    response = call("PUT", f"{base}/api/cameras/{camera['camera_id']}", body)
    version = response["status"]["config_sync"]["version"]
    report.update(version=version, put_s=round(time.time() - started, 3))
    deadline = started + timeout
    acked = None
    while time.time() < deadline:
        state = call("GET", f"{base}/api/cameras/{camera['camera_id']}")["status"]["config_sync"]
        if state["version"] == version and state["state"] != "pending" and acked is None:
            acked = time.time()
            report.update(state=state["state"], error=state["error"], ack_s=round(acked - started, 3))
            if state["state"] != "synced":
                return report
        if acked is not None and state.get("edge_version") == version:
            report["catalogue_s"] = round(time.time() - started, 3)
            report["ok"] = True
            return report
        time.sleep(0.1)
    report["error"] = report.get("error") or (
        "edge did not confirm in time: is the edge running new code with the same GUARD_EDGE_ID and broker?"
        if acked is None else "confirmation received, the edge catalogue did not show the version in time")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="адрес шлюза")
    parser.add_argument("--camera", help="camera_id; по умолчанию первая камера списка")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out", help="записать результат в JSON")
    args = parser.parse_args(argv)
    try:
        report = check(args.url.rstrip("/"), args.camera, args.timeout)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        report = {"gateway": args.url, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(text + "\n")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
