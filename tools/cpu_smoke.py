"""Bounded, local, real CPU pipeline + HTTP/WS/media smoke. Never downloads assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:  # Also support python tools/cpu_smoke.py.
    sys.path.insert(0, str(ROOT))


class SmokeFailure(RuntimeError):
    pass


def _timeout(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 60:
        raise argparse.ArgumentTypeError("timeout должен быть в (0, 60] секунд")
    return seconds


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, help="исходные данные; иначе GUARD_DATA_DIR или checkout/data, без .env")
    result.add_argument("--models-dir", type=Path, help="модели; иначе GUARD_MODELS_DIR или checkout/models, без .env")
    result.add_argument("--output-dir", type=Path, help="новый, ещё не существующий каталог результатов")
    result.add_argument("--timeout", type=_timeout, default=60.0, help="общий wall timeout, максимум 60 с")
    result.add_argument("--json", action="store_true", help="один итоговый JSON в stdout; runtime log сохраняется отдельно")
    result.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    return result


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "opencv-python-headless", "onnxruntime", "fastapi", "starlette", "httpx", "imageio-ffmpeg"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _read_ws(ws: Any, inbox: queue.Queue, stopped: threading.Event) -> None:
    try:
        while not stopped.is_set():
            inbox.put(ws.receive_json())
    except Exception as exc:
        if not stopped.is_set():
            inbox.put({"kind": "smoke_ws_error", "error": type(exc).__name__})


def _wait_ws(inbox: queue.Queue, history: list, predicate: Any, deadline: float) -> dict:
    for item in history:
        if predicate(item):
            return item
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SmokeFailure("Истёк общий срок ожидания нужного WS сообщения")
        try:
            item = inbox.get(timeout=min(remaining, 0.25))
        except queue.Empty:
            continue
        history.append(item)
        _require(item.get("kind") != "smoke_ws_error", "WS закрылся до завершения проверки")
        if predicate(item):
            return item


def _settings(args: argparse.Namespace, output: Path):
    from guard.pipeline.settings import Settings

    return Settings(
        data_dir=(args.data_dir or Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data")).expanduser().resolve(),
        models_dir=(args.models_dir or Path(os.environ.get("GUARD_MODELS_DIR") or ROOT / "models")).expanduser().resolve(),
        cameras_config=ROOT / "configs/cameras.cpu-demo.yaml",
        models_config=ROOT / "configs/models.cpu-baseline.yaml",
        db_path=output / "guard.db", media_dir=output / "media", frontend_dist=output / "no_frontend",
        backend="onnx-cpu", mode="local", autostart=True, ort_threads=2,
        max_batch=1, clip_buffer_mb=128, clip_max_width=640,
    )


def _run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    report_path = output / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1, "scope": "local async CPU, one camera, real file + ONNX + HTTP/WS",
        "status": "running", "ok": False,
        "started_at": datetime.now(timezone.utc).isoformat(), "timeout_s": args.timeout,
        "output_dir": str(output), "checks": [], "assets": {}, "versions": _versions(),
        "denominators": {"cameras_enabled": 0, "models_required": 0, "source_frames": None,
                         "frames_processed": None, "http_events_total": None, "ws_event_messages": 0,
                         "events_compared": 0, "operator_updates_checked": 0,
                         "jpeg_responses_decoded": 0, "clip_frames_decoded": 0},
        "not_verified": ["TPU", "RTSP", "multi-camera behavior", "detection quality benchmark", "browser UI"],
    }
    started = time.monotonic()
    history: list[dict] = []
    client = None
    engine = None
    http_count = 0
    # Reserve five seconds for graceful TestClient / engine shutdown. The parent
    # process enforces the full wall deadline even if a runtime request hangs.
    deadline = args._deadline - 5.0

    def checked(name: str, details: Any = None) -> None:
        report["checks"].append({"check": name, "status": "ok", "details": details})
        _write(report_path, report)

    def http(method: str, route: str, **kwargs):
        nonlocal http_count
        _require(time.monotonic() < deadline, "Общий timeout перед HTTP запросом")
        http_count += 1
        return client.request(method, route, **kwargs)

    def capture_runtime() -> None:
        nonlocal http_count
        if client is None:
            return
        for key, route in (("health", "/health"), ("cameras", "/api/cameras"),
                           ("metrics", "/api/metrics"), ("events", "/api/events?limit=1000")):
            try:
                http_count += 1
                response = client.get(route)
                report[key] = response.json() if response.status_code == 200 else {"http_status": response.status_code}
                _write(output / f"{key}.json", report[key])
            except Exception as exc:
                report[key] = {"capture_error": type(exc).__name__}
        cams = report.get("cameras")
        if isinstance(cams, list) and cams:
            report["denominators"]["source_frames"] = cams[0].get("status", {}).get("source", {}).get("frames")
            camera_id = cams[0].get("camera_id")
            report["denominators"]["frames_processed"] = report.get("metrics", {}).get("cameras", {}).get(camera_id, {}).get("frames_processed")
        report["denominators"]["http_events_total"] = report.get("events", {}).get("total")

    try:
        settings = _settings(args, output)
        from guard.doctor import run_checks
        from guard.pipeline.config import load_cameras, load_models, resolve_source_uri

        preflight = run_checks(settings)
        report["preflight"] = preflight
        _write(report_path, report)
        _require(preflight["ok"], "Offline preflight нашёл ошибки: runtime не запускался")
        cameras = load_cameras(settings.cameras_config)
        _require(len(cameras) == 1 and cameras[0].enabled, "Нужна ровно одна включённая demo камера")
        camera = cameras[0]
        _require(camera.source.type == "file" and camera.source.realtime and not camera.source.loop,
                 "Smoke допускает только локальный realtime file без loop")
        _require(preflight["required_models"] == ["det"], "CPU demo должен требовать только det")
        report["denominators"].update(cameras_enabled=1, models_required=1)
        models = load_models(settings.models_config, settings.models_dir)
        spec = models["det"]
        from guard.inference.base import resolve_model_path

        paths = {"camera_config": settings.cameras_config, "models_config": settings.models_config,
                 "det_onnx": resolve_model_path(spec.onnx_path, settings.models_dir),
                 "video": Path(resolve_source_uri(camera.source.uri, settings.data_dir))}
        metadata = settings.models_dir / "det.json"
        if metadata.is_file():
            paths["det_metadata"] = metadata
        report["assets"] = {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
                            for key, path in paths.items()}
        report["settings"] = {"backend": "onnx-cpu", "ort_threads": 2, "cv2_threads": 2,
                              "sample_fps": camera.sample_fps, "clip_pre_s": settings.clip_pre_s,
                              "clip_post_s": settings.clip_post_s, "db_path": str(settings.db_path),
                              "media_dir": str(settings.media_dir)}
        checked("offline_inputs_and_hashes")

        import cv2
        import numpy as np
        import onnxruntime as ort
        from fastapi.testclient import TestClient
        from guard.api.app import create_app
        from guard.pipeline.engine import PipelineEngine

        ort.disable_telemetry_events()
        cv2.setNumThreads(2)
        engine = PipelineEngine(settings)
        app = create_app(settings, engine=engine)
        with TestClient(app) as active_client:
            client = active_client
            with client.websocket_connect("/ws/events") as ws:
                inbox: queue.Queue = queue.Queue()
                stopped = threading.Event()
                reader = threading.Thread(target=_read_ws, args=(ws, inbox, stopped), daemon=True)
                reader.start()
                try:
                    health = http("GET", "/health").json()
                    _require(health.get("backend") == "onnx-cpu" and health.get("cameras_running") == 1,
                             "Реальная CPU камера не запустилась")
                    _require(not health.get("fallbacks"), "Pipeline использует fallback компоненты")
                    checked("real_engine_camera_started", health)
                    message = _wait_ws(inbox, history, lambda item: item.get("kind") == "event"
                                       and item.get("event", {}).get("camera_id") == camera.camera_id
                                       and item["event"].get("type") == "zone_intrusion", deadline)
                    event = message["event"]
                    event_id = event["id"]
                    response = http("GET", f"/api/events/{event_id}")
                    _require(response.status_code == 200, "WS событие отсутствует в HTTP журнале")
                    row = response.json()
                    immutable = ("id", "camera_id", "site", "zone_id", "rule_id", "type", "severity",
                                 "track_id", "started_at", "confirmed_at", "bbox", "snapshot_path")
                    _require(all(row.get(key) == event.get(key) for key in immutable), "HTTP/WS event mismatch")
                    report["event_id"] = event_id
                    _write(output / "event_initial_http.json", row)
                    report["denominators"]["events_compared"] = 1
                    checked("event_http_ws_parity", {"event_id": event_id, "compared_fields": immutable})

                    note = "local CPU smoke operator verification"
                    patch = http("PATCH", f"/api/events/{event_id}", json={"status": "confirmed", "operator_note": note})
                    _require(patch.status_code == 200, "Операторский PATCH отклонён")
                    update = _wait_ws(inbox, history, lambda item: item.get("kind") == "event_update"
                                      and item.get("event", {}).get("id") == event_id
                                      and item["event"].get("status") == "confirmed"
                                      and item["event"].get("operator_note") == note, deadline)
                    patched = http("GET", f"/api/events/{event_id}").json()
                    _require(patched.get("status") == "confirmed" and patched.get("operator_note") == note,
                             "Операторский PATCH не сохранился")
                    _write(output / "event_patched_http.json", patched)
                    report["denominators"]["operator_updates_checked"] = 1
                    checked("operator_patch_http_ws_readback", {"event_id": update["event"]["id"]})

                    jpeg_evidence = []
                    for name, route in (("camera_frame", f"/api/cameras/{camera.camera_id}/frame.jpg"),
                                        ("event_snapshot", f"/api/events/{event_id}/snapshot.jpg")):
                        response = http("GET", route)
                        _require(response.status_code == 200 and response.headers.get("content-type", "").startswith("image/jpeg"),
                                 f"{name}: JPEG HTTP недоступен")
                        image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                        _require(image is not None and image.size, f"{name}: JPEG не декодируется")
                        path = output / f"{name}.jpg"
                        path.write_bytes(response.content)
                        jpeg_evidence.append({"name": name, "bytes": len(response.content), "shape": list(image.shape), "sha256": _sha(path)})
                        report["denominators"]["jpeg_responses_decoded"] += 1
                    checked("http_jpeg_decode", jpeg_evidence)

                    clip_update = _wait_ws(inbox, history, lambda item: item.get("kind") == "event_update"
                                          and item.get("event", {}).get("id") == event_id
                                          and bool(item["event"].get("clip_path")), deadline)
                    clip_event = http("GET", f"/api/events/{event_id}").json()
                    _require(clip_event.get("clip_path") == clip_update["event"]["clip_path"], "HTTP/WS clip path mismatch")
                    response = http("GET", f"/api/events/{event_id}/clip.mp4")
                    _require(response.status_code == 200 and response.headers.get("content-type", "").startswith("video/mp4"),
                             "H264 clip HTTP недоступен")
                    clip = output / "event_clip.mp4"
                    clip.write_bytes(response.content)
                    cap = cv2.VideoCapture(str(clip))
                    decoded, fps, fourcc = 0, cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FOURCC))
                    codec = "".join(chr((fourcc >> (8 * index)) & 0xff) for index in range(4))
                    try:
                        while time.monotonic() < deadline:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            _require(frame is not None and frame.size, "Кадр клипа пуст")
                            decoded += 1
                    finally:
                        cap.release()
                    _require(time.monotonic() < deadline, "Timeout декодирования клипа")
                    _require(decoded >= 2 and fps > 0 and codec.lower() in {"avc1", "h264"}, "Не подтверждены H264 и декодирование нескольких кадров")
                    report["denominators"]["clip_frames_decoded"] = decoded
                    checked("http_h264_clip_decode", {"bytes": len(response.content), "frames_decoded": decoded,
                                                     "fps": fps, "codec": codec, "sha256": _sha(clip)})

                    capture_runtime()
                    status = report["cameras"][0]["status"]
                    metrics = report["metrics"]
                    cam_metrics = metrics.get("cameras", {}).get(camera.camera_id, {})
                    model_stats = metrics.get("inference", {}).get("models", {}).get("det", {})
                    _require(not status.get("errors") and not status.get("source", {}).get("errors"), "Камера/источник сообщает ошибки")
                    _require(cam_metrics.get("frames_processed", 0) > 0 and cam_metrics.get("errors", 0) == 0,
                             "Нет обработанных кадров или есть ошибки конвейера")
                    _require(model_stats.get("calls", 0) > 0 and model_stats.get("errors", 0) == 0,
                             "Отсутствуют inference calls либо есть ошибки модели")
                    _require(metrics.get("media", {}).get("clips_written", 0) > 0 and metrics.get("media", {}).get("clip_errors", 0) == 0,
                             "Отсутствуют завершённые клипы либо есть ошибки media")
                    checked("runtime_status_and_metrics", {"camera_errors": 0, "source_errors": 0, "clip_errors": 0})
                    report.update(ok=True, status="passed")
                finally:
                    if not report["ok"]:
                        capture_runtime()
                    stopped.set()
                    _write(output / "websocket_messages.json", history)
            reader.join(timeout=0.5)
            client = None
    except Exception as exc:
        report.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["http_requests"] = http_count
        report["elapsed_s"] = time.monotonic() - started
        report["ws_messages_observed"] = len(history)
        report["denominators"]["ws_event_messages"] = sum(item.get("kind") == "event" for item in history)
        _write(report_path, report)
    return 0 if report["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args._run:
        return _run(args)
    started = time.monotonic()
    if args.output_dir is None:
        from guard.pipeline.settings import Settings
        data = (args.data_dir or Settings.from_env().data_dir).expanduser()
        args.output_dir = data / "cpu_async" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        arguments.error("--output-dir уже существует; результаты не перезаписываются")
    output.mkdir(parents=True)
    command = [sys.executable, "-m", "tools.cpu_smoke", "--_run", "--output-dir", str(output),
               "--timeout", str(args.timeout), "--_deadline", str(started + args.timeout)]
    for option, path in (("--data-dir", args.data_dir), ("--models-dir", args.models_dir)):
        if path is not None:
            command.extend((option, str(path.expanduser().resolve())))
    with (output / "runtime.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True,
                                   env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        timed_out = False
        try:
            process.wait(timeout=max(0.01, args.timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The child may have exited between timeout and signal.
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1)
    path = output / "report.json"
    report = json.loads(path.read_text()) if path.is_file() else {
        "scope": "local async CPU", "output_dir": str(output), "versions": _versions(),
        "checks": [], "assets": {}, "denominators": {},
    }
    if timed_out or process.returncode != 0 or not report.get("ok"):
        report.update(ok=False, status="timeout" if timed_out else "failed")
        report.setdefault("error", "Общий wall timeout" if timed_out else "См. runtime.log")
    report["wall_s"] = time.monotonic() - started
    report["runtime_returncode"] = process.returncode
    _write(path, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(f"CPU smoke: {report['status']}; report={path}; runtime log={output / 'runtime.log'}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
