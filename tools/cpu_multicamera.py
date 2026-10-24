"""Bounded acceptance of three real realtime file cameras on the shared CPU backend.

No downloads, network sources, loops, .env loading, or replacement pipeline components.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import math
import os
from pathlib import Path
import queue
import resource
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = ROOT.parent.parent if ROOT.parent.name == ".worktrees" else ROOT
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))
from tools.cpu_smoke import _read_ws, _sha, _versions, _write

CAMERAS = (
    ("cpu_worker_zone", "a3dev_worker_zone_detection/worker-zone-detection.mp4", "s1_01.mp4"),
    ("cpu_store_aisle", "a3dev_store_aisle_detection/store-aisle-detection.mp4", "s1_02.mp4"),
    ("cpu_people_load", "a3dev_people_detection/people-detection.mp4", None),
)
SAMPLE_FPS = 5.0
MIN_COVERAGE = 0.80
MAX_PROGRESS_GAP_S = 5.0
IMMUTABLE_EVENT_FIELDS = ("id", "camera_id", "site", "zone_id", "rule_id", "type", "severity",
                          "track_id", "started_at", "confirmed_at", "bbox", "snapshot_path", "details")


class AcceptanceFailure(RuntimeError):
    pass


class CLIError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIError(message)


def _timeout(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or not 0 < seconds <= 180:
        raise argparse.ArgumentTypeError("timeout должен быть в (0, 180] секунд")
    return seconds


def parser() -> argparse.ArgumentParser:
    result = _Parser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=SHARED_ROOT / "data", help="локальные оригиналы videos/; .env не читается")
    result.add_argument("--models-dir", type=Path, default=SHARED_ROOT / "models", help="локальный yolov8n-640.onnx")
    result.add_argument("--output-dir", type=Path, help="новый каталог с отдельными DB, media и evidence")
    result.add_argument("--timeout", type=_timeout, default=120.0, help="общий wall deadline, по умолчанию 120 с, максимум 180 с")
    result.add_argument("--json", action="store_true", help="ровно один итоговый JSON в stdout")
    result.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    return result


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AcceptanceFailure(message)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """stdlib only: missing inputs fail before imports of any runtime component."""
    paths = {cid: args.data_dir / "videos" / relative for cid, relative, _ in CAMERAS}
    paths.update(det_onnx=args.models_dir / "yolov8n-640.onnx", manifest=ROOT / "tools/data_manifest.json",
                 cpu_registry=ROOT / "configs/models.cpu-baseline.yaml")
    metadata = args.models_dir / "det.json"
    if metadata.is_file():
        paths["det_metadata"] = metadata
    missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
    _require(not missing, "Отсутствуют непустые локальные входы: " + ", ".join(missing))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    entries = {item["name"]: item for item in manifest["videos"]}
    cameras = []
    for cid, relative, annotation in CAMERAS:
        zones = ([{key: zone[key] for key in ("zone_id", "name", "kind", "polygon")}
                  for zone in entries[annotation]["zones"]] if annotation else [{
                      "zone_id": "full_frame_load_test", "name": "Full frame: load test only", "kind": "danger",
                      "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                  }])
        _require(bool(zones), f"Нет S1 зон: {annotation}")
        cameras.append({
            "camera_id": cid, "name": cid, "site": "CPU acceptance", "enabled": True, "sample_fps": SAMPLE_FPS,
            "source": {"type": "file", "uri": str(paths[cid]), "realtime": True, "loop": False},
            "zones": zones,
            "rules": [{"rule_id": "s1_zone" if annotation else "load_test_zone", "type": "zone_intrusion",
                       "zones": [zone["zone_id"] for zone in zones], "params": {"min_conf": 0.35, "min_box_h_px": 40},
                       "min_duration_s": 2.0, "min_ratio": 0.7, "close_after_s": 3.0,
                       "cooldown_s": 30.0, "severity": "high"}],
        })
    return {"ok": True, "paths": paths, "cameras": cameras}


def _settings(args: argparse.Namespace, output: Path, cameras: list[dict]):
    from guard.pipeline.settings import Settings
    from guard.pipeline.config import load_models

    _write(output / "cameras.json", {"cameras": cameras})
    model = load_models(ROOT / "configs/models.cpu-baseline.yaml", args.models_dir)["det"]
    _require(model.tpu_program is None, "CPU baseline registry must not require a TPU program")
    _write(output / "models.json", {"models": [model.model_dump(mode="json")]})
    return Settings(data_dir=args.data_dir, models_dir=args.models_dir,
                    cameras_config=output / "cameras.json", models_config=output / "models.json",
                    db_path=output / "guard.db", media_dir=output / "media", frontend_dist=output / "no_frontend",
                    backend="onnx-cpu", mode="local", autostart=False, ort_threads=2, batching=False,
                    max_batch=1, clip_buffer_mb=128, clip_max_width=640)


class Observations:
    """Append each real process result to per-camera JSONL and an in-memory full-run sample."""
    def __init__(self, output: Path) -> None:
        self.rows: dict[str, list[dict]] = {cid: [] for cid, _, _ in CAMERAS}
        self._files = {cid: (output / f"frames-{cid}.jsonl").open("x", encoding="utf-8") for cid in self.rows}
        self._lock = threading.Lock()

    def add(self, camera_id: str, row: dict) -> None:
        with self._lock:
            self._files[camera_id].write(json.dumps(row, ensure_ascii=False) + "\n")
            self._files[camera_id].flush()
            self.rows[camera_id].append(row)

    def snapshot(self) -> dict[str, list[dict]]:
        with self._lock:
            return {cid: list(rows) for cid, rows in self.rows.items()}

    def close(self) -> None:
        for stream in self._files.values():
            stream.close()


@contextmanager
def _observe_process(worker_type: type, observations: Observations):
    """Call the original method exactly once and retain its result/exception unchanged."""
    original = worker_type.process

    @wraps(original)
    def observed(self, packet):
        row = {"capture_id": packet.capture_id, "seq": packet.seq, "capture_ts": packet.ts,
               "source_pos_s": packet.pos_s}
        try:
            result = original(self, packet)
        except BaseException as exc:
            row.update(ok=False, error=type(exc).__name__, done_wall=time.time(), done_mono=time.monotonic())
            observations.add(self.cfg.camera_id, row)
            raise
        row.update(ok=True, done_wall=time.time(), done_mono=time.monotonic(),
                   timings_ms=dict(result.timings),
                   detection_counts={model: len(items) for model, items in result.detections.items()},
                   rules=[rule.cfg.rule_id for rule in self.rules])
        observations.add(self.cfg.camera_id, row)
        return result

    worker_type.process = observed
    try:
        yield
    finally:
        worker_type.process = original


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    """Linear-interpolated full-run quantiles; never average per-camera quantiles."""
    values = sorted(values)
    def percentile(q: float):
        if not values:
            return None
        index = (len(values) - 1) * q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"n": len(values), "p50": percentile(0.5), "p95": percentile(0.95), "max": max(values) if values else None}


def _summarize(rows: list[dict], source_frames: int, native: dict, sample_fps: float = SAMPLE_FPS) -> dict:
    successful = [row for row in rows if row["ok"]]
    exposure = native["duration_s"]
    target = math.ceil(exposure * sample_fps)
    keys = sorted({key for row in successful for key in row["timings_ms"]})
    capture_gaps = [b["capture_ts"] - a["capture_ts"] for a, b in zip(successful, successful[1:])]
    done_gaps = [b["done_wall"] - a["done_wall"] for a, b in zip(successful, successful[1:])]
    return {
        "native_frames": native["frames"], "native_fps": native["fps"], "exposure_s": exposure,
        "sample_fps_configured": sample_fps, "configured_sample_target": target,
        "source_frames": source_frames, "frames_processed": len(successful), "processing_failures": len(rows) - len(successful),
        "native_frames_not_published": native["frames"] - source_frames,
        "skipped_by_sampling_or_overload": source_frames - len(successful),
        "sample_target_coverage": len(successful) / target if target else 0.0,
        "effective_processed_fps": len(successful) / exposure if exposure else 0.0,
        "timings_ms": {key: _distribution([row["timings_ms"][key] for row in successful if key in row["timings_ms"]]) for key in keys},
        "analysis_age_at_completion_ms": _distribution([(row["done_wall"] - row["capture_ts"]) * 1000 for row in successful]),
        "capture_gap_s": _distribution(capture_gaps), "completion_gap_s": _distribution(done_gaps),
        "det_observations": sum("det" in row["detection_counts"] for row in successful),
        "rule_observations": sum(bool(row["rules"]) for row in successful),
        "first": successful[0] if successful else None, "last": successful[-1] if successful else None,
    }


def _settled(workers: dict, events_created: int, media: dict) -> bool:
    return (len(workers) == len(CAMERAS)
            and all(worker.source.eof and not worker.source.running and worker.source.latest() is not None
                    and worker.last is not None and worker.last.seq == worker.source.latest().seq for worker in workers.values())
            and media.get("pending") == 0 and media.get("clip_errors") == 0
            and media.get("clips_written") == events_created)


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _sqlite_events(path: Path) -> list[dict]:
    # URI read-only prevents accidentally creating a missing database during readback.
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = [dict(row) for row in connection.execute("SELECT * FROM events ORDER BY id")]
    for row in rows:
        for key in ("bbox", "details", "latency_ms"):
            if isinstance(row.get(key), str):
                row[key] = json.loads(row[key])
    return rows


def _run(args: argparse.Namespace) -> int:
    output = args.output_dir
    report_path = output / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1, "scope": "three real realtime local file cameras, shared CPU ONNX, HTTP/WS/SQLite",
        "status": "running", "ok": False, "started_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output), "timeout_s": args.timeout, "versions": _versions(), "checks": [],
        "acceptance_thresholds": {"sample_target_coverage_min": MIN_COVERAGE, "running_progress_gap_max_s": MAX_PROGRESS_GAP_S,
                                  "basis": "chosen acceptance thresholds, not measured product guarantees"},
        "not_verified": ["TPU", "RTSP", "detection quality benchmark", "browser UI", "people video zone quality"],
        "denominators": {"cameras": 3, "required_models": 1, "events_compared": 0, "jpeg_cameras_decoded": 0},
        "health_is_readiness": False,
    }
    started = time.monotonic()
    deadline = args._deadline - 10.0  # reserve shutdown time; parent enforces the complete deadline
    observations = None
    engine = None
    history: list[dict] = []
    workers: dict = {}
    media_thread = None
    http_events: dict = {}
    native: dict = {}

    def checked(name: str, details: Any = None):
        report["checks"].append({"check": name, "status": "ok", "details": details})
        _write(report_path, report)

    try:
        preflight = _preflight(args)
        report["preflight"] = {"ok": True}
        report["assets"] = {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
                            for key, path in preflight["paths"].items()}
        settings = _settings(args, output, preflight["cameras"])
        report["settings"] = {"backend": "onnx-cpu", "mode": "local", "autostart": False,
                              "ort_threads": 2, "cv2_threads": 2, "batching": False, "sample_fps": SAMPLE_FPS,
                              "db_path": str(settings.db_path), "media_dir": str(settings.media_dir),
                              "clip_pre_s": settings.clip_pre_s, "clip_post_s": settings.clip_post_s}
        checked("local_inputs_and_hashes")
        import cv2
        import numpy as np
        import onnxruntime as ort
        from fastapi.testclient import TestClient
        from guard.pipeline.settings import Settings
        # app.py constructs a module-level app at import. Keep that convenience
        # app from reading .env; the acceptance app always receives this Settings.
        with patch.object(Settings, "from_env", return_value=settings):
            from guard.api.app import create_app
        from guard.pipeline.engine import PipelineEngine
        from guard.pipeline.worker import CameraWorker
        from guard.pipeline import compat

        ort.disable_telemetry_events()
        cv2.setNumThreads(2)
        for cid, _, _ in CAMERAS:
            cap = cv2.VideoCapture(str(preflight["paths"][cid]))
            try:
                _require(cap.isOpened(), f"Нельзя открыть {cid}")
                fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
                _require(math.isfinite(fps) and fps > 0 and math.isfinite(frames) and frames > 0, f"Нет native знаменателей {cid}")
                native[cid] = {"frames": int(frames), "fps": fps, "duration_s": frames / fps}
            finally:
                cap.release()
        report["native_inputs"] = native
        _require(deadline > time.monotonic() + max(item["duration_s"] for item in native.values()),
                 "Оставшегося deadline недостаточно для полного realtime EOF")
        observations = Observations(output)
        engine = PipelineEngine(settings)
        _require(not any(compat.FALLBACKS.values()) and engine.journal.__class__.__module__ == "guard.events.journal",
                 "Pipeline использует fallback компоненты")
        report["process_memory"] = {"rss_high_water_bytes_before_start": _rss_bytes(),
                                     "scope": "shared acceptance process; not attributable per camera or model"}
        with _observe_process(CameraWorker, observations):
            with TestClient(create_app(settings, engine=engine)) as client:
                def get(route: str):
                    _require(time.monotonic() < deadline, "Общий deadline перед HTTP")
                    response = client.get(route)
                    _require(response.status_code == 200, f"HTTP {route}: {response.status_code}")
                    return response

                with client.websocket_connect("/ws/events") as ws:
                    inbox: queue.Queue = queue.Queue()
                    stopped = threading.Event()
                    reader = threading.Thread(target=_read_ws, args=(ws, inbox, stopped), daemon=True)
                    reader.start()
                    try:
                        # accept() precedes subscribe() in the endpoint. Wait for
                        # subscription explicitly before workers can publish.
                        while not engine.bus.subscribers and time.monotonic() < deadline:
                            time.sleep(0.01)
                        _require(engine.bus.subscribers > 0, "WS observer не подписался до старта")
                        engine.start()
                        workers = dict(engine.workers)
                        media_thread = engine.media._thread
                        _require(len(workers) == 3, "Запустились не все 3 камеры")
                        report["health"] = get("/health").json()
                        _require(report["health"]["backend"] == "onnx-cpu" and not report["health"]["fallbacks"], "Неверный backend/fallbacks")
                        for cid, worker in workers.items():
                            _require(worker.model_ids == ["det"] and worker.rules and not worker.errors, f"{cid}: нет det/rules или есть ошибки")
                        checked("three_real_workers_models_and_rules", {cid: worker.status() for cid, worker in workers.items()})
                        jpeg_done: set[str] = set()
                        progress: dict[str, dict] = {}
                        last_report = 0.0
                        event_id = None
                        while time.monotonic() < deadline:
                            _require(engine.media.clip_errors == 0, "Ошибка записи клипа")
                            while True:
                                try:
                                    message = inbox.get_nowait()
                                except queue.Empty:
                                    break
                                history.append(message)
                                _require(message.get("kind") != "smoke_ws_error", "WS закрылся до завершения")
                                event = message.get("event", {})
                                if (event_id is None and message.get("kind") == "event" and event.get("camera_id") == CAMERAS[0][0]
                                        and event.get("type") == "zone_intrusion"):
                                    row = get(f"/api/events/{event['id']}").json()
                                    _require(all(row.get(key) == event.get(key) for key in IMMUTABLE_EVENT_FIELDS), "S1 HTTP/WS event mismatch")
                                    event_id = event["id"]
                                    report["event_id"] = event_id
                                    report["denominators"]["events_compared"] = 1
                                    _write(output / "s1_event_ws.json", event)
                                    _write(output / "s1_event_http.json", row)
                                    checked("worker_zone_s1_event_http_ws_parity", {"id": event_id, "fields": IMMUTABLE_EVENT_FIELDS})
                            now = time.monotonic()
                            for cid, worker in workers.items():
                                source, last = worker.source, worker.last
                                _require(not worker.errors and source.errors == 0 and worker.metrics.errors == 0, f"{cid}: runtime errors")
                                current = (source.latest().seq if source.latest() else 0, last.seq if last else 0)
                                previous = progress.setdefault(cid, {"source_seq": 0, "analysis_seq": 0, "source_changed": now, "analysis_changed": now})
                                for index, name in enumerate(("source", "analysis")):
                                    if current[index] != previous[name + "_seq"]:
                                        previous[name + "_seq"] = current[index]
                                        previous[name + "_changed"] = now
                                    if source.running:
                                        _require(now - previous[name + "_changed"] <= MAX_PROGRESS_GAP_S, f"{cid}: {name} не продвигается при running source")
                                if last is not None and cid not in jpeg_done:
                                    response = get(f"/api/cameras/{cid}/frame.jpg")
                                    image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
                                    _require(response.headers.get("content-type", "").startswith("image/jpeg") and image is not None and image.size, f"{cid}: JPEG не декодируется")
                                    path = output / f"frame-{cid}.jpg"
                                    path.write_bytes(response.content)
                                    jpeg_done.add(cid)
                                    report["denominators"]["jpeg_cameras_decoded"] = len(jpeg_done)
                            if now - last_report >= 2.0:
                                report["cameras"] = get("/api/cameras").json()
                                report["metrics_rolling_diagnostic_only"] = get("/api/metrics").json()
                                report["elapsed_s"] = now - started
                                _write(report_path, report)
                                last_report = now
                            if _settled(workers, engine.events_created, engine.media.stats()):
                                http_events = get("/api/events?limit=1000").json()
                                _require(http_events["total"] == len(http_events["items"]), "Event HTTP pagination не покрывает весь журнал")
                                _require(http_events["total"] == engine.events_created, "Не все созданные события сохранены в журнале")
                                if all(event.get("clip_path") for event in http_events["items"]):
                                    break
                            time.sleep(0.1)
                        else:
                            raise AcceptanceFailure("Deadline: не все источники EOF / analysis caught up / clips drained")
                        _require(event_id is not None, "Нет подтверждённого S1 worker-zone события через WS")
                        _require(len(jpeg_done) == 3, "JPEG не подтверждён для каждой камеры")
                        checked("all_three_eof_last_analysis_and_clips_drained", {cid: {"source_seq": worker.source.latest().seq, "analysis_seq": worker.last.seq} for cid, worker in workers.items()})
                        _write(output / "events_before_stop.json", http_events)
                        report["denominators"]["http_events_total"] = http_events["total"]
                        report["cameras"] = get("/api/cameras").json()
                        report["metrics_rolling_diagnostic_only"] = get("/api/metrics").json()
                        inference = report["metrics_rolling_diagnostic_only"]["inference"]
                        det_stats = inference.get("models", {}).get("det", {})
                        _require(engine.backend.loaded_models() == ["det"]
                                 and inference.get("providers") == ["CPUExecutionProvider"]
                                 and det_stats.get("calls", 0) > 0 and det_stats.get("errors", 0) == 0,
                                 "Нет успешных реальных CPU inference calls для единственной det модели")
                        checked("shared_cpu_detector_calls_no_errors", {"calls": det_stats["calls"], "providers": inference["providers"]})
                        report["progress"] = progress
                        engine.stop()
                        checked("graceful_engine_stop")
                    finally:
                        stopped.set()
                reader.join(timeout=1.0)
                _require(not reader.is_alive(), "WS reader не завершился")
        shutdown = {"engine_stopped": engine._stopped, "worker_registry_empty": not engine.workers,
                    "worker_threads_stopped": all(w._thread is None or not w._thread.is_alive() for w in workers.values()),
                    "source_threads_stopped": all(w.source._thread is None or not w.source._thread.is_alive() for w in workers.values()),
                    "media_thread_stopped": media_thread is None or not media_thread.is_alive(),
                    "backend_models_unloaded": engine.backend.loaded_models() == []}
        _require(all(shutdown.values()), "Некорректное завершение worker/source/media/backend")
        checked("clean_workers_sources_media_backend_shutdown", shutdown)
        persisted = _sqlite_events(settings.db_path)
        expected = {event["id"]: event for event in http_events["items"]}
        _require(set(expected) == {event["id"] for event in persisted}, "SQLite после stop: потеря/лишние события")
        for event in persisted:
            _require(all(event.get(key) == expected[event["id"]].get(key) for key in (*IMMUTABLE_EVENT_FIELDS, "clip_path")), "SQLite после stop: поля события отличаются")
        _write(output / "events_sqlite_after_stop.json", persisted)
        checked("sqlite_event_readback_after_stop", {"events": len(persisted)})
        successful = observations.snapshot()
        report["per_camera"] = {cid: _summarize(successful[cid], worker.source.status()["frames"], native[cid]) for cid, worker in workers.items()}
        for cid, summary in report["per_camera"].items():
            _require(summary["frames_processed"] > 0 and summary["processing_failures"] == 0, f"{cid}: пустой/ошибочный анализ")
            _require(summary["frames_processed"] == workers[cid].metrics.snapshot()["frames_processed"], f"{cid}: наблюдения не покрывают весь анализ")
            _require(summary["det_observations"] == summary["frames_processed"]
                     and summary["rule_observations"] == summary["frames_processed"], f"{cid}: кадры без det/rules")
            _require(summary["sample_target_coverage"] >= MIN_COVERAGE, f"{cid}: coverage {summary['sample_target_coverage']:.3f} < {MIN_COVERAGE}")
        spans = [(rows[0]["capture_ts"], rows[-1]["capture_ts"]) for rows in successful.values() if rows]
        report["three_camera_observed_overlap_s"] = max(0.0, min(end for _, end in spans) - max(start for start, _ in spans))
        report["process_memory"]["rss_high_water_bytes_after_stop"] = _rss_bytes()
        checked("full_run_raw_observations_and_coverage", {cid: summary["sample_target_coverage"] for cid, summary in report["per_camera"].items()})
        report.update(ok=True, status="passed")
    except Exception as exc:
        report.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
        if "preflight" not in report:
            report["preflight"] = {"ok": False}
    finally:
        if engine is not None and not engine._stopped:
            try:
                engine.stop()
            except Exception as exc:
                report.update(ok=False, status="failed", cleanup_error=f"{type(exc).__name__}: {exc}")
        if observations is not None:
            observations.close()
            report["raw_observations"] = {cid: {"path": str(output / f"frames-{cid}.jsonl"), "n": len(rows)} for cid, rows in observations.snapshot().items()}
            report["per_camera"] = {cid: _summarize(rows, workers[cid].source.status()["frames"], native[cid])
                                    for cid, rows in observations.snapshot().items() if cid in workers and cid in native}
        _write(output / "websocket_messages.json", history)
        report["ws_event_messages"] = sum(item.get("kind") == "event" for item in history)
        report["elapsed_s"] = time.monotonic() - started
        _write(report_path, report)
    return 0 if report["ok"] else 1


def _emit(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(f"CPU multicamera: {report['status']}; {report.get('error', report.get('output_dir', ''))}")


def _terminate_group(process: Any) -> None:
    """Only signal the session created for our child, including its ffmpeg jobs.

    Always send KILL after TERM: the leader can exit while a descendant ignores
    TERM. Waiting for the leader alone does not prove that its group is gone.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            if sig == signal.SIGKILL:
                raise


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser().parse_args(argv)
        args.data_dir = args.data_dir.expanduser().resolve()
        args.models_dir = args.models_dir.expanduser().resolve()
        if args._run:
            _require(args.output_dir is not None and args._deadline is not None and math.isfinite(args._deadline), "Internal run needs output and deadline")
            args.output_dir = args.output_dir.expanduser().resolve()
            if not args.output_dir.is_dir() or any(path.name != "runtime.log" for path in args.output_dir.iterdir()):
                raise CLIError("Internal output must be empty except for its parent's runtime.log")
            return _run(args)
        started = time.monotonic()
        output = (args.output_dir or args.data_dir / "cpu_multicamera" /
                  (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])).expanduser().resolve()
        if output.exists():
            raise CLIError("--output-dir уже существует; результаты не перезаписываются")
        output.mkdir(parents=True)
    except (CLIError, ValueError, OSError, AcceptanceFailure) as exc:
        _emit({"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "runtime_started": False}, "--json" in argv)
        return 2
    report_path = output / "report.json"
    try:
        _preflight(args)
    except Exception as exc:
        report = {"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "runtime_started": False,
                  "output_dir": str(output), "preflight": {"ok": False}, "versions": _versions()}
        _write(report_path, report)
        _emit(report, args.json)
        return 1
    command = [sys.executable, "-m", "tools.cpu_multicamera", "--_run", "--output-dir", str(output),
               "--data-dir", str(args.data_dir), "--models-dir", str(args.models_dir),
               "--timeout", str(args.timeout), "--_deadline", str(started + args.timeout)]
    timed_out = False
    process = None
    try:
        with (output / "runtime.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True,
                                       env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
            try:
                process.wait(timeout=max(0.01, args.timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_group(process)
        report = json.loads(report_path.read_text()) if report_path.is_file() else {"ok": False, "status": "failed"}
    except Exception as exc:
        report = {"ok": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if process is not None and process.returncode is None:
            try:
                _terminate_group(process)
            except Exception as cleanup_exc:
                report["cleanup_error"] = type(cleanup_exc).__name__
    if timed_out or process is None or process.returncode != 0 or not report.get("ok"):
        report.update(ok=False, status="timeout" if timed_out else "failed")
        report.setdefault("error", "Общий wall timeout; owned process group terminated" if timed_out else "См. runtime.log")
    report.update(wall_s=time.monotonic() - started, runtime_returncode=process.returncode if process else None,
                  runtime_started=process is not None, output_dir=str(output))
    _write(report_path, report)
    _emit(report, args.json)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
