"""Reproducible sequential CPU S1 evaluation of both fixed manifest clips.

No network, realtime source, media jobs or quality threshold tuning. Matching is
delegated exclusively to ml.eval.events_eval, including uncertain exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ("s1_01.mp4", "s1_02.mp4")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _asset(path: Path) -> dict:
    stat = path.stat()
    _require(path.is_file(), f"Not a regular input file: {path}")
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha(path)}


def _versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for name in ("numpy", "opencv-python-headless", "onnxruntime", "pydantic", "scipy", "lap"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_recipe(path: Path) -> tuple[dict, dict, list[dict]]:
    """Validate the fixed two-clip scope before importing inference libraries."""
    recipe = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(recipe, dict) and recipe.get("schema_version") == 1, "Unsupported S1 recipe")
    _require(recipe.get("duration_s") == 60 and recipe.get("sample_fps") == 5 and recipe.get("threads") == 2,
             "Baseline requires 60 s, 5 FPS and two threads")
    _require(recipe.get("model_id") == "det" and recipe.get("rule", {}).get("type") == "zone_intrusion",
             "Baseline requires det and zone_intrusion")
    _require(isinstance(recipe.get("tol_s"), (int, float)) and not isinstance(recipe["tol_s"], bool)
             and math.isfinite(recipe["tol_s"]) and recipe["tol_s"] >= 0, "Invalid tolerance")
    from guard.contracts import RuleConfig
    from guard.tracking import Tracker

    Tracker(classes={"person"}, **recipe["tracker"])  # Reuse real tracker argument validation; no inference.
    rule = RuleConfig.model_validate({"rule_id": "s1", **recipe["rule"]})
    _require(all(math.isfinite(value) and value >= 0 for value in
                 (rule.min_duration_s, rule.close_after_s, rule.cooldown_s))
             and math.isfinite(rule.min_ratio) and 0 <= rule.min_ratio <= 1, "Invalid confirmation/cooldown settings")
    _require(math.isfinite(float(rule.params.get("min_conf", .35))) and 0 <= float(rule.params.get("min_conf", .35)) <= 1
             and math.isfinite(float(rule.params.get("min_box_h_px", 40))) and float(rule.params.get("min_box_h_px", 40)) >= 0,
             "Invalid S1 rule parameters")
    manifest = json.loads((ROOT / recipe["manifest"]).read_text(encoding="utf-8"))
    clips = [item for item in manifest["videos"] if item.get("scenario") == "S1"]
    _require(sorted(item["name"] for item in clips) == list(VIDEOS), "Manifest must contain exactly both S1 clips")
    clips.sort(key=lambda item: item["name"])
    for clip in clips:
        segments = clip["segments"]
        _require(len(segments) == 1 and segments[0].get("ss") == 0 and segments[0].get("to") == 60,
                 "S1 mapping requires one original source segment [0, 60)")
        _require(segments[0]["source"] in recipe["source_files"], "Missing source path in S1 recipe")
        _require(clip.get("zones"), "S1 clip must have explicit annotated zones")
        _require(all(event["type"] == "zone_intrusion" and 0 <= event["t_start"] <= event["t_end"] <= 60
                     and isinstance(event["uncertain"], bool) for event in clip["events"]), "Invalid S1 labels")
    certain = sum(not event["uncertain"] for clip in clips for event in clip["events"])
    uncertain = sum(event["uncertain"] for clip in clips for event in clip["events"])
    _require((certain, uncertain) == (4, 3), "Expected exactly 4 certain and 3 uncertain S1 labels")
    return recipe, manifest, clips


def sample_plan(fps: float, duration_s: float = 60.0, sample_fps: float = 5.0) -> list[dict]:
    """First source frame at/after every fixed 0.2 s slot, excluding t=60."""
    _require(math.isfinite(fps) and fps >= sample_fps > 0, "Source FPS must be finite and >= sample FPS")
    _require(math.isfinite(duration_s) and duration_s > 0, "Duration must be positive")
    count = math.ceil(duration_s * sample_fps - 1e-9)
    plan = [{"sample_index": index, "target_s": index / sample_fps,
             "source_frame_index": math.ceil(index * fps / sample_fps - 1e-9)} for index in range(count)]
    _require(len({row["source_frame_index"] for row in plan}) == len(plan), "Duplicate sampled frame")
    _require(all(row["source_frame_index"] / fps < duration_s for row in plan), "Sample reaches excluded duration boundary")
    return plan


def labels_from_clips(clips: list[dict]):
    from ml.eval.events_eval import LabelEvent

    labels, records = [], []
    for clip in clips:
        for index, event in enumerate(clip["events"]):
            labels.append(LabelEvent(video=clip["name"], **event))
            records.append({"label_id": f"{clip['name']}:{index}", "video": clip["name"], **event})
    return labels, records


def score_predictions(labels, predictions, durations: dict, tolerance: float, label_records: list[dict]) -> dict:
    """Use existing evaluator for every classification, including exclusion IDs.

    Prefix replay exposes its decisions without implementing another matcher.
    Supplemental strict metrics only use its returned integer counters.
    """
    from ml.eval.events_eval import evaluate

    result = evaluate(labels, predictions, tolerance, durations=durations)
    _require(len(labels) == len(label_records), "Label IDs do not cover all labels")
    classifications = []
    ordered = sorted(predictions, key=lambda pred: (pred.video, pred.type, pred.confirmed, pred.start, pred.id))
    previous = evaluate(labels, [], tolerance, durations=durations)["overall"]
    counters = ("tp", "fp", "duplicates", "ignored_uncertain")
    for index, pred in enumerate(ordered):
        current = evaluate(labels, ordered[: index + 1], tolerance, durations=durations)["overall"]
        changed = [key for key in counters if current[key] - previous[key] == 1]
        _require(len(changed) == 1, "Evaluator prefix replay did not produce one prediction classification")
        classification = changed[0]
        entry = {"prediction_id": pred.id, "video": pred.video, "classification": classification,
                 "reason": f"existing_events_eval:{classification}"}
        if classification == "ignored_uncertain":
            entry["uncertain_label_ids"] = [record["label_id"] for label, record in zip(labels, label_records)
                                            if label.uncertain and evaluate([label], [pred], tolerance,
                                                durations=durations)["overall"]["ignored_uncertain"] == 1]
        classifications.append(entry)
        previous = current

    def strict(metrics: dict, exposure: float) -> dict:
        denominator = metrics["tp"] + metrics["fp"] + metrics["duplicates"]
        return {"precision_denominator_evaluator": metrics["tp"] + metrics["fp"],
                "raw_predictions": metrics["tp"] + metrics["fp"] + metrics["duplicates"] + metrics["ignored_uncertain"],
                "precision_denominator_strict": denominator,
                "recall_denominator": metrics["tp"] + metrics["fn"],
                "strict_precision": metrics["tp"] / denominator if denominator else 0.0,
                "alarms_false_or_duplicate_per_hour": (metrics["fp"] + metrics["duplicates"]) / (exposure / 3600),
                "positive_video_exposure_s": exposure,
                "independent_clean_video_exposure_s": 0,
                "predictions_excluded_uncertain": metrics["ignored_uncertain"]}

    return {"evaluator": result, "supplemental_strict": strict(result["overall"], sum(durations.values())),
            "label_counts": {"certain": sum(not label.uncertain for label in labels),
                             "uncertain": sum(label.uncertain for label in labels)},
            "excluded_uncertain_labels": [row for row in label_records if row["uncertain"]],
            "prediction_classifications": classifications,
            "uncertain_exclusions": [row for row in classifications if row["classification"] == "ignored_uncertain"]}


def _deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("S1 run exceeded shared wall deadline")


def persist_lifecycle_updates(journal, closed, reopened):
    """Persist final lifecycle state without replacing operator/origin details."""
    reopened_ids = {event.id for event in reopened}
    closed = [event for event in closed if event.id not in reopened_ids]
    for event in [*closed, *reopened]:
        details = {key: event.details[key] for key in ("reopened_count", "active_duration_s") if key in event.details}
        journal.update(event.id, ended_at=event.ended_at, **({"details_patch": details} if details else {}))
    return closed


def _run(args) -> int:
    output = args.output_dir.resolve()
    report = {"schema_version": 1, "scope": "sequential S1 component quality evaluation",
              "status": "running", "ok": False, "versions": _versions(), "per_video": {},
              "assets": {"runner": _asset(Path(__file__))},
              "coverage": {"videos_expected": 2, "videos_completed": 0, "selected_frames_expected": 600, "frames_processed": 0},
              "limitations": ["two positive videos, no independent clean S1 exposure", "fixed nominal-CFR source-time sampling",
                              "no async runtime, network, RTSP, media or TPU", "no production quality claim or threshold tuning"]}
    started = time.monotonic()
    backend = None
    try:
        from guard.pipeline.config import load_models
        from guard.inference.base import resolve_model_path

        recipe, manifest, clips = load_recipe(args.config)
        data_dir = (args.data_dir or Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data")).expanduser().resolve()
        models_dir = (args.models_dir or Path(os.environ.get("GUARD_MODELS_DIR") or ROOT / "models")).expanduser().resolve()
        registry = ROOT / recipe["model_registry"]
        model = load_models(registry, models_dir)["det"]
        files = {clip["name"]: data_dir / recipe["source_files"][clip["segments"][0]["source"]]["path"] for clip in clips}
        report["assets"].update({"config": _asset(args.config), "manifest": _asset(ROOT / recipe["manifest"]),
                                  "model_registry": _asset(registry), "det_onnx": _asset(resolve_model_path(model.onnx_path, models_dir)),
                                  "evaluator": _asset(ROOT / "ml/eval/events_eval.py")})
        if (models_dir / "det.json").is_file():
            report["assets"]["det_metadata"] = _asset(models_dir / "det.json")
        for clip in clips:
            source = clip["segments"][0]["source"]
            details = _asset(files[clip["name"]])
            _require(details["sha256"] == recipe["source_files"][source]["sha256"], "Source video SHA256 differs from fixed baseline")
            report["assets"][clip["name"]] = details
        report["effective_recipe"] = recipe
        report["git"] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                           "tracked_changes": subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()}
        _write(output / "effective_config.json", recipe)
        _write(output / "manifest.json", manifest)
        labels, label_records = labels_from_clips(clips)
        labels_dir = output / "labels"
        labels_dir.mkdir()
        for clip in clips:
            _write(labels_dir / f"{Path(clip['name']).stem}.json", {"video": clip["name"], "events": clip["events"]})
        _write(output / "labels_with_ids.json", label_records)
        mapping = {f"eval_{Path(clip['name']).stem}": {"video": clip["name"], "t0": 0.0,
                    "source_segment_start_s": 0.0, "capture_id": f"s1-quality:{clip['name']}"} for clip in clips}
        _write(output / "mapping.json", mapping)
        report["label_counts"] = {"certain": 4, "uncertain": 3}
        _write(output / "report.json", report)

        # Only now import / construct real inference, after all local inputs pass.
        import cv2
        import onnxruntime as ort
        from guard.contracts import CameraConfig, FrameContext
        from guard.pipeline import compat
        from guard.pipeline.detector import Detector
        from guard.inference.onnx_cpu import OnnxCpuBackend
        from guard.rules.base import build_rules, load_plugins
        from ml.eval.events_eval import predictions_from_events

        _require(not any(compat.FALLBACKS.values()), "Evaluation refuses fallback components")
        ort.disable_telemetry_events()
        cv2.setNumThreads(2)
        backend = OnnxCpuBackend(models_dir=models_dir, intra_op_threads=2)
        backend.load(model)
        _require("det" in backend.loaded_models(), "det was not actually loaded")
        if not model.classes:
            model = model.model_copy(update={"classes": backend.class_names("det") or []})
        _require("person" in model.classes, "det class names must resolve person, as in runtime Detector setup")
        _write(output / "effective_model.json", model.model_dump(mode="json"))
        detector = Detector(model, backend)
        manager = compat.EventManager()
        load_plugins()
        predictions, durations = [], {}
        _write(output / "predictions.json", [])
        _write(output / "durations.json", {})
        with compat.Journal(output / "guard.db") as journal:
            for clip in clips:
                _deadline(args._deadline)
                video = clip["name"]
                camera_id = f"eval_{Path(video).stem}"
                camera = CameraConfig.model_validate({"camera_id": camera_id, "name": clip["title"], "site": "S1 evaluation",
                    "source": {"type": "file", "uri": str(files[video]), "loop": False, "realtime": False},
                    "sample_fps": 5.0, "zones": clip["zones"],
                    "rules": [{"rule_id": "s1", "zones": [zone["zone_id"] for zone in clip["zones"]], **recipe["rule"]}]})
                _write(output / f"camera_{Path(video).stem}.json", camera.model_dump(mode="json"))
                rules = build_rules(camera)
                _require(len(rules) == 1 and rules[0].type == "zone_intrusion" and rules[0].required_models == frozenset({"det"}),
                         "Expected active real zone_intrusion with det")
                tracker = compat.Tracker(classes={"person"}, **recipe["tracker"])
                cap = cv2.VideoCapture(str(files[video]))
                _require(cap.isOpened(), f"Cannot open {video}")
                fps = cap.get(cv2.CAP_PROP_FPS)
                plan = sample_plan(fps)
                wanted = {row["source_frame_index"]: row for row in plan}
                expected_source = math.ceil(60 * fps - 1e-9)
                stats = {"source_fps": fps, "source_frames_expected": expected_source, "source_frames_traversed": 0,
                         "selected_frames_expected": len(plan), "selected_frames_decoded": 0, "frames_processed": 0,
                         "detections": 0, "person_track_observations": 0, "frames_with_violations": 0,
                         "created_events": 0, "close_notifications": 0, "reopen_transitions": 0,
                         "effective_tracker": recipe["tracker"], "duration_s": None}
                report["per_video"][video] = {"coverage": stats}
                _write(output / "report.json", report)
                video_events = []
                video_started = time.monotonic()
                trace_path = output / f"samples_{Path(video).stem}.jsonl"
                lifecycle_path = output / f"lifecycle_{Path(video).stem}.jsonl"
                try:
                    with trace_path.open("w", encoding="utf-8") as trace, lifecycle_path.open("w", encoding="utf-8") as lifecycle:
                        for frame_index in range(expected_source):
                            _deadline(args._deadline)
                            _require(cap.grab(), f"Early EOF in {video} at source frame {frame_index}; incomplete evaluation")
                            stats["source_frames_traversed"] += 1
                            if frame_index not in wanted:
                                continue
                            ok, frame = cap.retrieve()
                            _require(ok and frame is not None and frame.size, f"Decode failure in {video} frame {frame_index}")
                            stats["selected_frames_decoded"] += 1
                            pos_s = frame_index / fps
                            timings = {}
                            detections = detector.detect(frame, timings)
                            people = tracker.update([item for item in detections if item.cls == "person"], pos_s)
                            compat.assign_attributes(people, {"det": detections}, pos_s)
                            context = FrameContext(camera=camera, ts=pos_s, frame_idx=frame_index, frame=frame,
                                                   detections={"det": detections}, people=people, timings=timings,
                                                   model_observations={"det": "observed"})
                            violations = [violation for rule in rules for violation in rule.evaluate(context)]
                            new, closed, reopened_events = manager.process_with_updates(camera, pos_s, violations, timings)
                            for event in new:
                                event.details = {**event.details, "origin": {"capture_id": mapping[camera_id]["capture_id"],
                                    "frame_seq": frame_index, "source_pos_s": pos_s, "capture_ts": pos_s}}
                                journal.add(event)
                                predictions.append(event.model_copy(deep=True))
                                video_events.append(event.model_copy(deep=True))
                            if new:
                                _write(output / "predictions.json", [event.model_dump(mode="json") for event in predictions])
                            closed = persist_lifecycle_updates(journal, closed, reopened_events)
                            reopened = sorted(event.id for event in reopened_events)
                            if new or closed or reopened:
                                lifecycle.write(json.dumps({"source_pos_s": pos_s, "created": [e.id for e in new],
                                    "closed": [{"id": e.id, "ended_at": e.ended_at} for e in closed], "reopened": reopened}) + "\n")
                            stats["frames_processed"] += 1
                            report["coverage"]["frames_processed"] += 1
                            stats["detections"] += len(detections)
                            stats["person_track_observations"] += len(people)
                            stats["frames_with_violations"] += bool(violations)
                            stats["created_events"] += len(new)
                            stats["close_notifications"] += len(closed)
                            stats["reopen_transitions"] += len(reopened)
                            trace.write(json.dumps({**wanted[frame_index], "source_time_s": pos_s,
                                "decoder_pos_s": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                "capture_id": mapping[camera_id]["capture_id"], "detections": len(detections),
                                "model_observations": dict(context.model_observations),
                                "rule_observations": dict(context.rule_observations),
                                "raw_detections": [{"cls": item.cls, "conf": item.conf, "bbox": item.bbox} for item in detections],
                                "person_tracks": [{"id": track.track_id, "bbox": track.bbox} for track in people],
                                "violations": [{"track_id": item.track_id, "zone_id": item.zone_id, "bbox": item.bbox,
                                                "score": item.score} for item in violations],
                                "new_event_ids": [event.id for event in new], "timings_ms": timings}) + "\n")
                            if stats["frames_processed"] % 25 == 0 or new:
                                trace.flush()
                                lifecycle.flush()
                                _write(output / "report.json", report)
                finally:
                    cap.release()
                    stats["wall_s"] = time.monotonic() - video_started
                    _write(output / "report.json", report)
                _require(stats["frames_processed"] == stats["selected_frames_decoded"] == len(plan)
                         and stats["source_frames_traversed"] == expected_source, "Incomplete sample coverage")
                stats["duration_s"] = durations[video] = 60.0
                report["per_video"][video].update(score_predictions(
                    [label for label in labels if label.video == video], predictions_from_events(video_events, mapping),
                    {video: 60.0}, recipe["tol_s"], [row for row in label_records if row["video"] == video]))
                report["per_video"][video]["right_censored_open_events"] = [event.model_dump(mode="json") for event in manager.open_events(camera_id)]
                report["coverage"]["videos_completed"] += 1
                _write(output / "predictions.json", [event.model_dump(mode="json") for event in predictions])
                _write(output / "durations.json", durations)
                _write(output / "report.json", report)
            rows, total = journal.query(limit=len(predictions) + 1)
            _require(total == len(predictions) == len(rows), "Journal/event prediction count or complete readback mismatch")
            _write(output / "journal_events.json", [event.model_dump(mode="json") for event in rows])
        report["global"] = score_predictions(labels, predictions_from_events(predictions, mapping), durations, recipe["tol_s"], label_records)
        _write(output / "evaluation.json", report["global"]["evaluator"])
        report["inference_stats"] = backend.stats()
        _require(report["inference_stats"]["models"]["det"]["errors"] == 0, "Inference errors present")
        report["algorithmic_latency_definition"] = "source confirmed time minus matched label start; wall time is separate"
        report["exposure_definition"] = "120 s of positive-video exposure, includes uncertain intervals; no independent clean S1 zones"
        report["coverage"]["selected_frames_expected"] = 600
        report["coverage"]["frames_processed"] = sum(row["coverage"]["frames_processed"] for row in report["per_video"].values())
        for value in report["assets"].values():
            _require(_sha(Path(value["path"])) == value["sha256"], "Input content changed during evaluation")
        report["output_hashes"] = {str(path.relative_to(output)): _asset(path) for path in output.rglob("*") if path.is_file()
                                   and path.name not in {"report.json", "runtime.log"}}
        report.update(ok=True, status="completed")
    except Exception as exc:
        report.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            backend.close()
        report["wall_s"] = time.monotonic() - started
        _write(output / "report.json", report)
    return 0 if report["ok"] else 1


def _timeout(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 1800:
        raise argparse.ArgumentTypeError("timeout must be in (0, 1800] seconds")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=ROOT / "configs/eval-s1.json")
    result.add_argument("--data-dir", type=Path)
    result.add_argument("--models-dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--timeout", type=_timeout, default=180.0,
                        help="shared wall deadline in seconds, up to 1800 for heavy candidates")
    result.add_argument("--json", action="store_true")
    result.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args._run:
        return _run(args)
    started = time.monotonic()
    if args.output_dir is None:
        data = (args.data_dir or Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data")).expanduser()
        args.output_dir = data / "s1_evaluation" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        arguments.error("output-dir exists; evaluations are never overwritten")
    output.mkdir(parents=True)
    command = [sys.executable, "-m", "ml.eval.run_s1", "--_run", "--config", str(args.config.resolve()),
               "--output-dir", str(output), "--timeout", str(args.timeout), "--_deadline", str(started + args.timeout - 2)]
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
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    pass
    path = output / "report.json"
    report = json.loads(path.read_text()) if path.is_file() else {"scope": "sequential S1 component quality evaluation", "versions": _versions()}
    if timed_out or process.returncode != 0 or not report.get("ok"):
        report.update(ok=False, status="timeout" if timed_out else "failed")
        report.setdefault("error", "Shared wall timeout" if timed_out else "See runtime.log")
    report.update(output_dir=str(output), total_wall_s=time.monotonic() - started, runtime_returncode=process.returncode)
    report.setdefault("output_hashes", {})["runtime.log"] = _asset(output / "runtime.log")
    _write(path, report)
    print(json.dumps(report, ensure_ascii=False) if args.json else f"S1 evaluation: {report['status']}; report={path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
