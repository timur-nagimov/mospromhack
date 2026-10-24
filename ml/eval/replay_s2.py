"""Pinned offline S2 postprocessed-detection replay; never loads a model/video.

Continuous control must reproduce cached tracks, lifecycle, journal and metrics
before an optional independently evidenced source-boundary diagnostic can run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ml.eval import capture_boundary, run_s2

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SAMPLES = 831
SAMPLE_KEYS = ("sample_index", "target_s", "source_frame_index", "source_time_s")
TRACE_KEYS = ("identity_sha256", "coco_tracks_before_attributes", "raw_detections", "person_tracks", "violations")
require, asset, write, fingerprint = run_s2.require, run_s2.asset, run_s2.write, run_s2.audit.fingerprint


def read(path: Path):
    def bad_number(value):
        raise ValueError(f"Non-finite JSON number in {path.name}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad_number)


def lines(path: Path) -> list[dict]:
    def bad_number(value):
        raise ValueError(f"Non-finite JSON number in {path.name}: {value}")
    rows = path.read_text(encoding="utf-8").splitlines()
    require(all(line.strip() for line in rows), f"Blank cache row: {path.name}")
    return [json.loads(line, parse_constant=bad_number) for line in rows]


def pinned_file(path: Path, expected: dict) -> dict:
    require(path.is_file() and not path.is_symlink(), f"Missing/nonregular cache file: {path.name}")
    info = asset(path)
    require(info["sha256"] == expected["sha256"] and info["bytes"] == expected["bytes"],
            f"Pinned cache bytes/hash mismatch: {path.name}")
    return info


@dataclass
class Cache:
    directory: Path
    report: dict
    recipe: dict
    rows: dict
    lifecycle: dict
    cameras: dict
    mapping: dict
    durations: dict
    original: list
    primary: list
    predictions: list
    journal: list
    files: dict


def load_cache(directory: Path, report_sha256: str) -> Cache:
    """Fail closed on incomplete/tampered caches before importing runtime code."""
    directory = directory.resolve()
    require(re.fullmatch(r"[a-f0-9]{64}", report_sha256) is not None, "Explicit report SHA256 is required")
    report_path = directory / "report.json"
    report_info = asset(report_path)
    require(report_info["sha256"] == report_sha256, "Source report SHA256 mismatch")
    report = read(report_path)
    require(report["schema_version"] == 1 and report["status"] == "completed" and report["ok"] is True,
            "Source S2 evaluation must be complete and successful")
    require(report.get("ppe_input_mode", "full_frame") == "full_frame",
            "Legacy flat replay does not support person_crop ownership/masks")
    require(report["candidate"] in run_s2.MODEL_SHA and report["candidate"] != "det", "Unknown source PPE candidate")
    names = ["effective_config.json", "manifest.snapshot.json", "mapping.json", "durations.json", "original_labels.json",
             "primary_labels.json", "scope_exclusions.json", "predictions.json", "journal_events.json"]
    names += [f"{kind}_{Path(video).stem}.{suffix}" for video in run_s2.VIDEOS
              for kind, suffix in (("camera", "json"), ("samples", "jsonl"), ("lifecycle", "jsonl"))]
    files = {"report.json": report_info}
    for name in names:
        files[name] = pinned_file(directory / name, report["output_hashes"][name])
    recipe = run_s2.load_recipe(directory / "effective_config.json")
    require(recipe == report["effective_recipe"], "Source effective recipe differs from report")
    require(files["manifest.snapshot.json"]["sha256"] == recipe["manifest_sha256"], "Source manifest pin mismatch")
    manifest = read(directory / "manifest.snapshot.json")
    clips = [next(clip for clip in manifest["videos"] if clip["name"] == video) for video in run_s2.VIDEOS]
    original, primary, exclusions = run_s2.label_scopes(clips, recipe)
    require(original == read(directory / "original_labels.json") and primary == read(directory / "primary_labels.json")
            and exclusions == read(directory / "scope_exclusions.json"), "Cached GT/scope differs from frozen manifest")
    durations, mapping = read(directory / "durations.json"), read(directory / "mapping.json")
    require(set(durations) == set(run_s2.VIDEOS) and len(mapping) == len(run_s2.VIDEOS), "Incomplete clip durations/mapping")
    rows, lifecycle, cameras = {}, {}, {}
    for model_id in ("det", "ppe"):
        model = report["effective_models"][model_id]
        policy = recipe["det"] if model_id == "det" else recipe["ppe_thresholds"]
        expected_sha = recipe["det"]["sha256"] if model_id == "det" else recipe["candidates"][report["candidate"]]["sha256"]
        require(model["conf_threshold"] == policy["conf_threshold"] and model["iou_threshold"] == policy["iou_threshold"]
                and model["output_format"] == "yolov8" and model["input_size"] == [640, 640]
                and model["input_dtype"] == "float32" and model["tpu_program"] is None
                and report["assets"][f"{model_id}_onnx"]["sha256"] == expected_sha, "Cached model/threshold provenance drift")
        require(len(model["classes"]) == (80 if model_id == "det" else 14), "Missing source class names")
        require(model["classes"][0] == "person" if model_id == "det" else
                model["classes"] == recipe["candidates"][report["candidate"]]["classes"], "Source class mapping drift")
    for video in run_s2.VIDEOS:
        stem = Path(video).stem
        prepared = report["prep_provenance"]["encode"]["outputs"][video]
        require(prepared["fps"] == recipe["prepared_fps"] and prepared["frames"] > 0
                and durations[video] == prepared["frames"] / prepared["fps"], "Cached duration/preparation drift")
        camera = cameras[video] = read(directory / f"camera_{stem}.json")
        require(camera["camera_id"] == f"eval_{stem}" and camera["rules"] == [{"rule_id": "s2", **recipe["rule"]}]
                and camera["zones"] == [] and camera["sample_fps"] == recipe["sample_fps"], "Cached camera/rule drift")
        require(mapping[camera["camera_id"]]["video"] == video and mapping[camera["camera_id"]]["t0"] == 0,
                "Cached camera mapping drift")
        require(mapping[camera["camera_id"]]["source_segments"] == next(clip["segments"] for clip in clips if clip["name"] == video),
                "Cached mapping source segments differ from manifest")
        cached = rows[video] = lines(directory / f"samples_{stem}.jsonl")
        plan = run_s2.run_s1.sample_plan(prepared["fps"], durations[video], recipe["sample_fps"])
        coverage = report["per_video"][video]["coverage"]
        require(len(cached) == len(plan) == coverage["frames_processed"] == coverage["selected_frames_expected"]
                == coverage["selected_frames_decoded"] and coverage["source_frames_traversed"] == prepared["frames"],
                f"Incomplete source cache: {video}")
        for row, sample in zip(cached, plan):
            expected = {**sample, "source_time_s": sample["source_frame_index"] / prepared["fps"]}
            require({key: row[key] for key in SAMPLE_KEYS} == expected
                    and row["capture_id"] == f"s2-quality:{video}" and math.isfinite(row["decoder_pos_s"]),
                    f"Cached sample coordinate/order drift: {video}")
            require(row.get("ppe_input_mode", "full_frame") == "full_frame",
                    "Legacy flat replay does not support person_crop ownership/masks")
            require(set(row["raw_detections"]) == set(row["raw_outputs"]) == set(row["per_model_timings_ms"]) == {"det", "ppe"},
                    "Missing model cache for a source sample")
            for model_id in ("det", "ppe"):
                model = report["effective_models"][model_id]
                for det in row["raw_detections"][model_id]:
                    require(set(det) == {"cls", "cls_id", "conf", "bbox"}
                            and isinstance(det["cls_id"], int) and not isinstance(det["cls_id"], bool)
                            and 0 <= det["cls_id"] < len(model["classes"])
                            and model["classes"][det["cls_id"]] == det["cls"]
                            and model["conf_threshold"] <= det["conf"] <= 1
                            and len(det["bbox"]) == 4 and all(math.isfinite(value) for value in det["bbox"])
                            and det["bbox"][2] > det["bbox"][0] and det["bbox"][3] > det["bbox"][1],
                            "Malformed cached postprocessed detection")
                raw = row["raw_outputs"][model_id]
                channels = 4 + len(model["classes"])
                require(len(raw) == 1 and raw[0]["shape"] == [1, channels, 8400]
                        and raw[0]["dtype"] == "float32" and raw[0]["bytes"] == channels * 8400 * 4
                        and raw[0]["finite"] is True and re.fullmatch(r"[a-f0-9]{64}", raw[0]["sha256"]) is not None,
                        "Missing/malformed original raw-output hash (tensor itself is not cached)")
                require(all(math.isfinite(value) and value >= 0 for value in row["per_model_timings_ms"][model_id].values()),
                        "Invalid cached model timings")
            identity = {**expected, "coco_people": row["coco_tracks_before_attributes"]}
            require(fingerprint(identity) == row["identity_sha256"], "Cached COCO identity fingerprint mismatch")
        identity_sha = hashlib.sha256("".join(row["identity_sha256"] + "\n" for row in cached).encode()).hexdigest()
        require(identity_sha == report["per_video"][video]["coco_track_identity_sha256"], "Aggregate identity hash mismatch")
        lifecycle[video] = lines(directory / f"lifecycle_{stem}.jsonl")
    require(sum(map(len, rows.values())) == EXPECTED_SAMPLES == report["coverage"]["frames_processed"]
            == report["coverage"]["selected_frames_expected"] and report["coverage"]["videos_completed"] == 4,
            "Replay requires the complete 831-sample, four-video source cache")
    return Cache(directory, report, recipe, rows, lifecycle, cameras, mapping, durations, original, primary,
                 read(directory / "predictions.json"), read(directory / "journal_events.json"), files)


def load_boundaries(path: Path, sha256: str, cache: Cache) -> dict[str, dict]:
    return capture_boundary.load_boundaries(path, sha256, assets=cache.report["assets"],
        encoded=cache.report["prep_provenance"]["encode"],
        segments={video: cache.mapping[camera["camera_id"]]["source_segments"]
                  for video, camera in cache.cameras.items()}, samples=cache.rows)


def replace_ids(value, mapping):
    if isinstance(value, dict):
        return {key: replace_ids(item, mapping) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [replace_ids(item, mapping) for item in value]
    return mapping.get(value, value) if isinstance(value, str) else value


def event_ids(events: list[dict]) -> dict:
    counts, result = {}, {}
    for event in events:
        camera = event["camera_id"]
        counts[camera] = counts.get(camera, 0) + 1
        require(event["id"] not in result, "Duplicate source event ID")
        result[event["id"]] = f"{camera}:event-{counts[camera]}"
    return result


def normalized_events(events, ids):
    return sorted(replace_ids(events, ids), key=lambda event: event["id"])


def normalized_lifecycle(rows, ids):
    result = replace_ids(rows, ids)
    for row in result:
        row["created"].sort()
        row["closed"].sort(key=lambda event: event["id"])
        row["reopened"].sort()
    return result


class CachedDetector:
    """Returns fresh Detection objects; all cached values/timings stay unchanged."""
    def __init__(self, name: str):
        self.name, self.row = name, None

    def detect(self, frame, timings):
        from guard.contracts import Detection
        timings.update(self.row["per_model_timings_ms"][self.name])
        return [Detection(model=self.name, cls=det["cls"], cls_id=det["cls_id"], conf=det["conf"], bbox=tuple(det["bbox"]))
                for det in self.row["raw_detections"][self.name]]


def replay(cache: Cache, output: Path, boundaries: dict | None = None) -> dict:
    import numpy as np
    from guard.contracts import CameraConfig
    from guard.pipeline import compat
    from guard.rules.base import build_rules, load_plugins
    from ml.eval.events_eval import predictions_from_events

    boundaries = boundaries or {}
    output.mkdir()
    load_plugins()
    require(not any(compat.FALLBACKS.values()), "Replay refuses fallback components")
    predictions, per_video = [], {}
    with compat.Journal(output / "guard.db") as journal:
        for video in run_s2.VIDEOS:
            camera = CameraConfig.model_validate(cache.cameras[video])
            def new_state():
                return (compat.Tracker(classes={"person"}, **cache.recipe["tracker"]), compat.EventManager(), build_rules(camera))
            tracker, manager, rules = new_state()
            detectors = {name: CachedDetector(name) for name in ("det", "ppe")}
            prepared = cache.report["prep_provenance"]["encode"]["outputs"][video]
            frame = np.zeros((prepared["height"], prepared["width"], 3), np.uint8)
            lifecycle, traces, resets = [], [], []
            stats = dict(created_events=0, close_notifications=0, reopen_transitions=0, frames_with_violations=0,
                         frames_processed=0, boundary_closures=0)
            boundary, did_reset = boundaries.get(video), False
            for index, row in enumerate(cache.rows[video]):
                if boundary and not did_reset and row["source_frame_index"] >= boundary["cut_before_source_frame"]:
                    # Production API closes each incident at its measured last violation,
                    # then forgets all confirmation/cooldown/alias state for this capture.
                    closed = capture_boundary.close_capture(journal, manager, camera.camera_id)
                    resets.append({"before_sample_index": row["sample_index"], "before_source_frame_index": row["source_frame_index"],
                                   "boundary": boundary, "closed": [event.model_dump(mode="json") for event in closed],
                                   "closure_policy": "production last violation; reset Tracker + EventManager + rules"})
                    stats["boundary_closures"] += len(closed)
                    tracker, manager, rules = new_state()
                    did_reset = True
                for detector in detectors.values():
                    detector.row = row
                coordinates = {key: row[key] for key in SAMPLE_KEYS}
                result = run_s2.process_frame(camera, frame, coordinates, detectors, tracker, manager, rules)
                effective_capture_id = capture_boundary.effective_capture_id(row["capture_id"], boundary, int(did_reset))
                for event in result["new"]:
                    origin = {"capture_id": effective_capture_id, "frame_seq": row["source_frame_index"],
                              "source_pos_s": row["source_time_s"]}
                    if boundary:
                        origin.update(source_capture_id=row["capture_id"], segment_index=int(did_reset),
                                      source_row_sha256=fingerprint(row))
                    event.details = {**event.details, "origin": origin}
                    journal.add(event)
                    predictions.append(event.model_copy(deep=True))
                run_s2.run_s1.persist_lifecycle_updates(journal, result["closed"], result["reopened_events"])
                if result["new"] or result["closed"] or result["reopened"]:
                    lifecycle.append({"source_time_s": row["source_time_s"], "created": [event.id for event in result["new"]],
                                      "closed": [{"id": event.id, "ended_at": event.ended_at} for event in result["closed"]],
                                      "reopened": result["reopened"]})
                trace = copy.deepcopy(result["trace"])
                trace.update({key: row[key] for key in ("capture_id", "decoder_pos_s", "raw_outputs")})
                if boundary:
                    trace.update(capture_id=effective_capture_id, source_capture_id=row["capture_id"])
                trace["detection_sha256"] = fingerprint(row["raw_detections"])
                trace["source_row_sha256"] = fingerprint(row)
                trace["segment_index"] = int(did_reset)
                traces.append(trace)
                stats["frames_processed"] += 1
                stats["created_events"] += len(result["new"])
                stats["close_notifications"] += len(result["closed"])
                stats["reopen_transitions"] += len(result["reopened"])
                stats["frames_with_violations"] += bool(trace["violations"])
            require(not boundary or did_reset, "Configured boundary never reached")
            stem = Path(video).stem
            for name, rows in ((f"samples_{stem}.jsonl", traces), (f"lifecycle_{stem}.jsonl", lifecycle)):
                with (output / name).open("x", encoding="utf-8") as stream:
                    for row in rows:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            per_video[video] = {"coverage": stats, "boundary_resets": resets,
                                "right_censored_open_events": [event.model_dump(mode="json") for event in manager.open_events(camera.camera_id)]}
        journal_rows, total = journal.query(limit=len(predictions) + 1)
        require(total == len(predictions) == len(journal_rows), "Replay journal readback incomplete")
        journal_events = [event.model_dump(mode="json") for event in journal_rows]
    serialized = [event.model_dump(mode="json") for event in predictions]
    write(output / "predictions.json", serialized)
    write(output / "journal_events.json", journal_events)
    evaluation = run_s2.evaluate_scopes(cache.original, cache.primary, predictions_from_events(predictions, cache.mapping),
                                        cache.durations, cache.recipe)
    result = {"frames_processed": sum(row["coverage"]["frames_processed"] for row in per_video.values()),
              "per_video": per_video, "evaluation": evaluation, "predictions": serialized, "journal_events": journal_events}
    write(output / "report.json", result)
    return result


def control_parity(cache: Cache, result: dict, output: Path) -> dict:
    source_ids, replay_ids = event_ids(cache.predictions), event_ids(result["predictions"])
    require(normalized_events(cache.predictions, source_ids) == normalized_events(result["predictions"], replay_ids),
            "Continuous-control confirmation snapshots differ from source")
    require(normalized_events(cache.journal, source_ids) == normalized_events(result["journal_events"], replay_ids),
            "Continuous-control journal/closure state differs from source")
    require(replace_ids(cache.report["evaluation"], source_ids) == replace_ids(result["evaluation"], replay_ids),
            "Continuous-control metrics/classifications differ from source")
    for video in run_s2.VIDEOS:
        replayed = lines(output / f"samples_{Path(video).stem}.jsonl")
        for old, new in zip(cache.rows[video], replayed):
            require(all(fingerprint(old[key]) == fingerprint(new[key]) for key in TRACE_KEYS)
                    and replace_ids(old["new_event_ids"], source_ids) == replace_ids(new["new_event_ids"], replay_ids),
                    f"Continuous-control track/attribute/rule/sample drift at {video} sample {old['sample_index']}")
        lifecycle = lines(output / f"lifecycle_{Path(video).stem}.jsonl")
        require(normalized_lifecycle(cache.lifecycle[video], source_ids) == normalized_lifecycle(lifecycle, replay_ids),
                f"Continuous-control lifecycle transitions differ: {video}")
        previous = cache.report["per_video"][video]
        current = result["per_video"][video]
        require(all(previous["coverage"][key] == current["coverage"][key] for key in current["coverage"] if key != "boundary_closures")
                and normalized_events(previous["right_censored_open_events"], source_ids)
                == normalized_events(current["right_censored_open_events"], replay_ids), "Continuous-control counters/right censoring drift")
    return {"status": "matched", "samples": result["frames_processed"], "uuid_comparison": "camera + creation ordinal; semantic equality",
            "compared": ["all track identities/attributes/violations/detections", "confirmation snapshots including cached model timings",
                         "all lifecycle transitions and final journal/right-censoring", "both metric scopes and classifications"]}


def run(source: Path, report_sha256: str, output: Path, boundary_map: Path | None = None, boundary_sha256: str | None = None) -> int:
    require(not output.exists(), "Output directory exists; diagnostic artifacts are never overwritten")
    require(source.resolve() not in output.resolve().parents, "Output must be outside the immutable source cache")
    output.mkdir(parents=True)
    report = {"schema_version": 1, "status": "running", "ok": False, "scope": "postprocessed-detection offline diagnostic replay",
              "source_report_sha256": report_sha256, "inference_executed": False, "quality_primary": False,
              "timing_policy": "cached per-model timings retained; new tracking/rule/processor wall times describe replay only",
              "limitations": ["cached postprocessed detections only; raw ONNX tensors/pixels are not replayed",
                  "no actor GT; interval/type matches do not establish actor identity or association precision",
                  "explicit known source boundary only; no automatic scene-cut detector or deployment continuity claim",
                  "boundary treatment is diagnostic; the unchanged continuous benchmark remains primary",
                  "no async/live/media/TPU claim; training overlap remains unknown"]}
    try:
        cache = load_cache(source, report_sha256)
        require((boundary_map is None) == (boundary_sha256 is None), "Boundary map and its explicit SHA256 must be supplied together")
        boundaries = load_boundaries(boundary_map, boundary_sha256, cache) if boundary_map else {}
        report.update(candidate=cache.report["candidate"], effective_recipe=cache.recipe,
                      effective_models=cache.report["effective_models"], source_files=cache.files,
                      boundary_map=asset(boundary_map) if boundary_map else None, boundaries=boundaries,
                      replay_code=run_s2.audit.code_snapshot())
        frozen = output / "source_cache"
        frozen.mkdir()
        for name in cache.files:
            shutil.copyfile(cache.directory / name, frozen / name)
        if boundary_map:
            shutil.copyfile(boundary_map, output / "boundary-map.json")
        baseline = replay(cache, output / "continuous-control")
        report["control_parity"] = control_parity(cache, baseline, output / "continuous-control")
        report["continuous_control"] = {key: value for key, value in baseline.items() if key not in ("predictions", "journal_events")}
        if boundaries:
            treatment = replay(cache, output / "source-boundary-diagnostic", boundaries)
            require(treatment["frames_processed"] == baseline["frames_processed"], "Treatment sample coverage changed")
            report["source_boundary_diagnostic"] = {key: value for key, value in treatment.items() if key not in ("predictions", "journal_events")}
        for name, info in cache.files.items():
            pinned_file(cache.directory / name, info)
            pinned_file(frozen / name, info)
        require(run_s2.audit.code_snapshot()["sha256"] == report["replay_code"]["sha256"], "Replay source changed during evaluation")
        report.update(status="completed", ok=True, frames_processed=baseline["frames_processed"])
    except Exception as exc:
        report.update(status="failed", ok=False, error=f"{type(exc).__name__}: {exc}")
    report["output_hashes"] = {str(path.relative_to(output)): asset(path) for path in sorted(output.rglob("*"))
                               if path.is_file() and path != output / "report.json"}
    write(output / "report.json", report)
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-report-sha256", required=True)
    parser.add_argument("--boundary-map", type=Path)
    parser.add_argument("--boundary-map-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.source_dir.parent / ("s2-replay-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    if output.exists():
        parser.error("output-dir exists; replay artifacts are never overwritten")
    code = run(args.source_dir, args.source_report_sha256, output, args.boundary_map, args.boundary_map_sha256)
    print(output / "report.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
