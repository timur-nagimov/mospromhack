"""Full-duration, sequential CPU helmet-only S2 baseline on four pinned clips.

Uses production Detector/tracking/attributes/rule/EventManager/Journal. No
downloads, encoding, realtime source, actor GT or application default changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ml.eval import compare_s1_models as audit
from ml.eval import capture_boundary, run_s1

ROOT = run_s1.ROOT
VIDEOS = ("s2_01.mp4", "s2_02.mp4", "clean_01.mp4", "clean_02.mp4")
CLEAN = frozenset({"clean_01.mp4", "clean_02.mp4"})
MANIFEST_SHA = "06642131278f2b484b3c7b859600f1d4e55c95676b8e51e26481da019e848449"
MANIFEST_GIT = "79a8920d49f0de35cc37d7deb4ed92e08c1af927"
MODEL_SHA = {"det": "aa9699693737742033bc899670016413e1c3f9c2725b5fd0b302cfda44ce3a99",
             "buildvision": "55f4dd2f53158eac4ff093d938b9fbda80c99454a03672145bbc36f539988249",
             "vyra": "b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246"}
require, write, asset = audit.require, run_s1._write, run_s1._asset


def load_recipe(path: Path) -> dict:
    recipe = audit.read_json(path)
    require(recipe["schema_version"] == 1 and tuple(recipe["videos"]) == VIDEOS
            and set(recipe["negative_videos"]) == CLEAN and recipe["duration"] == "full_prepared_clip",
            "Recipe must cover all four full clips and both explicitly annotated clean videos")
    require(recipe["manifest_sha256"] == MANIFEST_SHA and recipe["manifest_git"] == MANIFEST_GIT, "Manifest pin changed")
    require((recipe["sample_fps"], recipe["prepared_fps"], recipe["threads"], recipe["max_batch"], recipe["t0"], recipe["tol_s"])
            == (5, 15, 2, 1, 0, 2), "Sampling/thread/time/tolerance policy changed")
    from guard.contracts import RuleConfig
    from guard.pipeline.settings import Settings

    expected_tracker = {"high_thresh": Settings.tracker_high_thresh, "low_thresh": Settings.tracker_low_thresh,
        "new_track_thresh": Settings.tracker_new_track_thresh, "match_iou": Settings.tracker_match_iou,
        "track_buffer_s": Settings.tracker_buffer_s, "min_hits": Settings.tracker_min_hits}
    require(recipe["tracker"] == expected_tracker, "Tracker must match frozen runtime defaults")
    expected_rule = RuleConfig(rule_id="s2", type="ppe_missing", zones=[],
        params={"required": ["helmet"], "min_box_h_px": 60}).model_dump(mode="json")
    require({"rule_id": "s2", **recipe["rule"]} == expected_rule, "Only helmet scope override is permitted; retain runtime rule defaults")
    require(recipe["det"] == {"onnx_path": "yolov8n-640.onnx", "sha256": MODEL_SHA["det"],
            "conf_threshold": .35, "iou_threshold": .5} and recipe["ppe_thresholds"] == {"conf_threshold": .35, "iou_threshold": .7},
            "Detector/threshold policy changed")
    require(set(recipe["candidates"]) == {"buildvision", "vyra"}
            and all(candidate["sha256"] == MODEL_SHA[name] for name, candidate in recipe["candidates"].items()), "PPE model pin changed")
    require({(row["video"], row["index"]) for row in recipe["primary_scope_exclusions"]}
            == {("s2_01.mp4", 2), ("s2_02.mp4", 3)} and len(recipe["primary_scope_exclusions"]) == 2,
            "Helmet-only scope must explicitly exclude exactly the two vest-only uncertain labels")
    return recipe


def label_scopes(clips: list[dict], recipe: dict) -> tuple[list[dict], list[dict], list[dict]]:
    original = [{"label_id": f"{clip['name']}:{index}", "video": clip["name"], **event}
                for clip in clips for index, event in enumerate(clip["events"])]
    require(sum(not row["uncertain"] for row in original) == 6 and sum(row["uncertain"] for row in original) == 2,
            "Original GT must retain six certain and two uncertain labels")
    excluded = []
    for policy in recipe["primary_scope_exclusions"]:
        clip = next(clip for clip in clips if clip["name"] == policy["video"])
        event = clip["events"][policy["index"]]
        require(event["uncertain"] and event["type"] == "ppe_missing"
                and event["note"] == policy["expected_note"] and audit.fingerprint(event) == policy["expected_label_sha256"],
                "Scope exclusion original label hash/note changed")
        excluded.append({**policy, "label_id": f"{policy['video']}:{policy['index']}", "original_label": event})
    excluded_ids = {row["label_id"] for row in excluded}
    primary = [row for row in original if row["label_id"] not in excluded_ids]
    require(len(primary) == 6 and not any(row["uncertain"] for row in primary), "Unexpected primary helmet GT scope")
    return original, primary, excluded


def preflight(data: Path, models: Path, recipe: dict, candidate: str) -> dict:
    """Validate prep/GT/model bytes before importing OpenCV, ORT or Detector."""
    from guard.pipeline.config import load_models
    from guard.inference.base import resolve_model_path

    manifest_path = data / recipe["manifest"]
    require(run_s1._sha(manifest_path) == recipe["manifest_sha256"], "Pinned manifest snapshot hash mismatch")
    manifest = audit.read_json(manifest_path)
    clips = [next(clip for clip in manifest["videos"] if clip["name"] == video) for video in VIDEOS]
    require(all(clip["scenario"] == ("clean" if clip["name"] in CLEAN else "S2")
                and isinstance(clip["events"], list) and (clip["events"] == [] if clip["name"] in CLEAN else bool(clip["events"]))
                for clip in clips), "No implicit clean videos or missing positive labels")
    original, primary, exclusions = label_scopes(clips, recipe)
    assets = {"manifest": asset(manifest_path), "download_report": asset(data / recipe["download_report"]),
              "encode_report": asset(data / recipe["encode_report"])}
    download = audit.read_json(data / recipe["download_report"])
    encoded = audit.read_json(data / recipe["encode_report"])
    require(download["manifest_sha256"] == recipe["manifest_sha256"] and download["manifest_git"] == recipe["manifest_git"], "Download report pin mismatch")
    require(encoded["schema_version"] == 1 and encoded["manifest_sha256"] == recipe["manifest_sha256"]
            and encoded["source_manifest_git"] == recipe["manifest_git"]
            and encoded["source_report_sha256"] == assets["download_report"]["sha256"], "Encode report provenance mismatch")
    require(set(encoded["tool_versions"]) >= {"opencv", "imageio_ffmpeg"}
            and isinstance(encoded["encoder"]["manifest_encode"], dict)
            and bool(encoded["encoder"]["ffmpeg_version"]), "Missing frozen encoder/tool provenance")
    sources = {segment["source"] for clip in clips for segment in clip["segments"]}
    require(len(sources) == 5 and {row["id"] for row in download["sources"]} == sources
            and len(download["sources"]) == 5, "Require exactly five source inputs")
    for row in download["sources"]:
        expected = manifest["sources"][row["id"]]
        path = data / "raw" / f"{row['id']}.mp4"
        require(Path(row["path"]).resolve() == path.resolve(), "Source report path differs from prepared raw source")
        info = assets[f"source:{row['id']}"] = asset(path)
        require(info["sha256"] == row["sha256"] == expected["sha256"]
                and info["bytes"] == row["bytes"] == expected["bytes"], "Raw source bytes/hash differ from frozen manifest")
    require(set(encoded["outputs"]) == set(VIDEOS), "Encode report must cover all four clips")
    prepared = {}
    for clip in clips:
        video = clip["name"]
        item = encoded["outputs"][video]
        require(item["relative_path"] == f"videos/{video}" and isinstance(item["frames"], int)
                and not isinstance(item["frames"], bool) and item["frames"] > 0
                and item["fps"] == recipe["prepared_fps"] and item["width"] == 1280 and item["height"] == 720
                and math.isfinite(item["duration_s"]) and abs(item["duration_s"] - item["frames"] / item["fps"]) < 1e-8,
                "Invalid frozen full-duration frame/FPS/size metadata")
        path = data / item["relative_path"]
        info = assets[video] = asset(path)
        require(info["sha256"] == item["sha256"] and info["bytes"] == item["bytes"], "Encoded clip bytes/hash mismatch")
        labels_path = data / "labels" / f"{Path(video).stem}.json"
        label_file = audit.read_json(labels_path)
        require(label_file.get("video") == video and label_file.get("events") == clip["events"], "Original GT file missing or differs from manifest")
        require(all(0 <= event["t_start"] <= event["t_end"] <= item["duration_s"]
                    and event["type"] == "ppe_missing" for event in clip["events"]), "Labels fall outside full observed clip/type scope")
        assets[f"labels:{video}"] = asset(labels_path)
        prepared[video] = {**item, "path": path}
    model_policy = recipe["candidates"][candidate]
    registry = ROOT / model_policy["registry"]
    assets["model_registry"] = asset(registry)
    specs = load_models(registry, models)
    require(set(specs) == {"det", "ppe"}, "Opt-in registry must contain det and ppe only")
    for name in ("det", "ppe"):
        spec = specs[name]
        policy = recipe["det"] if name == "det" else {**recipe["ppe_thresholds"], **model_policy}
        require(spec.task == "detect" and spec.input_size == (640, 640) and spec.input_dtype == "float32"
                and spec.output_format == "yolov8" and spec.tpu_program is None and spec.onnx_path == policy["onnx_path"]
                and spec.conf_threshold == policy["conf_threshold"] and spec.iou_threshold == policy["iou_threshold"], "Effective model spec differs from frozen recipe")
        onnx_path = resolve_model_path(spec.onnx_path, models)
        info = assets[f"{name}_onnx"] = asset(onnx_path)
        require(info["sha256"] == policy["sha256"], "Actual ONNX hash differs from pinned candidate")
        metadata_path = models / "det.json" if name == "det" else onnx_path.parent / "ppe.json"
        metadata = audit.read_json(metadata_path)
        assets[f"{name}_metadata"] = asset(metadata_path)
        require(metadata["model_id"] == name and metadata["onnx"] == onnx_path.name
                and metadata["onnx_sha256"] == info["sha256"] and metadata["onnx_bytes"] == info["bytes"]
                and metadata["classes"] == spec.classes and metadata["input_size"] == [640, 640]
                and metadata["output_format"] == "yolov8", "Model sidecar/name/classes/hash mismatch")
        if name == "ppe":
            require(spec.classes == model_policy["classes"] and metadata["checker"] == "passed"
                    and metadata["source"] == policy["source"] and metadata["revision"] == policy["revision"], "PPE class mapping/source/revision mismatch")
            require(metadata["input_dtype"] == "float32" and len(metadata["inputs"]) == len(metadata["outputs"]) == 1
                    and metadata["inputs"][0]["shape"] == [1, 3, 640, 640] and metadata["inputs"][0]["dtype"] == 1
                    and metadata["outputs"][0]["shape"] == [1, 18, 8400] and metadata["outputs"][0]["dtype"] == 1,
                    "PPE sidecar must describe static batch1 float32 640/14-class tensors")
        else:
            require(len(spec.classes) == 80 and spec.classes[0] == "person", "COCO det class ordering missing")
    return {"manifest": manifest, "clips": clips, "prepared": prepared, "assets": assets, "specs": specs,
            "original_labels": original, "primary_labels": primary, "scope_exclusions": exclusions,
            "prep_provenance": {"download": download, "encode": encoded}}


def sampled_frames(capture, prepared: dict, stats: dict, deadline: float):
    """Strict full decode with fixed slots; early OR late EOF is an error."""
    plan = run_s1.sample_plan(prepared["fps"], prepared["frames"] / prepared["fps"])
    wanted = {row["source_frame_index"]: row for row in plan}
    stats.update(source_frames_expected=prepared["frames"], source_frames_traversed=0,
                 selected_frames_expected=len(plan), selected_frames_decoded=0, frames_processed=0)
    for index in range(prepared["frames"]):
        run_s1._deadline(deadline)
        require(capture.grab(), f"Partial EOF at source frame {index}, expected {prepared['frames']}")
        stats["source_frames_traversed"] += 1
        if index in wanted:
            ok, frame = capture.retrieve()
            require(ok and frame is not None and frame.size, f"Selected frame decode failed: {index}")
            stats["selected_frames_decoded"] += 1
            yield {**wanted[index], "source_time_s": index / prepared["fps"]}, frame
    require(not capture.grab(), "Video has extra frames beyond frozen preparation report")


def process_frame(camera, frame, row: dict, detectors: dict, tracker, manager, rules, *, ppe_policy=None) -> dict:
    """Same real component ordering as CameraWorker; exceptions never become negatives."""
    from guard.contracts import FrameContext
    from guard.pipeline import compat
    from guard.pipeline.ppe import PpePolicy, PpeFrameError, attribute_coverage, infer_ppe

    started = time.perf_counter()
    per_model = {name: {} for name in detectors}
    ts = row["source_time_s"]
    policy = ppe_policy or PpePolicy()
    detections, ppe_result, model_observations = {}, None, {}
    for name, detector in detectors.items():
        if name != "ppe":
            detections[name] = detector.detect(frame, per_model[name])
            model_observations[name] = "observed"
        elif policy.mode == "full_frame":
            ppe_result = infer_ppe(frame, [], detector, ts, policy)
            if not ppe_result.complete:
                raise PpeFrameError(ppe_result)
            detections[name], per_model[name] = ppe_result.detections, ppe_result.timings
            model_observations[name] = "observed"
    tracking_started = time.perf_counter()
    people = tracker.update([item for item in detections["det"] if item.cls == "person"], ts)
    identities = [{"track_id": track.track_id, "cls": track.cls, "bbox": list(track.bbox), "conf": track.conf,
                   "first_seen": track.first_seen, "last_seen": track.last_seen, "hits": track.hits} for track in people]
    identity = {**row, "coco_people": identities}
    tracking_ms = (time.perf_counter() - tracking_started) * 1000
    if policy.mode == "person_crop" and "ppe" in detectors:
        ppe_result = infer_ppe(frame, people, detectors["ppe"], ts, policy)
        if not ppe_result.complete:
            raise PpeFrameError(ppe_result)
        detections["ppe"], per_model["ppe"] = ppe_result.detections, ppe_result.timings
        model_observations["ppe"] = "observed" if ppe_result.counts["eligible"] else "not_observable"
    attribute_started = time.perf_counter()
    if policy.mode == "person_crop" and ppe_result is not None:
        compat.assign_attributes(people, detections, ts, ppe_by_track=ppe_result.by_track, ppe_unknown=ppe_result.unknown)
    else:
        compat.assign_attributes(people, detections, ts)
    rules_started = time.perf_counter()
    timings = {key: sum(model.get(key, 0) for model in per_model.values()) for key in {key for model in per_model.values() for key in model}}
    context = FrameContext(camera=camera, ts=ts, frame_idx=row["source_frame_index"], frame=frame,
                           detections=detections, people=people, timings=timings, model_observations=model_observations)
    violations = [violation for rule in rules for violation in rule.evaluate(context)]
    alert_started = time.perf_counter()
    new, closed, reopened_events = manager.process_with_updates(camera, ts, violations, timings)
    reopened_ids = {event.id for event in reopened_events}
    closed = [event for event in closed if event.id not in reopened_ids]
    timings.update(tracking=tracking_ms + (rules_started - attribute_started) * 1000, rules=(alert_started - rules_started) * 1000,
                   alert=(time.perf_counter() - alert_started) * 1000, processor=(time.perf_counter() - started) * 1000)
    raw = {name: [{"cls": item.cls, "cls_id": item.cls_id, "conf": item.conf, "bbox": list(item.bbox)} for item in values]
           for name, values in detections.items()}
    trace = {**row, "identity_sha256": audit.fingerprint(identity), "coco_tracks_before_attributes": identities,
        "raw_detections": raw, "person_tracks": [{"track_id": track.track_id, "bbox": list(track.bbox),
            "attributes": track.attributes} for track in people],
        "violations": [{"track_id": item.track_id, "zone_id": item.zone_id, "score": item.score,
                         "bbox": list(item.bbox), "details": item.details} for item in violations],
        "ppe": ppe_result.audit_record() if ppe_result is not None else None,
        "attribute_coverage": attribute_coverage(people, ts),
        "model_observations": dict(context.model_observations), "rule_observations": dict(context.rule_observations),
        "new_event_ids": [event.id for event in new], "per_model_timings_ms": per_model, "timings_ms": timings}
    return {"trace": trace, "new": new, "closed": closed, "reopened": sorted(reopened_ids), "reopened_events": reopened_events}


def add_counts(target: dict, values: dict) -> None:
    """Sum explicit observation denominators, never average per-frame ratios."""
    for key, value in values.items():
        if isinstance(value, dict):
            add_counts(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value


def score_scope(records: list[dict], predictions, durations: dict, recipe: dict) -> dict:
    from ml.eval.events_eval import LabelEvent

    labels = [LabelEvent(**{key: value for key, value in row.items() if key != "label_id"}) for row in records]
    result = run_s1.score_predictions(labels, predictions, durations, recipe["tol_s"], records)
    strict = result["supplemental_strict"]
    strict.pop("independent_clean_video_exposure_s")
    strict.update(positive_video_exposure_s=sum(value for name, value in durations.items() if name not in CLEAN),
                  annotated_helmet_clean_exposure_s=sum(value for name, value in durations.items() if name in CLEAN),
                  alarms_rate_exposure_s=sum(durations.values()))
    return result


def evaluate_scopes(original: list[dict], primary: list[dict], predictions, durations: dict, recipe: dict) -> dict:
    require(set(durations) == set(VIDEOS) and all(value > 0 and math.isfinite(value) for value in durations.values()),
            "Metrics require complete durations for all four clips; no implicit negative exposure")
    result = {}
    for name, records in (("primary_helmet_only", primary), ("secondary_original_legacy", original)):
        per_video = {video: score_scope([row for row in records if row["video"] == video],
            [pred for pred in predictions if pred.video == video], {video: durations[video]}, recipe) for video in VIDEOS}
        clean_metrics = score_scope([], [pred for pred in predictions if pred.video in CLEAN],
                                     {video: durations[video] for video in CLEAN}, recipe)
        result[name] = {"global": score_scope(records, predictions, durations, recipe), "per_video": per_video,
            "positive_videos": score_scope([row for row in records if row["video"] not in CLEAN],
                [pred for pred in predictions if pred.video not in CLEAN], {video: duration for video, duration in durations.items() if video not in CLEAN}, recipe),
            "clean_only": clean_metrics,
            "clean_fp_per_hour": clean_metrics["evaluator"]["overall"]["false_alarms_per_hour"],
            "clean_fp_count": clean_metrics["evaluator"]["overall"]["fp"],
            "clean_exposure_s": sum(durations[video] for video in CLEAN)}
    return result


def run(args) -> int:
    output = args.output_dir
    report = {"schema_version": 1, "scope": "full-duration sequential helmet-only S2 component event baseline",
        "status": "running", "ok": False, "candidate": args.candidate, "versions": run_s1._versions(),
        "assets": {"runner": asset(Path(__file__)), "config": asset(args.config), "evaluator": asset(ROOT / "ml/eval/events_eval.py")},
        "coverage": {"videos_expected": 4, "videos_completed": 0, "frames_processed": 0}, "per_video": {},
        "ppe_input_policy": {"mode": getattr(args, "ppe_input_mode", "full_frame"), "padding_per_side": .1,
            "observed_tracks_only": True, "owner_bound": True, "request_batch": 1,
            "note": "Crop policy only; full_frame retains existing call/assignment behavior"},
        "ppe_input_mode": getattr(args, "ppe_input_mode", "full_frame"),
        "ppe_observations": {}, "attribute_coverage": {},
        "limitations": ["no actor IDs; three s2_02 certain GT intervals share time/type and differ only in notes",
            "event matching does not prove actor/PPE association quality", "training overlap with candidates is unverified",
            "source-time sequential components; no async/realtime/RTSP/media/TPU claim",
            "helmet-only primary scope excludes two explicitly pinned vest-only uncertain labels; legacy scope is secondary",
            "throughput is descriptive; separate offline execution can prioritize measured quality"]}
    started = time.monotonic()
    backend = None
    try:
        frozen = audit.code_snapshot()
        report["provenance"] = {"executable_code": frozen, "code_checks": []}
        recipe = load_recipe(args.config)
        inputs = preflight(args.data_dir, args.models_dir, recipe, args.candidate)
        report["assets"].update(inputs["assets"])
        boundary_path = getattr(args, "boundary_map", None)
        boundary_sha = getattr(args, "boundary_map_sha256", None)
        require(bool(boundary_path) == bool(boundary_sha), "Boundary map and SHA256 must be supplied together")
        boundaries = {}
        if boundary_path is not None:
            encoded = inputs["prep_provenance"]["encode"]
            samples = {video: [{**row, "source_time_s": row["source_frame_index"] / prepared["fps"]}
                               for row in run_s1.sample_plan(prepared["fps"], prepared["frames"] / prepared["fps"], recipe["sample_fps"])]
                       for video, prepared in inputs["prepared"].items()}
            boundaries = capture_boundary.load_boundaries(boundary_path, boundary_sha, assets=inputs["assets"], encoded=encoded,
                segments={clip["name"]: clip["segments"] for clip in inputs["clips"]}, samples=samples)
            report["assets"]["boundary_map"] = asset(boundary_path)
            require(report["assets"]["boundary_map"]["sha256"] == boundary_sha, "Boundary-map asset drift after validation")
            report["capture_boundary_protocol"] = {"mode": "pinned_encoder_part_boundary", "map_sha256": boundary_sha,
                "boundaries": boundaries, "state_reset": "durable last-violation closure/readback before fresh Tracker/EventManager/rules",
                "note": "Separate controlled protocol; original continuous primary is unchanged"}
        report["effective_recipe"] = recipe
        report["prep_provenance"] = inputs["prep_provenance"]
        report["effective_models"] = {name: spec.model_dump(mode="json") for name, spec in inputs["specs"].items()}
        report["git"] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "tracked_changes": subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, text=True).splitlines()}
        report["scope_exclusions"] = inputs["scope_exclusions"]
        write(output / "original_labels.json", inputs["original_labels"])
        write(output / "primary_labels.json", inputs["primary_labels"])
        write(output / "scope_exclusions.json", inputs["scope_exclusions"])
        write(output / "effective_config.json", recipe)
        originals = output / "original_gt"
        originals.mkdir()
        for video in VIDEOS:
            source = args.data_dir / "labels" / f"{Path(video).stem}.json"
            (originals / source.name).write_bytes(source.read_bytes())
        write(output / "manifest.snapshot.json", inputs["manifest"])
        mapping = {f"eval_{Path(video).stem}": {"video": video, "t0": 0, "capture_id": f"s2-quality:{video}",
                   "source_segments": next(clip["segments"] for clip in inputs["clips"] if clip["name"] == video)} for video in VIDEOS}
        write(output / "mapping.json", mapping)
        write(output / "report.json", report)

        import cv2
        import numpy as np
        import onnxruntime as ort
        from guard.contracts import CameraConfig
        from guard.pipeline import compat
        from guard.pipeline.detector import Detector as ProductionDetector
        from guard.pipeline.ppe import PpePolicy, PpeFrameError
        from guard.inference.onnx_cpu import OnnxCpuBackend
        from guard.rules.base import build_rules, load_plugins
        from ml.eval.events_eval import predictions_from_events

        require(not any(compat.FALLBACKS.values()), "Evaluation refuses fallback components")
        ort.disable_telemetry_events()
        cv2.setNumThreads(2)

        def open_capture(prepared):
            capture = cv2.VideoCapture(str(prepared["path"]))
            try:
                require(capture.isOpened() and abs(capture.get(cv2.CAP_PROP_FPS) - prepared["fps"]) < 1e-8
                    and capture.get(cv2.CAP_PROP_FRAME_COUNT) == prepared["frames"]
                    and capture.get(cv2.CAP_PROP_FRAME_WIDTH) == prepared["width"]
                    and capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == prepared["height"], "Video decoder metadata differs from frozen preparation")
            except BaseException:
                capture.release()
                raise
            return capture

        for prepared in inputs["prepared"].values():
            run_s1._deadline(args._deadline)
            open_capture(prepared).release()  # Metadata for ALL four clips before ORT construction.

        class AuditedBackend(OnnxCpuBackend):
            def infer_timed(self, model_id, batch):
                outputs, timings = super().infer_timed(model_id, batch)
                require(all(np.isfinite(value).all() for value in outputs), "Non-finite raw ONNX output")
                self.last_outputs[model_id] = [{"shape": list(value.shape), "dtype": str(value.dtype),
                    "bytes": value.nbytes, "sha256": hashlib.sha256(value.tobytes()).hexdigest(), "finite": True} for value in outputs]
                return outputs, timings

        backend = AuditedBackend(models_dir=args.models_dir, intra_op_threads=2, max_batch=1)
        backend.last_outputs = {}
        detectors = {}
        for name, spec in inputs["specs"].items():
            backend.load(spec)
            detectors[name] = ProductionDetector(spec, backend)
        require(set(backend.loaded_models()) == {"det", "ppe"}, "Required real models were not loaded")
        load_plugins()
        predictions, durations, identity_hashes = [], {}, {}
        write(output / "predictions.json", [])
        with compat.Journal(output / "guard.db") as journal:
            for clip in inputs["clips"]:
                video = clip["name"]
                audit.check_code_snapshot(frozen, report, f"before_{video}")
                camera = CameraConfig.model_validate({"camera_id": f"eval_{Path(video).stem}", "name": clip["title"], "site": "S2 evaluation",
                    "source": {"type": "file", "uri": str(inputs["prepared"][video]["path"]), "loop": False, "realtime": False},
                    "sample_fps": 5, "zones": [], "rules": [{"rule_id": "s2", **recipe["rule"]}]})
                write(output / f"camera_{Path(video).stem}.json", camera.model_dump(mode="json"))
                def new_state():
                    rules = build_rules(camera)
                    require(len(rules) == 1 and rules[0].type == "ppe_missing"
                            and rules[0].required_models == frozenset({"det", "ppe"}), "Active real S2 rule missing")
                    return compat.Tracker(classes={"person"}, **recipe["tracker"]), compat.EventManager(), rules
                tracker, manager, rules = new_state()
                boundary, did_reset = boundaries.get(video), False
                resets = []
                stats = {"created_events": 0, "close_notifications": 0, "reopen_transitions": 0, "frames_with_violations": 0}
                report["per_video"][video] = {"coverage": stats, "ppe_observations": {}, "attribute_coverage": {}}
                if boundary:
                    report["per_video"][video]["boundary_resets"] = resets
                    stats["boundary_closures"] = 0
                capture = open_capture(inputs["prepared"][video])
                identities = hashlib.sha256()
                video_started = time.monotonic()
                try:
                    with (output / f"samples_{Path(video).stem}.jsonl").open("w") as trace, (output / f"lifecycle_{Path(video).stem}.jsonl").open("w") as lifecycle:
                        for row, frame in sampled_frames(capture, inputs["prepared"][video], stats, args._deadline):
                            if boundary and not did_reset and row["source_frame_index"] >= boundary["cut_before_source_frame"]:
                                closed = capture_boundary.close_capture(journal, manager, camera.camera_id)
                                resets.append({"before_sample_index": row["sample_index"],
                                    "before_source_frame_index": row["source_frame_index"], "boundary": boundary,
                                    "closed": [event.model_dump(mode="json") for event in closed],
                                    "closure_policy": "production last violation; reset Tracker + EventManager + rules"})
                                stats["boundary_closures"] += len(closed)
                                lifecycle.write(json.dumps({"source_time_s": row["source_time_s"], "kind": "capture_boundary",
                                    "created": [], "closed": [{"id": event.id, "ended_at": event.ended_at} for event in closed],
                                    "reopened": []}) + "\n")
                                tracker, manager, rules = new_state()
                                did_reset = True
                            source_capture_id = f"s2-quality:{video}"
                            effective_capture_id = capture_boundary.effective_capture_id(source_capture_id, boundary, int(did_reset))
                            origin = {"capture_id": effective_capture_id,
                                "frame_seq": row["source_frame_index"], "source_pos_s": row["source_time_s"]}
                            trace_origin = {"capture_id": effective_capture_id}
                            if boundary_path is not None:
                                trace_origin["source_frame"] = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                                    "bytes": frame.nbytes, "sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                                    "layout": "decoded BGR, C-order bytes"}
                            if boundary:
                                origin.update(source_capture_id=source_capture_id, segment_index=int(did_reset))
                                trace_origin.update(source_capture_id=source_capture_id, segment_index=int(did_reset),
                                                    boundary_evidence_sha256=boundary["evidence_sha256"])
                            try:
                                result = process_frame(camera, frame, row, detectors, tracker, manager, rules,
                                    ppe_policy=PpePolicy(getattr(args, "ppe_input_mode", "full_frame"), audit=True))
                            except PpeFrameError as exc:
                                trace.write(json.dumps({**row, **trace_origin, "ppe": exc.audit, "status": "failed"}, allow_nan=False) + "\n")
                                report["ppe_failure"] = {"video": video, **row, "audit": exc.audit}
                                raise
                            for event in result["new"]:
                                event.details = {**event.details, "origin": origin}
                                journal.add(event)
                                predictions.append(event.model_copy(deep=True))
                            run_s1.persist_lifecycle_updates(journal, result["closed"], result["reopened_events"])
                            if result["new"] or result["closed"] or result["reopened"]:
                                lifecycle.write(json.dumps({"source_time_s": row["source_time_s"], "created": [event.id for event in result["new"]],
                                    "closed": [{"id": event.id, "ended_at": event.ended_at} for event in result["closed"]], "reopened": result["reopened"]}) + "\n")
                            trace.write(json.dumps({**result["trace"], **trace_origin,
                                "decoder_pos_s": capture.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                "raw_outputs": {name: values for name, values in backend.last_outputs.items()
                                    if name != "ppe" or getattr(args, "ppe_input_mode", "full_frame") == "full_frame"}}, allow_nan=False) + "\n")
                            for target in (report, report["per_video"][video]):
                                add_counts(target["ppe_observations"], result["trace"]["ppe"]["counts"])
                                add_counts(target["attribute_coverage"], result["trace"]["attribute_coverage"])
                            identities.update((result["trace"]["identity_sha256"] + "\n").encode())
                            stats["frames_processed"] += 1
                            report["coverage"]["frames_processed"] += 1
                            stats["created_events"] += len(result["new"])
                            stats["close_notifications"] += len(result["closed"])
                            stats["reopen_transitions"] += len(result["reopened"])
                            stats["frames_with_violations"] += bool(result["trace"]["violations"])
                            if stats["frames_processed"] % 25 == 0 or result["new"]:
                                trace.flush()
                                write(output / "predictions.json", [event.model_dump(mode="json") for event in predictions])
                                write(output / "report.json", report)
                    require(not boundary or did_reset, "Configured source boundary never reached")
                    require(stats["frames_processed"] == stats["selected_frames_decoded"] == stats["selected_frames_expected"], "Incomplete processed sample coverage")
                finally:
                    capture.release()
                    stats["wall_s"] = time.monotonic() - video_started
                    write(output / "report.json", report)
                durations[video] = stats["source_frames_traversed"] / inputs["prepared"][video]["fps"]
                stats.update(duration_s=durations[video], source_fps=inputs["prepared"][video]["fps"])
                identity_hashes[video] = identities.hexdigest()
                report["per_video"][video]["coco_track_identity_sha256"] = identities.hexdigest()
                report["per_video"][video]["right_censored_open_events"] = [event.model_dump(mode="json") for event in manager.open_events(camera.camera_id)]
                report["coverage"]["videos_completed"] += 1
                audit.check_code_snapshot(frozen, report, f"after_{video}")
                write(output / "durations.json", durations)
            rows, total = journal.query(limit=len(predictions) + 1)
            require(total == len(predictions) == len(rows), "Event/Journal complete readback mismatch")
            write(output / "journal_events.json", [event.model_dump(mode="json") for event in rows])
        require(report["coverage"]["videos_completed"] == 4, "All four videos must complete")
        write(output / "predictions.json", [event.model_dump(mode="json") for event in predictions])
        report["inference_stats"] = backend.stats()
        for name in ("det", "ppe"):
            expected = report["ppe_observations"]["calls"] if name == "ppe" else report["coverage"]["frames_processed"]
            stats = report["inference_stats"]["models"].get(name, {})
            require(stats.get("errors", 0) == 0 and stats.get("calls", 0) == stats.get("items", 0) == expected
                    and (not expected or stats.get("max_batch_seen") == 1), f"{name} inference errors or incomplete call coverage")
        require(report["inference_stats"]["providers"] == ["CPUExecutionProvider"] and report["inference_stats"]["intra_op_threads"] == 2,
                "CPU provider/thread policy mismatch")
        report["evaluation"] = evaluate_scopes(inputs["original_labels"], inputs["primary_labels"], predictions_from_events(predictions, mapping), durations, recipe)
        report["coco_track_identity_sha256"] = audit.fingerprint(identity_hashes)
        report["identity_fingerprint_definition"] = "source sample coordinates and COCO-only pre-attribute tracks; excludes PPE outputs and attributes, not actor GT"
        report["coverage"]["selected_frames_expected"] = sum(row["coverage"]["selected_frames_expected"] for row in report["per_video"].values())
        for info in report["assets"].values():
            require(run_s1._sha(Path(info["path"])) == info["sha256"], "Input asset changed during evaluation")
        audit.check_code_snapshot(frozen, report, "final")
        report["output_hashes"] = {str(path.relative_to(output)): asset(path) for path in output.rglob("*") if path.is_file()
                                   and path.name not in {"report.json", "runtime.log"}}
        report.update(ok=True, status="completed")
    except Exception as exc:
        report.pop("evaluation", None)
        report.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if backend is not None:
            report.setdefault("inference_stats", backend.stats())
            backend.close()
        report["wall_s"] = time.monotonic() - started
        write(output / "report.json", report)
    return 0 if report["ok"] else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate", choices=("buildvision", "vyra"), required=True)
    result.add_argument("--ppe-input-mode", choices=("full_frame", "person_crop"), default="full_frame")
    result.add_argument("--boundary-map", type=Path, help="Pinned independent encoder-part boundary evidence; opt-in protocol")
    result.add_argument("--boundary-map-sha256", help="Required SHA256 of --boundary-map")
    result.add_argument("--config", type=Path, default=ROOT / "configs/eval-s2.json")
    result.add_argument("--data-dir", type=Path, default=Path(os.environ.get("GUARD_DATA_DIR", ROOT / "data")) / "s2_public_20261024")
    result.add_argument("--models-dir", type=Path, default=Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models")))
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--timeout", type=audit.bounded_timeout, default=900.0)
    result.add_argument("--json", action="store_true")
    result.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if bool(args.boundary_map) != bool(args.boundary_map_sha256):
        arguments.error("--boundary-map and --boundary-map-sha256 must be supplied together")
    if args._run:
        return run(args)
    started = time.monotonic()
    output = (args.output_dir or args.data_dir / "evaluations" /
        (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + args.candidate + "-" + uuid4().hex[:8])).expanduser().resolve()
    if output.exists():
        arguments.error("output-dir exists; S2 evaluations are never overwritten")
    output.mkdir(parents=True)
    command = [sys.executable, "-m", "ml.eval.run_s2", "--_run", "--_deadline", str(started + args.timeout - 2),
        "--candidate", args.candidate, "--ppe-input-mode", args.ppe_input_mode,
        "--config", str(args.config.expanduser().resolve()), "--data-dir", str(args.data_dir.expanduser().resolve()),
        "--models-dir", str(args.models_dir.expanduser().resolve()), "--output-dir", str(output), "--timeout", str(args.timeout)]
    if args.boundary_map is not None:
        command.extend(["--boundary-map", str(args.boundary_map.expanduser().resolve()),
                        "--boundary-map-sha256", args.boundary_map_sha256])
    with (output / "runtime.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True,
                                   env={**os.environ, **audit.THREAD_ENV})
        timed_out = False
        try:
            process.wait(timeout=max(.01, args.timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            cleanup = audit.terminate_owned_group(process)
    report_path = output / "report.json"
    report = audit.read_json(report_path) if report_path.is_file() else {}
    if timed_out or process.returncode != 0 or not report.get("ok") or not cleanup["leader_reaped"]:
        report.pop("evaluation", None)
        report.update(ok=False, status="timeout" if timed_out else "failed")
        report.setdefault("error", "Shared wall timeout" if timed_out else "See runtime.log")
    report.update(output_dir=str(output), total_wall_s=time.monotonic() - started, runtime_returncode=process.returncode,
                  process_group_cleanup=cleanup)
    report.setdefault("output_hashes", {})["runtime.log"] = asset(output / "runtime.log")
    write(report_path, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False) if args.json else f"S2 evaluation: {report['status']}; report={report_path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
