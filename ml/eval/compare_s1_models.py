"""Bounded, sequential YOLOv8n versus s/x comparison on fixed S1 segments.

Two fresh run_s1 evaluations share inputs, sampling, rules and tracker settings.
Only model assets vary. This is an exploratory positive-video comparison, not
an independent quality benchmark or automatic model promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ml.eval import run_s1

ROOT = run_s1.ROOT
CANDIDATE_VARIANTS = ("s", "x")
SAMPLE_KEYS = ("sample_index", "target_s", "source_frame_index", "source_time_s", "decoder_pos_s", "capture_id")
COMMON_ASSETS = ("runner", "evaluator", "manifest", *run_s1.VIDEOS)
COMMON_OUTPUTS = ("manifest.json", "labels_with_ids.json", "mapping.json", "durations.json",
                  "camera_s1_01.json", "camera_s1_02.json", "labels/s1_01.json", "labels/s1_02.json")
THREAD_ENV = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2"}
METADATA_SEMANTICS = ("model_id", "task", "input_size", "input", "output_format", "classes", "source_classes",
                      "export_device", "export_threads", "torch_threads", "torch_interop_threads", "simplified_by",
                      "versions", "opset", "inputs", "outputs", "ort_output_shapes", "problems")


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def code_snapshot(root: Path = ROOT) -> dict:
    """Hash executable source bytes, including untracked Python modules."""
    files = {path.relative_to(root).as_posix(): run_s1._sha(path)
             for directory in ("guard", "ml") for path in sorted((root / directory).rglob("*.py")) if path.is_file()}
    require(files, "No executable guard/ml source files found")
    return {"definition": "guard/**/*.py + ml/**/*.py; relative paths and SHA256, including untracked files",
            "file_count": len(files), "files": files, "sha256": fingerprint(files)}


def check_code_snapshot(frozen: dict, report: dict, stage: str) -> None:
    current = code_snapshot()
    changed = sorted(path for path in frozen["files"].keys() | current["files"].keys()
                     if frozen["files"].get(path) != current["files"].get(path))
    report["provenance"]["code_checks"].append({"stage": stage, "sha256": current["sha256"],
                                                "matches_frozen": not changed, "changed_paths": changed})
    require(not changed, f"Executable source changed at {stage}: {', '.join(changed)}")


def normalized_model(model: dict) -> dict:
    return {key: value for key, value in model.items() if key != "onnx_path"}


def validate_model(registry: Path, models_dir: Path, variant: str) -> dict:
    """Check local bytes/sidecar/source claims without ONNX, torch or ORT imports."""
    require(variant in ("n", *CANDIDATE_VARIANTS), "Unsupported model variant; expected n/s/x")
    from guard.pipeline.config import load_models

    specs = load_models(registry, models_dir)
    require("det" in specs, "Registry must provide logical model_id det")
    spec = specs["det"].model_dump(mode="json")
    require(spec["task"] == "detect" and spec["input_size"] == [640, 640]
            and spec["input_dtype"] == "float32" and spec["output_format"] == "yolov8"
            and spec["conf_threshold"] == .25 and spec["iou_threshold"] == .5 and spec["tpu_program"] is None,
            "Comparison requires CPU YOLOv8 detect 640, float32, conf=.25, IoU=.5")
    expected = f"yolov8{variant}-640.onnx"
    require(spec["onnx_path"] == expected, f"Expected registry ONNX filename {expected}")
    metadata_path = models_dir / "det.json"
    metadata = read_json(metadata_path)
    require(metadata.get("model_id") == "det" and metadata.get("task") == "detect"
            and metadata.get("onnx") == expected and metadata.get("input_size") == [640, 640]
            and metadata.get("output_format") == "yolov8", "Metadata role, ONNX filename or shape mismatch")
    require(metadata.get("classes") == spec["classes"] and len(spec["classes"]) == 80
            and spec["classes"][0] == "person" and metadata.get("source_classes") == spec["classes"],
            "Expected unchanged 80-class COCO ordering, including person")
    require(metadata.get("inputs") == {"images": [1, 3, 640, 640]}
            and metadata.get("outputs") == {"output0": [1, 84, 8400]}
            and metadata.get("ort_output_shapes") == [[1, 84, 8400]], "Expected verified static CPU export tensor shapes")
    onnx = run_s1._asset(models_dir / expected)
    require(onnx["sha256"] == metadata.get("onnx_sha256") and onnx["bytes"] == metadata.get("onnx_bytes"),
            "Metadata ONNX hash/size does not match actual bytes")
    source = metadata.get("source", "")
    require(isinstance(source, str) and re.fullmatch(
        rf"https://github\.com/ultralytics/assets/releases/download/v\d+\.\d+\.\d+/yolov8{variant}\.pt", source),
        "Expected official Ultralytics assets pretrained source URL")
    weights_path = Path(metadata.get("weights", ""))
    if not weights_path.is_absolute():
        weights_path = models_dir / weights_path
    require(weights_path.name == f"yolov8{variant}.pt", "Unexpected source weights filename")
    weights = run_s1._asset(weights_path)
    require(weights["sha256"] == metadata.get("weights_sha256"), "Source weights hash differs from metadata")
    require(metadata.get("export_device") == "cpu" and not metadata.get("problems"), "Unverified/failed CPU export")
    return {"registry": run_s1._asset(registry), "onnx": onnx, "metadata": run_s1._asset(metadata_path),
            "weights": weights, "effective_model": spec, "metadata_content": metadata,
            "metadata_semantics": {key: metadata.get(key) for key in METADATA_SEMANTICS},
            "source_claim": source}


def distribution(values: list[float]) -> dict:
    require(values and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) and value >= 0 for value in values), "Invalid/missing timing samples")
    ordered = sorted(values)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        return ordered[lower] + (ordered[math.ceil(index)] - ordered[lower]) * (index - lower)

    return {"n": len(values), "sum_ms": sum(values), "mean_ms": sum(values) / len(values),
            "p50_ms": percentile(.5), "p95_ms": percentile(.95), "max_ms": max(values)}


def load_samples(directory: Path, video: str, coverage: dict) -> list[dict]:
    rows = [json.loads(line) for line in (directory / f"samples_{Path(video).stem}.jsonl").read_text().splitlines()]
    require(len(rows) == coverage["frames_processed"] == coverage["selected_frames_decoded"] == 300,
            f"Incomplete raw sample coverage: {video}")
    require(coverage["source_frames_traversed"] == coverage["source_frames_expected"], f"Incomplete decode: {video}")
    plan = run_s1.sample_plan(coverage["source_fps"])
    for row, expected in zip(rows, plan):
        require(all(row[key] == value for key, value in expected.items()), f"Wrong source frame/sample grid: {video}")
        require(abs(row["source_time_s"] - row["source_frame_index"] / coverage["source_fps"]) < 1e-9,
                f"Wrong source timestamp: {video}")
        require(row["capture_id"] == f"s1-quality:{video}" and math.isfinite(row["decoder_pos_s"]), "Invalid capture mapping")
        require(row["detections"] == len(row["raw_detections"]), "Detection count disagrees with raw trace")
        for key in ("preprocess", "infer", "postprocess"):
            distribution([row["timings_ms"][key]])
    return rows


def timing_summary(rows: list[dict], coverage: dict) -> dict:
    keys = sorted({key for row in rows for key in row["timings_ms"]})
    stages = {key: distribution([row["timings_ms"][key] for row in rows if key in row["timings_ms"]]) for key in keys}
    require(stages["infer"]["n"] == len(rows), "Inference timings do not cover every selected frame")
    return {"stages_ms": stages, "sample_denominator": len(rows),
            "instrumented_stage_sum_ms": distribution([sum(row["timings_ms"].values()) for row in rows]),
            "video_loop_wall_s": coverage["wall_s"], "amortized_video_loop_ms_per_frame": coverage["wall_s"] * 1000 / len(rows),
            "definition": "full raw samples, no rolling window; video-loop wall includes decode, tracking, rules, journal and trace IO",
            "unmeasured_individual_stages": [key for key in ("decode", "tracking", "rules", "alert", "total") if key not in stages]}


def metrics(section: dict) -> dict:
    return {key: section[key] for key in ("evaluator", "supplemental_strict", "label_counts", "excluded_uncertain_labels",
                                         "prediction_classifications", "uncertain_exclusions")}


def sample_difference(a: dict, b: dict) -> dict:
    classes_a, classes_b = Counter(row["cls"] for row in a["raw_detections"]), Counter(row["cls"] for row in b["raw_detections"])
    return {"sample_index": a["sample_index"], "source_frame_index": a["source_frame_index"], "source_time_s": a["source_time_s"],
            "detections": {"baseline": a["detections"], "candidate": b["detections"], "delta": b["detections"] - a["detections"],
                           "classes_baseline": dict(classes_a), "classes_candidate": dict(classes_b),
                           "raw_equal": a["raw_detections"] == b["raw_detections"]},
            "person_tracks": {"baseline": a["person_tracks"], "candidate": b["person_tracks"],
                              "count_delta": len(b["person_tracks"]) - len(a["person_tracks"])},
            "violations": {"baseline": a["violations"], "candidate": b["violations"]},
            "created_event_ids": {"baseline": a["new_event_ids"], "candidate": b["new_event_ids"]}}


def compare_runs(directories: dict[str, Path], model_assets: dict[str, dict], output: Path, expected_recipe: dict) -> dict:
    """Audit paired coverage and copy unchanged evaluator metrics; no new matcher."""
    reports = {name: read_json(path / "report.json") for name, path in directories.items()}
    a, b = reports["baseline"], reports["candidate"]
    for name, report in reports.items():
        require(report.get("ok") and report.get("status") == "completed", f"{name} run did not complete")
        require(report["coverage"] == {"videos_expected": 2, "videos_completed": 2,
                "selected_frames_expected": 600, "frames_processed": 600}, f"{name} incomplete 600-frame coverage")
        require(report["label_counts"] == {"certain": 4, "uncertain": 3}, f"{name} label scope changed")
        inference = report["inference_stats"]
        require(inference["models"]["det"]["errors"] == 0 and inference["models"]["det"]["calls"] == 600
                and inference["models"]["det"]["items"] == 600 and inference["models"]["det"]["max_batch_seen"] == 1
                and inference["backend"] == "onnx-cpu" and inference["providers"] == ["CPUExecutionProvider"]
                and inference["intra_op_threads"] == 2, f"{name} inference errors/count/provider/thread mismatch")
        for key, model_key in (("det_onnx", "onnx"), ("det_metadata", "metadata"), ("model_registry", "registry")):
            require(report["assets"][key]["sha256"] == model_assets[name][model_key]["sha256"], f"{name} model asset changed: {key}")
        require(normalized_model(read_json(directories[name] / "effective_model.json")) ==
                normalized_model(model_assets[name]["effective_model"]), f"{name} actual model settings changed")
        for relative, asset in report["output_hashes"].items():
            require(run_s1._sha(directories[name] / relative) == asset["sha256"], f"{name} artifact hash changed: {relative}")
    config_a = {key: value for key, value in a["effective_recipe"].items() if key != "model_registry"}
    config_b = {key: value for key, value in b["effective_recipe"].items() if key != "model_registry"}
    require(config_a == config_b, "Effective config differs beyond model_registry")
    require(config_a == {key: value for key, value in expected_recipe.items() if key != "model_registry"},
            "Effective config differs from the fixed preflight recipe")
    require(a["git"] == b["git"] and a["versions"] == b["versions"], "Code/environment changed between runs")
    for key in COMMON_ASSETS:
        require(a["assets"][key]["sha256"] == b["assets"][key]["sha256"], f"Paired input differs: {key}")
    paired_hashes = {}
    for relative in COMMON_OUTPUTS:
        hashes = {name: run_s1._sha(path / relative) for name, path in directories.items()}
        require(len(set(hashes.values())) == 1, f"Paired config/GT/mapping differs: {relative}")
        paired_hashes[relative] = hashes["baseline"]
    result = {"paired_config_sha256": fingerprint(config_a), "paired_artifact_hashes": paired_hashes,
              "per_video": {}, "runs": {}, "attribution_limit": "Run-local track IDs are not actor GT or cross-model identities. Raw differences are observations, not causal correctness labels."}
    all_rows = {"baseline": [], "candidate": []}
    labels = read_json(directories["baseline"] / "labels_with_ids.json")
    for video in run_s1.VIDEOS:
        rows = {name: load_samples(path, video, reports[name]["per_video"][video]["coverage"]) for name, path in directories.items()}
        signatures = {name: fingerprint([{key: row[key] for key in SAMPLE_KEYS} for row in values]) for name, values in rows.items()}
        require(len(set(signatures.values())) == 1, f"Paired sample indices/timestamps differ: {video}")
        differences = [sample_difference(left, right) for left, right in zip(rows["baseline"], rows["candidate"])]
        diff_path = output / f"paired_{Path(video).stem}.jsonl"
        diff_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in differences))
        boundaries = [bound for row in labels if row["video"] == video for bound in (row["t_start"], row["t_end"])]
        near = [row for row in differences if any(abs(row["source_time_s"] - bound) <= 2 for bound in boundaries)
                and (not row["detections"]["raw_equal"] or row["violations"]["baseline"] != row["violations"]["candidate"])]
        # Fixed source-time landmarks are chosen independently of either model's outputs.
        landmarks = (0, 50, 100, 150, 200, 250, 299)
        result["per_video"][video] = {"paired_samples": 300, "sample_fingerprints": signatures,
            "raw_difference_artifact": run_s1._asset(diff_path),
            "frames_detection_count_changed": sum(row["detections"]["delta"] != 0 for row in differences),
            "frames_raw_detections_changed": sum(not row["detections"]["raw_equal"] for row in differences),
            "frames_track_count_changed": sum(row["person_tracks"]["count_delta"] != 0 for row in differences),
            "frames_violation_presence_changed": sum(bool(row["violations"]["baseline"]) != bool(row["violations"]["candidate"]) for row in differences),
            "boundary_examples": near[:12], "boundary_example_definition": "first 12 changed source frames within 2s of any annotated boundary; descriptive, not new matching/actor labels",
            "manual_review_landmarks": [{key: rows["baseline"][index][key] for key in SAMPLE_KEYS} for index in landmarks]}
        for name in rows:
            all_rows[name].extend(rows[name])
    for name, report in reports.items():
        coverage = {"wall_s": sum(row["coverage"]["wall_s"] for row in report["per_video"].values())}
        result["runs"][name] = {"report": run_s1._asset(directories[name] / "report.json"),
            "global": metrics(report["global"]), "timing": timing_summary(all_rows[name], coverage),
            "per_video": {video: {"metrics": metrics(report["per_video"][video]),
                "coverage": report["per_video"][video]["coverage"],
                "timing": timing_summary([row for row in all_rows[name] if row["capture_id"] == f"s1-quality:{video}"],
                                          report["per_video"][video]["coverage"])} for video in run_s1.VIDEOS},
            "versions": report["versions"], "inference_stats": report["inference_stats"], "wall_s": report["wall_s"]}
    return result


def run_child(config: Path, models_dir: Path, data_dir: Path, output: Path, timeout: float, deadline: float) -> dict:
    output.mkdir()
    # Internal entry point intentionally avoids a second detached process group:
    # the wrapper's outer watchdog owns this complete subtree. On return (also
    # after a per-run timeout), outer cleanup kills remaining group members.
    command = [sys.executable, "-m", "ml.eval.run_s1", "--_run", "--config", str(config), "--models-dir", str(models_dir),
               "--data-dir", str(data_dir), "--output-dir", str(output), "--_deadline", str(min(time.monotonic() + timeout, deadline) - 1)]
    with (output / "runtime.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                 env={**os.environ, **THREAD_ENV})
        timed_out = False
        try:
            child.wait(timeout=max(.01, min(timeout, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.terminate()
            try:
                child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=1)
    return {"command": command, "returncode": child.returncode, "timed_out": timed_out,
            "runtime_log": run_s1._asset(output / "runtime.log")}


def terminate_owned_group(process: subprocess.Popen, grace_s: float = 1.0) -> dict:
    """Clean a group created by main(start_new_session=True), even if its leader exited.

    Leader wait/reaping is not a process-group liveness test. Always send KILL
    after the TERM grace period: a descendant may ignore TERM after its parent
    has already exited. ProcessLookupError means the owned group is gone.
    """
    cleanup = {"term_sent": False, "kill_sent": False, "leader_reaped": False}
    grace_deadline = time.monotonic() + grace_s
    try:
        os.killpg(process.pid, signal.SIGTERM)
        cleanup["term_sent"] = True
    except ProcessLookupError:
        pass
    if cleanup["term_sent"]:
        try:
            process.wait(timeout=max(.001, grace_s))
        except subprocess.TimeoutExpired:
            pass
        remaining = grace_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        try:
            os.killpg(process.pid, signal.SIGKILL)
            cleanup["kill_sent"] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
        cleanup["leader_reaped"] = True
    except subprocess.TimeoutExpired:
        pass
    return cleanup


def execute(args) -> int:
    output = args.output_dir
    report = {"schema_version": 1, "scope": "exploratory sequential S1 model comparison", "ok": False, "status": "running",
              "candidate_variant": args.candidate_variant,
              "assets": {"wrapper": run_s1._asset(Path(__file__))}, "children": {},
              "limits": {"global_timeout_s": args.timeout, "per_run_timeout_s": args.run_timeout, "threads": 2},
              "limitations": ["4 certain + 3 uncertain intervals on two positive videos; independent clean exposure is zero",
                  "no production promotion, threshold tuning or default model switch", "no actor-level person GT",
                  "sequential component evaluation does not prove async or multicamera realtime performance",
                  "official source URL is recorded provenance; local hashes verify consistency, not publisher attestation",
                  "timing order is fixed baseline then candidate; no warmup removal or randomized performance trial"]}
    started = time.monotonic()
    try:
        frozen_code = code_snapshot()
        report["provenance"] = {"executable_code": frozen_code, "code_checks": []}
        recipe, _, _ = run_s1.load_recipe(ROOT / "configs/eval-s1.json")
        registries = {"baseline": ROOT / recipe["model_registry"],
                      "candidate": ROOT / f"configs/models.yolov8{args.candidate_variant}-640.yaml"}
        model_dirs = {"baseline": args.baseline_models_dir, "candidate": args.candidate_models_dir}
        assets = {name: validate_model(registries[name], path, "n" if name == "baseline" else args.candidate_variant)
                  for name, path in model_dirs.items()}
        require(normalized_model(assets["baseline"]["effective_model"]) == normalized_model(assets["candidate"]["effective_model"]),
                "Model semantics differ beyond ONNX path/weights")
        require(assets["baseline"]["metadata_semantics"] == assets["candidate"]["metadata_semantics"],
                "Export metadata semantics/versions differ; only model weights may vary")
        require(assets["baseline"]["onnx"]["sha256"] != assets["candidate"]["onnx"]["sha256"]
                and assets["baseline"]["weights"]["sha256"] != assets["candidate"]["weights"]["sha256"],
                "Baseline and candidate must be distinct model artifacts")
        require(assets["baseline"]["source_claim"].replace("yolov8n.pt", "MODEL.pt") ==
                assets["candidate"]["source_claim"].replace(f"yolov8{args.candidate_variant}.pt", "MODEL.pt"), "Source release differs")
        report["models"] = assets
        report["assets"]["base_config"] = run_s1._asset(ROOT / "configs/eval-s1.json")
        directories = {name: output / name for name in model_dirs}
        for name in directories:
            require(time.monotonic() < args._deadline - 2, "Global deadline exhausted before both fresh runs")
            config = {**recipe, "model_registry": str(registries[name].resolve())}
            config_path = output / f"config_{name}.json"
            run_s1._write(config_path, config)
            check_code_snapshot(frozen_code, report, f"before_{name}")
            run_s1._write(output / "report.json", report)
            report["children"][name] = run_child(config_path, model_dirs[name], args.data_dir, directories[name], args.run_timeout, args._deadline)
            check_code_snapshot(frozen_code, report, f"after_{name}")
            run_s1._write(output / "report.json", report)
            require(not report["children"][name]["timed_out"] and report["children"][name]["returncode"] == 0,
                    f"{name} evaluation failed or timed out; inspect its report/runtime.log")
        report["comparison"] = compare_runs(directories, assets, output, recipe)
        for model in assets.values():
            for key in ("registry", "onnx", "metadata", "weights"):
                require(run_s1._sha(Path(model[key]["path"])) == model[key]["sha256"], f"Model asset changed: {key}")
        require(run_s1._sha(Path(__file__)) == report["assets"]["wrapper"]["sha256"], "Wrapper changed during run")
        require(run_s1._sha(ROOT / "configs/eval-s1.json") == report["assets"]["base_config"]["sha256"], "Base config changed during run")
        check_code_snapshot(frozen_code, report, "final")
        report["comparison"]["executable_code_sha256"] = frozen_code["sha256"]
        report.update(ok=True, status="completed")
    except Exception as exc:
        report.update(ok=False, status="failed", error=f"{type(exc).__name__}: {exc}")
    report["worker_wall_s"] = time.monotonic() - started
    run_s1._write(output / "report.json", report)
    return 0 if report["ok"] else 1


def bounded_timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be numeric") from exc
    if not math.isfinite(result) or not 3 <= result <= 1800:
        raise argparse.ArgumentTypeError("timeout must be in [3, 1800] seconds")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=Path(os.environ.get("GUARD_DATA_DIR", ROOT / "data")))
    result.add_argument("--baseline-models-dir", type=Path, default=Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models")))
    result.add_argument("--candidate-variant", choices=CANDIDATE_VARIANTS, default="s")
    result.add_argument("--candidate-models-dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--run-timeout", type=bounded_timeout, default=180.0)
    result.add_argument("--timeout", type=bounded_timeout, default=420.0)
    result.add_argument("--json", action="store_true")
    result.add_argument("--_run", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--_deadline", type=float, help=argparse.SUPPRESS)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args._run:
        return execute(args)
    started = time.monotonic()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.baseline_models_dir = args.baseline_models_dir.expanduser().resolve()
    candidate = (args.candidate_models_dir or args.baseline_models_dir /
                 f"candidates/yolov8{args.candidate_variant}-640").expanduser().resolve()
    output = (args.output_dir or args.data_dir / "model_comparison" /
              (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])).expanduser().resolve()
    if output.exists():
        arguments.error("output-dir exists; comparison artifacts are never overwritten")
    output.mkdir(parents=True)
    command = [sys.executable, "-m", "ml.eval.compare_s1_models", "--_run", "--_deadline", str(started + args.timeout - 2),
               "--output-dir", str(output), "--data-dir", str(args.data_dir), "--baseline-models-dir", str(args.baseline_models_dir),
               "--candidate-models-dir", str(candidate), "--candidate-variant", args.candidate_variant,
               "--timeout", str(args.timeout), "--run-timeout", str(args.run_timeout)]
    with (output / "wrapper.log").open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, start_new_session=True,
                                   env={**os.environ, **THREAD_ENV})
        timed_out = False
        try:
            process.wait(timeout=max(.01, args.timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # This also cleans descendants left after a per-run timeout or an
            # already-exited worker; never use leader returncode as group state.
            cleanup = terminate_owned_group(process)
    path = output / "report.json"
    report = read_json(path) if path.is_file() else {"scope": "exploratory sequential S1 model comparison"}
    if timed_out or process.returncode != 0 or not report.get("ok") or not cleanup["leader_reaped"]:
        report.update(ok=False, status="timeout" if timed_out else "failed")
        report.setdefault("error", "Global wall timeout" if timed_out else "See wrapper.log")
    report.update(output_dir=str(output), total_wall_s=time.monotonic() - started, worker_returncode=process.returncode,
                  process_group_cleanup=cleanup)
    run_s1._write(path, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False) if args.json else f"S1 model comparison: {report['status']}; report={path}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
