"""Frozen 141-image PPE test evaluation through the production CPU Detector.

Run ``python -m ml.eval.run_ppe_onnx --help``. The candidate remains experimental;
the public test split has unverified overlap with the author's training corpus.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from ml.eval_ppe import evaluate_predictions, image_key, load_gt, normalize
from tools.prepare_ppe_candidate import (
    DATA_NAMES, IMAGE_SUFFIXES, PROFILES,
    asset, get_profile, inspect_onnx, require, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_TEST_FINGERPRINT = "800a03725b2c6fc37b8c80794735a03666fd424cc63b4eeecb54b548e37b6791"
DETECTOR_GRIDS = {(640, 640): 8400, (1280, 1280): 33600}
RECIPE = {"split": "test", "expected_images": 141, "input_size": [640, 640],
          "conf_ap": 0.001, "nms_iou": 0.7, "conf_op": 0.35, "threads": 2,
          "ap": "COCO 101-point interpolated AP50", "secondary_iou": {"no_helmet": 0.3},
          "decoder": "production Detector: single-label argmax, class-aware NMS, no max_det cap",
          "quality_gate": "experimental_only", "training_overlap": "unverified",
          "data_sha256": DATA_TEST_FINGERPRINT}


def effective_recipe(spec, *, candidate: bool = False) -> dict:
    """Derive a fresh resolution recipe; PPE pins remain strictly 640."""
    require(spec.task == "detect" and spec.input_dtype == "float32",
            "Frozen recipe needs raw detection float32")
    require(not candidate or spec.output_format == "yolov8", "Pinned PPE profiles require raw YOLOv8")
    size = tuple(spec.input_size)
    recipe = copy.deepcopy(RECIPE)
    if spec.output_format == "dfine":
        require(size == (640, 640), "Pinned D-FINE profile requires 640x640")
        require(len(spec.classes) == 80, "D-FINE needs explicit 80-class registry")
        recipe.update(output_format="dfine", queries=300, top_k=300, nms_applied=False,
                      nms_iou_note="reference threshold retained; D-FINE uses no NMS",
                      preprocessing="RGB PIL BILINEAR direct resize; float32 /255 NCHW; no normalize/pad",
                      decoder="production Detector: focal sigmoid, top-Q over query/class pairs, no NMS",
                      score_comparison=">", working_point_score_comparison=">=", input_size_source="registry ModelSpec",
                      resolution_experiment=False, architecture_experiment=True,
                      overrides_from_yolov8={"preprocessing": "direct resize", "decoder": "D-FINE focal top-Q, no NMS"})
        return recipe
    require(spec.output_format == "yolov8", "Frozen recipe needs raw YOLOv8 or explicit D-FINE")
    require(size in DETECTOR_GRIDS, "Generic detector input size must be 640x640 or 1280x1280")
    require(not candidate or size == (640, 640), "Pinned PPE profiles require 640x640")
    recipe.update(input_size=list(size), grid_count=DETECTOR_GRIDS[size], strides=[8, 16, 32],
                  input_size_source="registry ModelSpec",
                  resolution_experiment=size != (640, 640),
                  overrides_from_640={} if size == (640, 640) else {
                      "input_size": {"reference": [640, 640], "effective": list(size)},
                      "grid_count": {"reference": 8400, "effective": DETECTOR_GRIDS[size]}})
    return recipe


def validate_raw_outputs(outputs, class_count: int, recipe: dict) -> None:
    if recipe.get("output_format") == "dfine":
        from guard.vision.detr import validate_dfine_outputs
        validate_dfine_outputs(outputs, class_count, batch_size=1, queries=recipe["queries"])
        return
    expected = (1, 4 + class_count, recipe["grid_count"])
    require(len(outputs) == 1 and outputs[0].shape == expected,
            f"Per-image ONNX shape drift: expected {expected}")
    require(np.isfinite(outputs[0]).all(), "Non-finite per-image model output")


def dataset_inputs(data: Path) -> tuple[list[Path], dict, list[dict], str]:
    """Fail before inference for missing labels, split/count drift or bad images."""
    import cv2
    import yaml

    meta = yaml.safe_load((data / "data.yaml").read_text(encoding="utf-8"))
    names = meta["names"] if isinstance(meta["names"], dict) else dict(enumerate(meta["names"]))
    require(names == dict(enumerate(DATA_NAMES)), "Expected official Construction-PPE class order")
    directory = data / "images" / "test"
    images = sorted(p for p in directory.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    require(len(images) == RECIPE["expected_images"], "Frozen recipe requires exactly 141 test images")
    records = [{"relative_path": "data.yaml", **asset(data / "data.yaml")}]
    for image in images:
        label = data / "labels" / "test" / (image.stem + ".txt")
        require(label.is_file(), f"Missing label: {label}")
        frame = cv2.imread(str(image))
        require(frame is not None and frame.ndim == 3, f"Unreadable image: {image}")
        for line in label.read_text().splitlines():
            fields = line.split()
            require(len(fields) == 5, f"Malformed YOLO label: {label}")
            require(fields[0].isdigit() and int(fields[0]) in names, f"Unknown class: {label}")
            coords = np.array([float(value) for value in fields[1:]])
            require(np.isfinite(coords).all() and (coords >= 0).all() and (coords <= 1).all()
                    and (coords[2:] > 0).all(), f"Invalid box: {label}")
        records.extend({"relative_path": str(path.relative_to(data)), **asset(path)} for path in (image, label))
    loaded_images, gt = load_gt(data, ["test"], names)
    require(images == loaded_images, "Ground-truth loader image scope differs")
    identity = [{"relative_path": row["relative_path"], "bytes": row["bytes"], "sha256": row["sha256"]}
                for row in records]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(digest == DATA_TEST_FINGERPRINT, "Dataset bytes differ from the pinned official test snapshot")
    return images, gt, records, digest


def semantic_smoke(detections, frame: np.ndarray, frame_index: int) -> dict:
    """Real frame predictions → one production association observation → S2 rule.

    Fresh tracks per still image avoid inventing temporal continuity. This tests
    canonical mapping and unknown preservation. A single negative observation
    cannot satisfy production's temporal PPE guard; no history is fabricated.
    This does not measure tracking quality or duration-confirmed event metrics.
    """
    from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Track
    from guard.rules.plugins.ppe_missing import PpeMissingRule
    from guard.tracking.attributes import assign_attributes

    working = [d for d in detections if d.conf >= RECIPE["conf_op"]]
    ts = float(frame_index)
    people = [Track(track_id=index + 1, cls="person", bbox=det.bbox, conf=det.conf,
                    first_seen=ts, last_seen=ts)
              for index, det in enumerate(d for d in working if d.cls == "person")]
    assign_attributes(people, {"ppe": working}, ts)
    camera = CameraConfig(camera_id="ppe-public-test", name="PPE public test", site="offline",
                          source=SourceConfig(type="images", uri="public-test", realtime=False))
    rule = PpeMissingRule(RuleConfig(rule_id="s2-semantic-smoke", type="ppe_missing",
                                    params={"required": ["helmet", "vest"], "min_box_h_px": 60}), camera)
    violations = rule.evaluate(FrameContext(camera=camera, ts=ts, frame_idx=frame_index, frame=frame, people=people))
    require(all(person.attributes["respirator"] is None for person in people), "Unsupported respirator became known")
    return {"observation_scope": "single_image_fresh_tracks", "temporal_event_evaluation": False,
            "canonical_ppe_classes": sorted({d.cls for d in working if d.cls in {"helmet", "no_helmet", "vest", "no_vest"}}),
            "people": [{"track_id": p.track_id, "bbox": list(p.bbox), "conf": p.conf,
                         "attributes": {key: p.attributes[key] for key in ("helmet", "vest", "respirator")}}
                        for p in people],
            "violations": [{"track_id": v.track_id, "bbox": list(v.bbox), "score": v.score,
                             "details": v.details} for v in violations],
            "evidence": "one production association observation per fresh track; canonical mapping/unknown only; no temporal event or quality claim"}


def infer_images(detector, images: list[Path], data: Path, output: Path, *, include_smoke: bool = True) -> tuple[dict, list[dict], list[dict]]:
    import cv2

    predictions = defaultdict(list)
    records, errors = [], []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, image in enumerate(images):
            key = image_key(data, image)
            row = {"image_key": key, "index": index, "status": "ok"}
            try:
                frame = cv2.imread(str(image))
                require(frame is not None, f"Unreadable image: {image}")
                timings = {}
                started = time.perf_counter()
                detections = detector.detect(frame, timings=timings)
                row["total_ms"] = (time.perf_counter() - started) * 1000
                row["timings_ms"] = timings
                row["detections"] = [{"cls": d.cls, "cls_id": d.cls_id, "conf": d.conf, "bbox": list(d.bbox)}
                                     for d in detections]
                if include_smoke:
                    row["semantic_smoke"] = semantic_smoke(detections, frame, index)
                for det in detections:
                    canonical = normalize(det.cls)
                    if canonical is not None:
                        predictions[canonical].append((key, det.conf, np.asarray(det.bbox)))
            except Exception as exc:  # Record every failed frame; never score failure as empty predictions.
                row = {"image_key": key, "index": index, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
                errors.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            records.append(row)
    return predictions, records, errors


def inspect_detector_onnx(path: Path, classes: list[str], input_size=(640, 640)) -> dict:
    """Check raw static FLOAT YOLOv8 detect heads at the explicit registry size."""
    import onnx

    size = tuple(input_size)
    require(size in DETECTOR_GRIDS, "Unsupported detector input size")
    require(classes, "Detector class names are required")
    model = onnx.load(str(path), load_external_data=False)
    require(not any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL
                    for t in model.graph.initializer), "External model data is unsupported")
    require(not model.functions and all(node.domain in ("", "ai.onnx") for node in model.graph.node),
            "Custom ONNX operators/functions are unsupported")
    require(not any(node.op_type == "NonMaxSuppression" for node in model.graph.node), "Embedded NMS is unsupported")
    onnx.checker.check_model(model, full_check=True)
    shapes = [[d.dim_value for d in value.type.tensor_type.shape.dim]
              for value in [*model.graph.input, *model.graph.output]]
    require(len(model.graph.input) == len(model.graph.output) == 1
            and shapes == [[1, 3, *size], [1, 4 + len(classes), DETECTOR_GRIDS[size]]],
            f"Baseline raw YOLOv8 interface mismatch for {list(size)} / {DETECTOR_GRIDS[size]} anchors")
    require(all(value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
                for value in [*model.graph.input, *model.graph.output]), "Detector input/output must be float32")
    metadata = {item.key: item.value for item in model.metadata_props}
    checks = {}
    for key, expected in (("imgsz", list(size)), ("names", dict(enumerate(classes)))):
        if key in metadata:
            require(ast.literal_eval(metadata[key]) == expected, f"Detector {key} metadata mismatch")
            checks[key] = "matched"
        else:
            checks[key] = "absent; registry and static graph used"
    if "task" in metadata:
        require(metadata["task"] == "detect", "Detector task metadata mismatch")
    return {"checker": "passed", "input_shape": shapes[0], "output_shape": shapes[1],
            "input_dtype": "float32", "output_dtype": "float32", "grid_count": DETECTOR_GRIDS[size],
            "classes": classes, "metadata": metadata, "metadata_checks": checks}


def run(data: Path, models: Path, config: Path, output_root: Path, model_id: str = "ppe",
        candidate_profile: str = "buildvision") -> tuple[int, Path]:
    from guard.inference.onnx_cpu import OnnxCpuBackend
    from guard.pipeline.detector import Detector
    from ml.registry import load_specs

    data, models, config = data.resolve(), models.resolve(), config.resolve()
    output = output_root.resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "preparing", "recipe": copy.deepcopy(RECIPE), "output": str(output),
              "model_id": model_id, "candidate_profile": candidate_profile if model_id == "ppe" else None,
              "model_status": "experimental", "training_overlap_with_test": "unverified"}
    write_json(output / "report.json", report)
    backend = None
    try:
        specs = load_specs(config)
        require(model_id in specs, f"Model role absent from registry: {model_id}")
        spec = specs[model_id]
        report["registry_input_size"] = list(spec.input_size)
        recipe = effective_recipe(spec, candidate=model_id == "ppe")
        report["recipe"] = recipe
        path = models / spec.onnx_path
        candidate = model_id == "ppe"
        dfine = spec.output_format == "dfine"
        profile = get_profile(candidate_profile) if candidate else None
        sidecar_path = path.parent / f"{model_id}.json"
        if candidate:
            require("det" in specs and spec.classes == profile.classes, "Candidate needs logical det/ppe and canonical classes")
            inspection = inspect_onnx(path, path.parent / profile.source_config if profile.source_config else None,
                                      profile.name)
            sidecar = json.loads(sidecar_path.read_text())
            require(sidecar["classes"] == spec.classes and sidecar["revision"] == profile.revision
                    and sidecar["sha256"] == profile.files[profile.model_name][1]
                    and sidecar.get("candidate_profile", "buildvision") == profile.name,
                    "Canonical sidecar/registry drift")
        elif dfine:
            from ml.eval.inspect_dfine import inspect_dfine_onnx
            inspection = inspect_dfine_onnx(path, spec.classes, spec.input_size)
        images, gt, manifest, data_sha = dataset_inputs(data)
        write_json(output / "inputs.json", manifest)
        report["provenance"] = {"model": asset(path), "config": asset(config),
                                "sidecar": asset(sidecar_path) if sidecar_path.exists() else None,
                                "revision": profile.revision if candidate else inspection["revision"] if dfine else None,
                                "candidate_profile": profile.name if candidate else None,
                                "source_manifest": asset(path.parent / "source.json") if (candidate or dfine) and (path.parent / "source.json").exists() else None,
                                "license_provenance": sidecar.get("license_provenance") if candidate else None,
                                "data_sha256": data_sha, "images": len(images),
                                "runner": asset(Path(__file__)), "metrics": asset(ROOT / "ml" / "eval_ppe.py"),
                                "detector": asset(ROOT / "guard" / "pipeline" / "detector.py"),
                                "preprocess": asset(ROOT / "guard" / "vision" / "preprocess.py"),
                                "decoder": asset(ROOT / "guard" / "vision" / ("detr.py" if dfine else "yolo.py")),
                                "versions": {name: importlib.metadata.version(name)
                                             for name in ("numpy", "onnx", "onnxruntime", "pydantic", "Pillow")}}
        # No registry/runtime mutation: low AP threshold is local to this frozen evaluation copy.
        eval_spec = spec.model_copy(update={"conf_threshold": recipe["conf_ap"], "iou_threshold": recipe["nms_iou"]})

        class StrictCpuBackend(OnnxCpuBackend):
            def infer_timed(self, model_id, batch):
                outputs, timings = super().infer_timed(model_id, batch)
                validate_raw_outputs(outputs, len(eval_spec.classes), recipe)
                return outputs, timings

        backend = StrictCpuBackend(models_dir=models, intra_op_threads=recipe["threads"])
        backend.load(eval_spec)
        # Resolve empty registry classes exactly as production PipelineEngine.get_detector.
        if not eval_spec.classes:
            names = backend.class_names(model_id)
            require(names, "No model class names in registry or ONNX metadata")
            eval_spec = eval_spec.model_copy(update={"classes": names})
        report["provenance"]["class_names_source"] = "registry" if spec.classes else "onnx_metadata_via_backend"
        if not candidate and not dfine:
            inspection = inspect_detector_onnx(path, eval_spec.classes, eval_spec.input_size)
        report["provenance"]["inspection"] = inspection
        detector = Detector(eval_spec, backend)
        started = time.perf_counter()
        predictions, records, errors = infer_images(detector, images, data, output, include_smoke=candidate)
        report.update(status="failed" if errors else "complete", images_attempted=len(records),
                      images_succeeded=len(records) - len(errors), errors=errors, backend=backend.stats(),
                      wall_seconds=time.perf_counter() - started)
        if errors:
            report["metrics"] = None
            write_json(output / "report.json", report)
            return 1, output
        supported = {normalize(name) for name in eval_spec.classes} - {None}
        report["metrics"] = evaluate_predictions(predictions, gt, supported, recipe["conf_op"])
        report["class_map"] = {name: normalize(name) for name in eval_spec.classes}
        times = np.array([r["total_ms"] for r in records])
        report["latency_ms"] = {"mean": float(times.mean()), "p50": float(np.percentile(times, 50)),
                                "p95": float(np.percentile(times, 95)), "includes": "Detector preprocess+infer+postprocess"}
        smoke = [{"image_key": r["image_key"], **r["semantic_smoke"]} for r in records] if candidate else []
        write_json(output / "semantic-smoke.json", {
            "status": "complete" if candidate else "not_applicable", "frames": len(smoke), "conf": recipe["conf_op"],
            "people": sum(len(row["people"]) for row in smoke),
            "instantaneous_rule_hits": sum(len(row["violations"]) for row in smoke),
            "unsupported_attributes": ["respirator"], "frames_with_unknown_helmet": sum(
                any(p["attributes"]["helmet"] is None for p in row["people"]) for row in smoke),
            "quality_evidence": False, "temporal_event_evaluation": False, "per_image": smoke})
        write_json(output / "report.json", report)
        return 0, output
    except Exception as exc:
        report.update(status="failed", metrics=None, error=f"{type(exc).__name__}: {exc}")
        write_json(output / "report.json", report)
        raise
    finally:
        if backend is not None:
            backend.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="prepared test-only directory containing data.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--models-config", "--config", dest="config", type=Path,
                        help="defaults to the selected candidate profile's opt-in registry")
    parser.add_argument("--candidate-profile", choices=tuple(PROFILES), default="buildvision")
    parser.add_argument("--model-id", default="ppe", help="logical registry role; use det for the COCO baseline")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "ppe_evaluation")
    args = parser.parse_args(argv)
    config = args.config or ROOT / "configs" / get_profile(args.candidate_profile).registry
    code, output = run(args.data, args.models_dir, config, args.output_root, args.model_id, args.candidate_profile)
    print(output / "report.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
