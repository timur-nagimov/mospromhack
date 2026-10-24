"""Bounded paired RF-DETR native/ONNX quality evaluation, with failed parity retained."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from tools import prepare_rfdetr_candidate as source
from tools.diagnose_rfdetr_candidate import array_info, load_primary, pinned_json

GRAPH_SHA = "cd7a94b8c524905349a9639e78c72b66c04a223bbaf369c5d18811cc754aefa4"
GRAPH_BYTES = 460466535
POSTPROCESS_FILES = {
    "models/postprocess.py": "86dc3aaa85283da17c36b0263473e8e2db6225eed90bd91920a6ffeb27c04e0e",
    "utilities/box_ops.py": "b71ae71cd71889f9207374edc4807532494c6f76d31aae973388407933cd67c0",
}
ARMS = ("native", "onnx")
RECIPE = {"images": 141, "person_gt": 236, "split": "test", "input_shape": [1, 3, 880, 880],
          "threads": 2, "conf_ap": .001, "conf_op": .35, "iou": .5, "top_k": 300,
          "ap": "existing ml.eval_ppe.evaluate_predictions COCO101 AP50", "nms": False,
          "postprocess": "pinned RF PostProcess: sigmoid all91, stable descending score/ascending flat ID, top300, cxcywh->xyxy, scale+clip",
          "score_floor_comparison": ">=", "operating_comparison": ">=", "training_overlap": "unverified",
          "primary_numeric_parity": "failed", "promotion": False, "target_tpu": "not_compiled_or_verified"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_dataset(images, gt, records, data):
    import numpy as np
    from ml.eval_ppe import image_key
    keys = [image_key(data, image) for image in images]
    source.require(len(images) == RECIPE["images"] and len(set(keys)) == len(keys)
                   and len({image.resolve() for image in images}) == len(images)
                   and len({image.stem for image in images}) == len(images), "Dataset needs exactly141 unique image identities")
    image_records = [row for row in records if row["relative_path"].startswith("images/")]
    source.require(len(image_records) == len(images) and len({row["sha256"] for row in image_records}) == len(images),
                   "Duplicate or incomplete image content manifest")
    for rows in gt.values():
        source.require(set(rows).issubset(keys), "GT references an image outside the frozen scope")
        for boxes in rows.values():
            source.require(boxes.ndim == 2 and boxes.shape[1] == 4 and np.isfinite(boxes).all()
                           and (boxes[:, 2:] > boxes[:, :2]).all()
                           and len(np.unique(boxes, axis=0)) == len(boxes), "Invalid or duplicate GT boxes")
    source.require(sum(len(boxes) for boxes in gt.get("person", {}).values()) == RECIPE["person_gt"], "Expected exactly236 person GT")
    serial = {c: {key: boxes.tolist() for key, boxes in sorted(rows.items())} for c, rows in sorted(gt.items())}
    return {"image_keys": keys, "image_keys_sha256": fingerprint(keys), "ground_truth": serial,
            "ground_truth_sha256": fingerprint(serial), "person_gt": RECIPE["person_gt"]}


def read_rgb(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        source.require(image.getexif().get(274, 1) in (None, 1), "Nontrivial EXIF orientation would change GT/source coordinates")
        rgb = np.asarray(image.convert("RGB")).copy()
    source.require(rgb.ndim == 3 and rgb.shape[2] == 3 and min(rgb.shape[:2]) > 0, "Invalid RGB source image")
    return rgb


def validate_postprocessed(value, height, width):
    import numpy as np
    boxes, scores, labels, probabilities = (value[name] for name in ("boxes", "scores", "labels", "all_scores"))
    source.require(boxes.shape == (300, 4) and boxes.dtype == np.float32 and np.isfinite(boxes).all(), "Invalid decoded boxes")
    source.require(scores.shape == (300,) and scores.dtype == np.float32 and np.isfinite(scores).all()
                   and (scores >= 0).all() and (scores <= 1).all(), "Invalid decoded scores")
    source.require(labels.shape == (300,) and labels.dtype == np.int64 and (labels >= 0).all() and (labels < 91).all(), "Invalid sparse labels")
    source.require(probabilities.shape == (1, 300, 91) and probabilities.dtype == np.float32
                   and np.isfinite(probabilities).all() and (probabilities >= 0).all() and (probabilities <= 1).all(), "Invalid all91 sigmoid scores")
    source.require((boxes[:, :2] >= 0).all() and (boxes[:, 2:] >= boxes[:, :2]).all()
                   and (boxes[:, [0, 2]] <= width).all() and (boxes[:, [1, 3]] <= height).all(), "Decoded boxes outside source coordinates")


def infer_pairs(images, data, output, layout, calls, postprocess, *, preprocess=source.preprocess_rgb, reader=read_rgb):
    """One shared tensor per image. Any arm error invalidates aggregate metrics."""
    import numpy as np
    from ml.eval_ppe import image_key, normalize
    predictions = {arm: defaultdict(list) for arm in ARMS}
    records, errors = [], []
    with (output / "images.jsonl").open("w") as stream:
        for index, image in enumerate(images):
            key, arrays = image_key(data, image), {}
            row = {"index": index, "image_key": key, "primary_numeric_parity": "failed", "arms": {}, "status": "failed"}
            try:
                started = time.perf_counter()
                rgb = reader(image)
                height, width = rgb.shape[:2]
                tensor = preprocess(rgb)  # exactly once, before either runtime
                shared = tensor.numpy()
                source.require(shared.shape == (1, 3, 880, 880) and shared.dtype == np.float32
                               and np.isfinite(shared).all() and shared.flags.c_contiguous, "Invalid shared preprocessing tensor")
                tensor_info = array_info(shared)
                row.update(image=source.asset(image), source_shape=list(rgb.shape), input=tensor_info,
                           shared_preprocess_ms=(time.perf_counter() - started) * 1000)
                for arm in ARMS:
                    arm_row = {"status": "failed", "primary_numeric_parity": "failed"}
                    try:
                        source.require(array_info(shared) == tensor_info, "Shared tensor changed before runtime")
                        begin = time.perf_counter()
                        arm_row["inference_attempted"] = True
                        raw = calls[arm](tensor if arm == "native" else shared)
                        arm_row["inference_ms"] = (time.perf_counter() - begin) * 1000
                        source.require(array_info(shared) == tensor_info, "Runtime mutated the shared tensor")
                        # Preserve numeric malformed outputs as evidence, without object/pickle arrays.
                        for name, value in zip(("dets", "logits"), raw):
                            if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                                arrays[arm + "_raw_" + name] = value
                        source.validate_outputs(raw)
                        arm_row["raw_outputs"] = {name: array_info(value) for name, value in zip(("dets", "logits"), raw)}
                        begin = time.perf_counter()
                        decoded = postprocess(raw, (height, width))
                        validate_postprocessed(decoded, height, width)
                        arm_row["postprocess_ms"] = (time.perf_counter() - begin) * 1000
                        arrays.update({arm + "_" + name: value for name, value in decoded.items()})
                        all_predictions = []
                        for box, score, label in zip(decoded["boxes"], decoded["scores"], decoded["labels"]):
                            native_id, conf = int(label), float(score)
                            name = layout["logit_index_to_name"][native_id]
                            canonical = normalize(name) if name is not None else None
                            all_predictions.append({"native_class_id": native_id, "native_class_name": name,
                                                    "canonical_class": canonical, "confidence": conf, "box_xyxy": box.tolist()})
                            if canonical is not None and conf >= RECIPE["conf_ap"]:
                                predictions[arm][canonical].append((key, conf, box.copy()))
                        arm_row.update(status="complete", predictions=all_predictions,
                                       ap_person_predictions=sum(p["canonical_class"] == "person" and p["confidence"] >= RECIPE["conf_ap"] for p in all_predictions))
                    except Exception as exc:
                        arm_row["error"] = f"{type(exc).__name__}: {exc}"
                        errors.append({"image_key": key, "arm": arm, "error": arm_row["error"]})
                    row["arms"][arm] = arm_row
                source.require(array_info(shared) == tensor_info, "Shared tensor changed after paired evaluation")
                row["status"] = "complete" if all(value["status"] == "complete" for value in row["arms"].values()) else "failed"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.append({"image_key": key, "arm": "shared_input", "error": row["error"]})
                for arm in ARMS:
                    row["arms"].setdefault(arm, {"status": "not_run", "primary_numeric_parity": "failed", "error": row["error"]})
            if arrays:
                cache = output / f"sample-{index:03d}.npz"
                np.savez_compressed(cache, **arrays)
                row["tensor_file"] = source.asset(cache)
                row["tensors"] = {name: array_info(value) for name, value in arrays.items()}
            records.append(row)
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
    source.require(len(records) == len(images) and len({row["image_key"] for row in records}) == len(images), "Incomplete paired frame accounting")
    return predictions, records, errors


def make_runtimes(weights, graph):
    import torch
    import onnxruntime as ort
    from rfdetr.models.postprocess import PostProcess
    model, provenance = source.build_native(weights)
    source.require(not model.training and not model._export and provenance["config"]["num_select"] == 300,
                   "Quality native arm must be normal eval, never source export mode")
    postprocessor = PostProcess(num_select=300).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
    source.require(session.get_providers() == ["CPUExecutionProvider"], "Original ONNX must use CPU only")
    def native(tensor):
        source.require(not model.training and not model._export, "Native arm unexpectedly entered export/training mode")
        with torch.inference_mode():
            return [value.detach().cpu().numpy().copy() for value in source.raw_pair(model(tensor))]
    def onnx(batch):
        return session.run(["dets", "labels"], {"images": batch})
    def postprocess(raw, size):
        boxes, logits = (torch.from_numpy(value) for value in raw)
        with torch.inference_mode():
            values = postprocessor({"pred_boxes": boxes, "pred_logits": logits}, torch.tensor([size], dtype=torch.int64))
            source.require(len(values) == 1, "Postprocessor batch mismatch")
            result = {key: values[0][key].detach().cpu().numpy().copy() for key in ("boxes", "scores", "labels")}
            result["all_scores"] = logits.sigmoid().numpy().copy()
        return result
    return {"native": native, "onnx": onnx}, postprocess, provenance


def code_assets():
    root = Path(importlib.metadata.distribution("rfdetr").locate_file("rfdetr"))
    for relative, expected in POSTPROCESS_FILES.items():
        source.require(source.sha(root / relative) == expected, f"Pinned postprocess source drift: {relative}")
    paths = [Path(__file__), Path(source.__file__), root / "models/lwdetr.py", root / "models/transformer.py",
             root / "models/postprocess.py", root / "utilities/box_ops.py",
             root / "assets/coco_classes.py", source.ROOT / "tools/diagnose_rfdetr_candidate.py",
             source.ROOT / "tools/prepare_ppe_candidate.py",
             source.ROOT / "ml/eval_ppe.py", source.ROOT / "ml/eval/run_ppe_onnx.py"]
    paths.extend(sorted((root / "models/backbone").rglob("*.py")))
    return paths


def score_complete(predictions, gt, layout, records, errors):
    from ml.eval_ppe import evaluate_predictions, normalize
    if (errors or len(records) != RECIPE["images"] or len({row["image_key"] for row in records}) != len(records)
            or any(row["status"] != "complete" or any(row["arms"].get(arm, {}).get("status") != "complete" for arm in ARMS)
                   for row in records)):
        return {arm: None for arm in ARMS}
    supported = {normalize(name) for name in layout["logit_index_to_name"] if name is not None} - {None}
    source.require(supported == {"person"}, "Unexpected PPE support in sparse COCO class layout")
    return {arm: evaluate_predictions(predictions[arm], gt, supported, RECIPE["conf_op"]) for arm in ARMS}


def archive_sources(output):
    archived = []
    for relative in ("tools/evaluate_rfdetr_candidate.py", "tools/prepare_rfdetr_candidate.py",
                     "tools/diagnose_rfdetr_candidate.py", "tools/prepare_ppe_candidate.py",
                     "ml/eval/run_ppe_onnx.py", "ml/eval_ppe.py"):
        path, copy = source.ROOT / relative, output / "code-sources" / relative
        initial = source.asset(path)
        source.require(not copy.exists(), "Archived source must be new")
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
        copied = source.asset(copy)
        source.require(source.asset(path) == initial and copied["sha256"] == initial["sha256"]
                       and copied["bytes"] == initial["bytes"], "Source changed while archiving")
        archived.append({"source": initial, "archive": copied})
    return archived


def evaluate(args):
    import numpy as np
    from ml.eval.run_ppe_onnx import dataset_inputs, DATA_TEST_FINGERPRINT
    _, _, initial_assets = load_primary(args.primary_parity_dir)
    export, export_asset = pinned_json(args.export_dir / "manifest.json")
    graph = args.export_dir / "model.onnx"
    graph_asset = source.asset(graph)
    source.require(export.get("ok") is True and export.get("source_revision") == source.SOURCE_REVISION
                   and graph_asset == export["onnx"] and graph_asset["sha256"] == GRAPH_SHA
                   and graph_asset["bytes"] == GRAPH_BYTES, "Expected original pinned static ONNX, never debug/re-export")
    initial_assets.update({item["path"]: item for item in (export_asset, graph_asset)})
    source.inspect_graph(graph)
    images, gt, data_records, data_sha = dataset_inputs(args.data)
    source.require(data_sha == DATA_TEST_FINGERPRINT, "Frozen dataset fingerprint drift")
    identity = validate_dataset(images, gt, data_records, args.data)
    paths = [*initial_assets, args.weights, *code_assets()]
    before = {str(Path(path).resolve()): source.asset(path) for path in paths}
    before.update({row["path"]: {key: row[key] for key in ("path", "bytes", "sha256")} for row in data_records})
    source.require(all(before[path] == value for path, value in initial_assets.items()), "Pinned asset changed during bootstrap")
    layout = source.class_layout()
    source.require(layout["logit_count"] == 91 and layout["logit_index_to_name"][1] == "person"
                   and layout["logit_index_to_name"][77] == "cell phone" and layout["background_class_id"] is None,
                   "Sparse class contract drift")
    source.write(args.output_dir / "inputs.json", {"primary_numeric_parity": "failed", "data_sha256": data_sha,
                                                 **identity, "files": data_records})
    calls, postprocess, native = make_runtimes(args.weights, graph)
    started = time.perf_counter()
    predictions, records, errors = infer_pairs(images, args.data, args.output_dir, layout, calls, postprocess)
    wall = time.perf_counter() - started
    source.require(all(source.asset(path) == item for path, item in before.items()), "Input/code asset changed during quality evaluation")
    metrics = score_complete(predictions, gt, layout, records, errors)
    arms = {}
    for arm in ARMS:
        succeeded = [row["arms"][arm] for row in records if row["arms"][arm]["status"] == "complete"]
        latency = np.array([row["inference_ms"] for row in succeeded])
        arms[arm] = {"primary_numeric_parity": "failed", "metrics": metrics[arm], "images_in_scope": len(records),
                     "images_attempted": sum(row["arms"][arm].get("inference_attempted", False) for row in records),
                     "images_succeeded": len(succeeded), "inference_ms": {"mean": float(latency.mean()),
                     "p50": float(np.percentile(latency, 50)), "p95": float(np.percentile(latency, 95))} if len(latency) else None,
                     "latency_scope": "runtime inference only; shared decode/preprocess, postprocess/cache excluded"}
    return {"status": "failed" if errors else "complete", "arms": arms, "errors": errors,
            "images_attempted": len(records), "images_succeeded_both": sum(row["status"] == "complete" for row in records),
            "paired_wall_s": wall, "recipe": RECIPE, "data_sha256": data_sha,
            "ground_truth_sha256": identity["ground_truth_sha256"], "image_keys_sha256": identity["image_keys_sha256"],
            "class_layout": layout, "native": native, "input_assets_before_after_equal": True, "input_assets": before,
            "source_archive": {"revision": source.SOURCE_REVISION, "url": source.SOURCE_URL, "sha256": source.SOURCE_SHA},
            "plus_wheel": {"url": source.PLUS_URL, "sha256": source.PLUS_SHA}, "preprocessing": source.PREPROCESS,
            "primary_numeric_parity": "failed", "primary_failure_unchanged": True, "promotion": False,
            "target_tpu": "not_compiled_or_verified", "training_overlap_with_test": "unverified"}


def worker(args):
    report = {"schema_version": 1, "ok": False, "status": "running", "stage": "quality_comparison",
              "primary_numeric_parity": "failed", "primary_failure_unchanged": True, "promotion": False,
              "target_tpu": "not_compiled_or_verified", "tool": source.asset(__file__), "recipe": RECIPE,
              "source_revision": source.SOURCE_REVISION, "source_archive_sha256": source.SOURCE_SHA,
              "plus_wheel_sha256": source.PLUS_SHA, "expected_weights_sha256": source.WEIGHTS_SHA,
              "expected_original_onnx_sha256": GRAPH_SHA,
              "arms": {arm: {"metrics": None, "primary_numeric_parity": "failed"} for arm in ARMS}}
    # The parent watchdog can preserve parity/provenance framing even if this
    # worker is killed before its final write.
    source.write(args.output_dir / "manifest.json", report)
    try:
        report["archived_sources"] = archive_sources(args.output_dir)
        source.write(args.output_dir / "manifest.json", report)
        with source.offline_network():
            report["versions"] = source.verify_environment()
            report.update(evaluate(args))
        source.require(all(source.asset(item["source"]["path"]) == item["source"] for item in report["archived_sources"]),
                       "Archived harness/helper source changed during evaluation")
        report["ok"] = report["status"] == "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        for arm in report["arms"].values():
            arm["metrics"] = None
    source.write(args.output_dir / "manifest.json", report)
    return 0 if report["ok"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "weights", "export-dir", "primary-parity-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.)
    parser.add_argument("--max-rss-gb", type=float, default=12.)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    args.data = args.data.resolve()
    args.command = "quality_comparison"
    return worker(args) if args._worker else source.bounded_run(args, arguments, module="tools.evaluate_rfdetr_candidate")


if __name__ == "__main__":
    raise SystemExit(main())
