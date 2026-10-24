"""Small D-FINE contract/pipeline tests; no downloaded model inference."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
from PIL import Image

from guard.contracts import ModelSpec
from guard.pipeline.detector import BatchScheduler, Detector
from guard.vision.detr import decode_dfine, decode_dfine_batch
from guard.vision.preprocess import DirectResizeMeta, letterbox, resize_rgb

ROOT = Path(__file__).resolve().parents[1]


def spec(**overrides):
    fields = dict(model_id="det", onnx_path="unused.onnx", output_format="dfine", classes=["person", "cell phone"],
                  input_size=(8, 8), conf_threshold=.35)
    fields.update(overrides)
    return ModelSpec(**fields)


def outputs():
    return [np.array([[[4., 3.], [2., -1000.]]], np.float32),
            np.array([[[.5, .5, .4, .6], [.1, .1, .4, .4]]], np.float32)]


def test_direct_resize_matches_pil_source_processor_without_padding_or_normalization():
    frame = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    original = frame.copy()
    tensor, meta = resize_rgb(frame, (9, 7))
    expected = np.asarray(Image.fromarray(frame[..., ::-1]).resize((7, 9), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(tensor, expected.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255))
    assert tensor.shape == (1, 3, 9, 7) and tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert meta == DirectResizeMeta(orig_w=5, orig_h=3)
    np.testing.assert_array_equal(frame, original)
    yolo, _ = letterbox(frame, (9, 7))
    assert not np.array_equal(tensor, yolo)


@pytest.mark.parametrize("frame,size", [(np.zeros((0, 3, 3), np.uint8), (8, 8)),
    (np.zeros((3, 3), np.uint8), (8, 8)), (np.zeros((3, 3, 3), np.float32), (8, 8)),
    (np.zeros((3, 3, 3), np.uint8), (0, 8)), (np.zeros((3, 3, 3), np.uint8), (True, 8))])
def test_direct_resize_rejects_invalid_input(frame, size):
    with pytest.raises(ValueError):
        resize_rgb(frame, size)


def test_focal_topk_selects_query_class_pairs_without_argmax_or_nms():
    raw = outputs()
    before = [array.copy() for array in raw]
    detections = decode_dfine(raw, DirectResizeMeta(200, 100), spec(iou_threshold=0))
    # Top-Q=2, both pairs come from query 0; its two labels survive identical boxes.
    assert [det.cls for det in detections] == ["person", "cell phone"]
    assert [det.cls_id for det in detections] == [0, 1]
    np.testing.assert_allclose([det.bbox for det in detections], [[60, 20, 140, 80]] * 2, atol=1e-5)
    np.testing.assert_allclose([det.conf for det in detections], 1 / (1 + np.exp(-np.array([4., 3.]))))
    assert all(det.model == "det" and det.keypoints is None for det in detections)
    for array, original in zip(raw, before):
        np.testing.assert_array_equal(array, original)


def test_sigmoid_is_stable_threshold_is_strict_and_box_coordinates_are_clipped():
    raw = [np.array([[[1000.], [0.], [-1000.]]], np.float32),
           np.array([[[.05, .1, .4, .5], [.5, .5, .2, .2], [.5, .5, .2, .2]]], np.float32)]
    with np.errstate(over="raise", invalid="raise"):
        detections = decode_dfine(raw, DirectResizeMeta(200, 100), spec(classes=["person"], conf_threshold=.5))
    assert len(detections) == 1 and detections[0].conf == 1.
    np.testing.assert_allclose(detections[0].bbox, [0, 0, 50, 35], atol=1e-5)
    assert decode_dfine(raw, DirectResizeMeta(200, 100), spec(classes=["person"], conf_threshold=1.)) == []


def test_batch_decode_uses_original_dimensions_per_frame():
    raw = [np.repeat(array, 2, axis=0) for array in outputs()]
    result = decode_dfine_batch(raw, [DirectResizeMeta(200, 100), DirectResizeMeta(100, 200)], spec())
    np.testing.assert_allclose(result[0][0].bbox, [60, 20, 140, 80], atol=1e-5)
    np.testing.assert_allclose(result[1][0].bbox, [30, 40, 70, 160], atol=1e-5)
    with pytest.raises(ValueError, match="shape mismatch"):
        decode_dfine_batch(raw, [DirectResizeMeta(200, 100)], spec())


@pytest.mark.parametrize("index,value", [(0, np.nan), (0, np.inf), (0, -np.inf), (1, np.nan), (1, np.inf)])
def test_nonfinite_anywhere_fails_the_whole_frame_instead_of_silently_filtering(index, value):
    raw = outputs()
    raw[index].flat[-1] = value
    with pytest.raises(ValueError, match="Non-finite"):
        decode_dfine(raw, DirectResizeMeta(200, 100), spec())


@pytest.mark.parametrize("mutation", [lambda raw: raw[:1], lambda raw: raw[::-1],
    lambda raw: [raw[0][0], raw[1]], lambda raw: [raw[0], raw[1][:, :1]],
    lambda raw: [raw[0].astype(np.float64), raw[1]]])
def test_wrong_output_contract_is_an_explicit_error(mutation):
    with pytest.raises(ValueError):
        decode_dfine(mutation(outputs()), DirectResizeMeta(200, 100), spec())


def test_fdr_out_of_range_proposals_are_scaled_and_clipped_and_low_scores_do_not_fail_the_frame():
    # Actual pinned graph ends in affine FDR cxcywh, without output sigmoid/clamp.
    # "Normalized" is a unit convention, not a hard bound on every raw proposal.
    raw = [np.array([[[4.], [3.], [-1000.]]], np.float32),
           np.array([[[-.1, .5, .6, 1.4], [.5, .5, 2.4, 2.4], [-3., 4., 100., -7.]]], np.float32)]
    result = decode_dfine(raw, DirectResizeMeta(200, 100), spec(classes=["person"]))
    assert len(result) == 2
    np.testing.assert_allclose([det.bbox for det in result], [[0, 0, 40, 100], [0, 0, 200, 100]], atol=1e-5)


@pytest.mark.parametrize("overrides", [{"task": "pose"}, {"input_dtype": "uint8"}, {"classes": []},
    {"classes": ["person", "person"]}, {"conf_threshold": float("nan")}])
def test_detector_rejects_unsupported_dfine_spec_before_inference(overrides):
    with pytest.raises(ValueError):
        Detector(spec(**overrides), object())


def test_format_is_explicit_and_yolo_never_guesses_dfine_from_shapes():
    detector = Detector(spec(output_format="yolov8"), object())
    _, meta = detector.preprocess(np.zeros((100, 200, 3), np.uint8))
    with pytest.raises(ValueError, match="one concatenated"):
        detector.postprocess(outputs(), meta)
    with pytest.raises(ValueError, match="direct resize"):
        decode_dfine(outputs(), meta, spec())


def test_adapter_import_and_decode_require_no_torch_torchvision_or_transformers():
    code = '''
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'transformers'}:
        raise AssertionError('Heavy ML framework imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import numpy as np
from guard.contracts import ModelSpec
from guard.pipeline.detector import Detector
model = ModelSpec(model_id='det', onnx_path='unused', output_format='dfine', classes=['person'], input_size=(8,8))
detector = Detector(model, object())
_, meta = detector.preprocess(np.zeros((10,20,3), np.uint8))
raw = [np.array([[[2.]]], np.float32), np.array([[[.5,.5,.4,.4]]], np.float32)]
assert len(detector.postprocess(raw, meta)) == 1
'''
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10)


def make_graph(path, *, queries=2, classes=2, output_names=("logits", "pred_boxes"), height="height"):
    """Tiny dynamic-batch raw graph; only Expand of small known tensors."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    logits = np.full((1, queries, classes), -20., np.float32)
    logits[0, 0, 0] = 4.
    logits[0, 1, 1] = 3.
    boxes = np.tile(np.array([.5, .5, .4, .6], np.float32), (1, queries, 1))
    initializers = [numpy_helper.from_array(value, name=name) for value, name in (
        (logits, "base_logits"), (boxes, "base_boxes"), (np.array([0], np.int64), "axis0"),
        (np.array([queries, classes], np.int64), "logit_tail"), (np.array([queries, 4], np.int64), "box_tail"))]
    nodes = [helper.make_node("Shape", ["pixel_values"], ["input_shape"]),
             helper.make_node("Gather", ["input_shape", "axis0"], ["batch"]),
             helper.make_node("Concat", ["batch", "logit_tail"], ["logit_shape"], axis=0),
             helper.make_node("Concat", ["batch", "box_tail"], ["box_shape"], axis=0),
             helper.make_node("Expand", ["base_logits", "logit_shape"], [output_names[0]]),
             helper.make_node("Expand", ["base_boxes", "box_shape"], [output_names[1]])]
    graph = helper.make_graph(nodes, "tiny-raw-dfine", [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT,
                               ["batch_size", 3, height, "width"])],
                              [helper.make_tensor_value_info(output_names[0], TensorProto.FLOAT, ["batch_size", queries, classes]),
                               helper.make_tensor_value_info(output_names[1], TensorProto.FLOAT, ["batch_size", queries, 4])], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def test_real_tiny_cpu_onnx_through_production_batcher_uses_dfine_preprocess_and_per_frame_boxes(tmp_path):
    from guard.inference.onnx_cpu import OnnxCpuBackend

    path = make_graph(tmp_path / "tiny.onnx")
    model = spec(onnx_path=str(path))
    backend = OnnxCpuBackend(intra_op_threads=2)
    backend.load(model)
    scheduler = BatchScheduler(backend, "det", max_batch=2, timeout_s=.5)
    detector = Detector(model, backend, scheduler)
    barrier = Barrier(2)

    def detect(shape):
        frame = np.full((*shape, 3), [10, 20, 30], np.uint8)
        tensor, meta = detector.preprocess(frame)
        assert isinstance(meta, DirectResizeMeta)
        np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([30, 20, 10]) / 255)
        timings = {}
        barrier.wait(timeout=3)
        return detector.detect(frame, timings), timings

    try:
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(detect, [(100, 200), (200, 100)]))
        assert [[det.cls for det in result] for result, _ in results] == [["person", "cell phone"]] * 2
        np.testing.assert_allclose(results[0][0][0].bbox, [60, 20, 140, 80], atol=1e-5)
        np.testing.assert_allclose(results[1][0][0].bbox, [30, 40, 70, 160], atol=1e-5)
        assert backend.stats()["models"]["det"]["calls"] == 1
        assert all(set(timings) == {"preprocess", "queue", "transfer", "infer", "postprocess"} for _, timings in results)
    finally:
        scheduler.stop()
        backend.close()


def test_batcher_rejects_wrong_backend_batch_for_every_waiting_frame():
    class WrongBatch:
        def infer_timed(self, model_id, batch):
            return [np.zeros((3, 2, 2), np.float32)], {}

    scheduler = BatchScheduler(WrongBatch(), "det", max_batch=2, timeout_s=.5)
    try:
        futures = [scheduler.submit(np.zeros((1, 3, 8, 8), np.float32)) for _ in range(2)]
        for future in futures:
            with pytest.raises(ValueError, match="output batch size"):
                future.result(timeout=3)
    finally:
        scheduler.stop()


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    import ml.eval.inspect_dfine as inspector
    from ml.registry import load_specs

    classes = load_specs(ROOT / "configs/models.dfine-x-640.yaml")["det"].classes
    config = {"model_type": "d_fine", "num_queries": 300, "use_focal_loss": True,
              "id2label": {str(index): name for index, name in enumerate(classes)}}
    processor = {"do_resize": True, "do_rescale": True, "do_normalize": False, "do_pad": False,
                 "size": {"height": 640, "width": 640}, "resample": 2, "rescale_factor": 1 / 255,
                 "image_processor_type": "RTDetrImageProcessor"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(processor))

    def build(**kwargs):
        path = make_graph(tmp_path / "model.onnx", queries=300, classes=80, **kwargs)
        monkeypatch.setattr(inspector, "DFINE_FILES", {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
                                                     for name in inspector.DFINE_FILES})
        return path, classes

    return build


def test_inspector_checks_dynamic_graph_source_classes_and_pins_before_any_ort(artifact):
    from ml.eval.inspect_dfine import inspect_dfine_onnx

    path, classes = artifact()
    report = inspect_dfine_onnx(path, classes)
    assert report["input_shape"] == ["batch_size", 3, "height", "width"]
    assert report["output_shapes"] == [["batch_size", 300, 80], ["batch_size", 300, 4]]
    assert report["effective_input_shape"] == [1, 3, 640, 640] and report["top_k"] == 300
    assert classes[0] == "person" and classes[67] == "cell phone"
    with pytest.raises(ValueError, match="id2label"):
        inspect_dfine_onnx(path, classes[::-1])
    with pytest.raises(ValueError, match="640x640"):
        inspect_dfine_onnx(path, classes, (1280, 1280))
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        inspect_dfine_onnx(path, classes)


@pytest.mark.parametrize("kwargs,error", [({"output_names": ("pred_boxes", "logits")}, "ordered"),
                                        ({"height": 1280}, "shape mismatch")])
def test_inspector_rejects_wrong_names_and_shape_even_if_test_pin_matches(artifact, kwargs, error):
    from ml.eval.inspect_dfine import inspect_dfine_onnx

    path, classes = artifact(**kwargs)
    with pytest.raises(ValueError, match=error):
        inspect_dfine_onnx(path, classes)


@pytest.mark.parametrize("name,key,value,error", [("config.json", "use_focal_loss", False, "decoding"),
    ("preprocessor_config.json", "do_normalize", True, "processor"),
    ("preprocessor_config.json", "do_pad", True, "processor")])
def test_inspector_rejects_unimplemented_source_recipe_even_if_test_pin_matches(artifact, monkeypatch, name, key, value, error):
    import ml.eval.inspect_dfine as inspector

    path, classes = artifact()
    source = path.parent / name
    raw = json.loads(source.read_text())
    raw[key] = value
    source.write_text(json.dumps(raw))
    monkeypatch.setitem(inspector.DFINE_FILES, name, hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match=error):
        inspector.inspect_dfine_onnx(path, classes)


def test_dfine_recipe_is_explicit_preserves_reference_and_does_not_weaken_ppe_contract():
    from ml.eval.run_ppe_onnx import RECIPE, effective_recipe, validate_raw_outputs
    from ml.registry import load_specs

    original = copy.deepcopy(RECIPE)
    model = load_specs(ROOT / "configs/models.dfine-x-640.yaml")["det"]
    recipe = effective_recipe(model)
    assert recipe["output_format"] == "dfine" and recipe["top_k"] == recipe["queries"] == 300
    assert recipe["nms_applied"] is False and recipe["score_comparison"] == ">"
    for key in ("conf_ap", "conf_op", "nms_iou", "threads", "ap", "data_sha256", "expected_images"):
        assert recipe[key] == original[key]
    assert RECIPE == original
    valid = [np.zeros((1, 300, 80), np.float32), np.full((1, 300, 4), .5, np.float32)]
    validate_raw_outputs(valid, 80, recipe)
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_raw_outputs([array[:, :200] for array in valid], 80, recipe)
    with pytest.raises(ValueError, match="raw YOLOv8"):
        effective_recipe(model, candidate=True)
    with pytest.raises(ValueError, match="640x640"):
        effective_recipe(model.model_copy(update={"input_size": (1280, 1280)}))


@pytest.mark.parametrize("failed", [False, True])
def test_eval_uses_production_dfine_detector_and_failures_never_become_empty_predictions(artifact, monkeypatch, failed):
    import cv2
    import yaml
    import guard.inference.onnx_cpu as cpu
    import ml.eval.run_ppe_onnx as runner
    from ml.registry import load_specs

    path, classes = artifact()
    model = load_specs(ROOT / "configs/models.dfine-x-640.yaml")["det"]
    model = model.model_copy(update={"onnx_path": path.name})
    config = path.parent / "registry.yaml"
    config.write_text(yaml.safe_dump({"models": [model.model_dump(mode="json")]}))
    image = path.parent / "images" / "test" / "frame.jpg"
    image.parent.mkdir(parents=True)
    cv2.imwrite(str(image), np.zeros((100, 200, 3), np.uint8))
    gt = {name: {"test/frame.jpg": np.array([[60., 20., 140., 80.]])} for name in ("person", "helmet", "vest")}
    monkeypatch.setattr(runner, "dataset_inputs", lambda data: ([image], gt, [], "fixture-fingerprint"))
    monkeypatch.setattr(runner, "inspect_detector_onnx", lambda *args: pytest.fail("D-FINE must not use YOLO inspector"))
    if failed:
        real_infer = cpu.OnnxCpuBackend.infer_timed

        def bad_output(self, model_id, batch):
            raw, timings = real_infer(self, model_id, batch)
            raw[0][0, -1, -1] = np.nan
            return raw, timings

        monkeypatch.setattr(cpu.OnnxCpuBackend, "infer_timed", bad_output)
    code, output = runner.run(path.parent, path.parent, config, path.parent / "reports", model_id="det")
    report = json.loads((output / "report.json").read_text())
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert code == int(failed) and report["status"] == ("failed" if failed else "complete")
    assert report["recipe"]["output_format"] == "dfine" and report["recipe"]["nms_applied"] is False
    assert report["provenance"]["data_sha256"] == "fixture-fingerprint"
    assert report["provenance"]["class_names_source"] == "registry"
    assert report["provenance"]["inspection"]["classes"] == classes
    assert report["images_attempted"] == 1 and report["images_succeeded"] == int(not failed)
    assert rows[0]["image_key"] == "test/frame.jpg" and rows[0]["status"] == report["status"].replace("complete", "ok")
    if failed:
        assert report["metrics"] is None and len(report["errors"]) == 1
        assert "Non-finite" in report["errors"][0]["error"] and "detections" not in rows[0]
    else:
        assert report["metrics"]["coverage"]["core"]["denominator"] == 3
        assert report["metrics"]["coverage"]["per_class"]["helmet"]["status"] == "unsupported"
        assert report["metrics"]["core_ap50(person,helmet,vest)"] == .333  # public metrics round to 3 decimals
        assert [det["cls"] for det in rows[0]["detections"]] == ["person", "bicycle"]
        assert json.loads((output / "semantic-smoke.json").read_text())["status"] == "not_applicable"
