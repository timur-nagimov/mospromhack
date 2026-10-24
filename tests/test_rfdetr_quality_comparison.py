"""Paired quality tests with synthetic outputs; never import RF-DETR or run ORT."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from ml.eval_ppe import CORE_AP50
from tools import evaluate_rfdetr_candidate as quality
from tools import prepare_rfdetr_candidate as source


def class_layout():
    names = {index: f"category {index}" for index in source.COCO_IDS}
    names.update({1: "person", 77: "cell phone", 90: "toothbrush"})
    return source.make_class_layout(names)


def raw_outputs():
    boxes = np.tile(np.array([.5, .5, 1., 1.], np.float32), (1, 300, 1))
    logits = np.full((1, 300, 91), -12., np.float32)
    logits[0, 0, [1, 77, 90, 0]] = [2., 1., .5, -.5]
    return boxes, logits


def fake_postprocess(raw, size):
    # Contract stub; production delegates to the hash-pinned official PostProcess.
    h, w = size
    labels = np.zeros(300, np.int64)
    labels[:4] = [1, 77, 90, 0]
    probabilities = torch.from_numpy(raw[1]).sigmoid().numpy().copy()
    scores = np.full(300, .00001, np.float32)
    scores[:4] = probabilities[0, 0, labels[:4]]
    return {"boxes": np.tile(np.array([0, 0, w, h], np.float32), (300, 1)), "scores": scores,
            "labels": labels, "all_scores": probabilities}


def image_scope(tmp_path, count=2):
    data = tmp_path / "dataset"
    directory, output = data / "images/test", tmp_path / "output"
    directory.mkdir(parents=True)
    output.mkdir()
    images = []
    for i in range(count):
        path = directory / f"frame-{i}.png"
        Image.new("RGB", (5, 3), (255, 128, i)).save(path)
        images.append(path)
    return data, images, output


def test_each_image_has_one_source_preprocess_tensor_shared_by_both_arms_and_all91_saved(tmp_path):
    data, images, output = image_scope(tmp_path)
    preprocessed, native_tensors, onnx_tensors = [], [], []
    def preprocess(rgb):
        preprocessed.append(rgb.copy())
        return source.preprocess_rgb(rgb)
    def native(tensor):
        native_tensors.append(tensor.numpy())
        return raw_outputs()
    def onnx(batch):
        onnx_tensors.append(batch)
        assert np.shares_memory(native_tensors[-1], batch)
        return raw_outputs()
    predictions, records, errors = quality.infer_pairs(images, data, output, class_layout(),
                                                      {"native": native, "onnx": onnx}, fake_postprocess, preprocess=preprocess)
    assert not errors and len(records) == len(preprocessed) == 2
    assert set(predictions["native"]) == {"person"} and len(predictions["native"]["person"]) == 2
    for row, native_input, onnx_input in zip(records, native_tensors, onnx_tensors):
        assert quality.array_info(native_input) == quality.array_info(onnx_input) == row["input"]
        assert row["primary_numeric_parity"] == "failed"
        assert row["arms"]["native"]["predictions"][1]["native_class_id"] == 77
        assert row["arms"]["native"]["predictions"][2]["native_class_id"] == 90
        assert row["arms"]["native"]["predictions"][3]["native_class_name"] is None
        with np.load(row["tensor_file"]["path"], allow_pickle=False) as cache:
            for arm in quality.ARMS:
                assert cache[arm + "_raw_logits"].shape == cache[arm + "_all_scores"].shape == (1, 300, 91)
                assert cache[arm + "_all_scores"][0, 0, 90] > .6  # Class90 was not discarded as background.
    expected = (np.array([1., 128 / 255, 0.]) - source.PREPROCESS["mean"]) / source.PREPROCESS["std"]
    np.testing.assert_allclose(native_tensors[0][0, :, 0, 0], expected, atol=1e-6)
    assert len((output / "images.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("failure", ["nan", "shape", "raise"])
def test_any_runtime_error_keeps_all_frame_records_and_prevents_partial_quality_success(tmp_path, failure):
    data, images, output = image_scope(tmp_path)
    calls = []
    def onnx(batch):
        calls.append(1)
        if failure == "raise": raise RuntimeError("fake inference failed")
        boxes, logits = raw_outputs()
        if failure == "nan": logits[0, 0, 0] = np.nan
        if failure == "shape": logits = logits[..., :80]
        return boxes, logits
    predictions, records, errors = quality.infer_pairs(images, data, output, class_layout(),
                                                      {"native": lambda tensor: raw_outputs(), "onnx": onnx}, fake_postprocess)
    assert len(calls) == len(records) == len(errors) == 2
    assert all(row["arms"]["native"]["status"] == "complete" and row["arms"]["onnx"]["status"] == "failed" for row in records)
    assert quality.score_complete(predictions, {}, class_layout(), records, errors) == {"native": None, "onnx": None}
    assert len((output / "images.jsonl").read_text().splitlines()) == 2


def test_mutating_native_input_prevents_onnx_from_receiving_a_different_tensor(tmp_path):
    data, images, output = image_scope(tmp_path, count=1)
    def mutated(tensor):
        tensor.add_(1)
        return raw_outputs()
    def forbidden(batch):
        pytest.fail("ONNX cannot receive a tensor changed by the native arm")
    _, records, errors = quality.infer_pairs(images, data, output, class_layout(), {"native": mutated, "onnx": forbidden}, fake_postprocess)
    assert errors and records[0]["status"] == "failed"
    assert "mutated" in records[0]["arms"]["native"]["error"]
    assert not records[0]["arms"]["onnx"].get("inference_attempted", False)


def test_unreadable_frame_has_explicit_not_run_arms_and_later_frames_are_still_attempted(tmp_path):
    data, images, output = image_scope(tmp_path)
    attempted = []
    def reader(path):
        if path == images[0]: raise ValueError("decode failed")
        return quality.read_rgb(path)
    def call(value):
        attempted.append(1)
        return raw_outputs()
    _, records, errors = quality.infer_pairs(images, data, output, class_layout(), dict.fromkeys(quality.ARMS, call),
                                            fake_postprocess, reader=reader)
    assert len(records) == 2 and len(attempted) == 2 and len(errors) == 1
    assert all(arm["status"] == "not_run" for arm in records[0]["arms"].values())
    assert records[1]["status"] == "complete"


def test_exif_orientation_is_rejected_instead_of_rotating_dataset_coordinates(tmp_path):
    image = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (4, 2)).save(image, exif=exif)
    with pytest.raises(ValueError, match="EXIF orientation"):
        quality.read_rgb(image)


def complete_records():
    return [{"image_key": f"test/{i}.jpg", "status": "complete", "arms": {arm: {"status": "complete"} for arm in quality.ARMS}}
            for i in range(141)]


def test_shared_ap101_engine_preserves_person236_and_unsupported_core_denominator():
    boxes = np.array([[i * 10, 0, i * 10 + 5, 5] for i in range(236)], float)
    key = "test/0.jpg"
    gt = {"person": {key: boxes}, "helmet": {key: boxes[:1]}, "vest": {key: boxes[:1]}}
    predictions = {arm: {"person": [(key, .9, box) for box in boxes]} for arm in quality.ARMS}
    metrics = quality.score_complete(predictions, gt, class_layout(), complete_records(), [])
    for arm in quality.ARMS:
        row = metrics[arm]
        assert row["person"] == {"ap50": 1., "p": 1., "r": 1., "n_gt": 236}
        assert row[CORE_AP50] == .333 and row["coverage"]["core"]["denominator"] == 3
        assert row["helmet"]["ap50"] == row["vest"]["ap50"] == 0
        assert row["coverage"]["per_class"]["no_vest"]["status"] == "no_ground_truth"
        assert row["no_vest"] is None


@pytest.mark.parametrize("failure", ["missing", "duplicate", "arm", "error"])
def test_missing_duplicate_failed_or_error_rows_never_yield_metrics(failure):
    rows = complete_records()
    errors = []
    if failure == "missing": rows.pop()
    if failure == "duplicate": rows[-1]["image_key"] = rows[0]["image_key"]
    if failure == "arm": rows[0]["arms"]["onnx"]["status"] = "failed"
    if failure == "error": errors.append({"error": "failure"})
    assert quality.score_complete({}, {}, class_layout(), rows, errors) == {"native": None, "onnx": None}


def dataset_fixture(tmp_path):
    data = tmp_path / "dataset"
    images = [data / "images/test" / f"{i}.jpg" for i in range(141)]
    records = [{"relative_path": f"images/test/{i}.jpg", "sha256": f"{i:064x}"} for i in range(141)]
    gt = {"person": {"test/0.jpg": np.array([[i * 10, 0, i * 10 + 5, 5] for i in range(236)], float)}}
    return data, images, records, gt


@pytest.mark.parametrize("failure", [None, "image", "content", "gt_count", "gt_duplicate", "gt_scope"])
def test_frozen_scope_checks_exact141_unique_images_and236_valid_gt(tmp_path, failure):
    data, images, records, gt = dataset_fixture(tmp_path)
    if failure == "image": images[-1] = images[0]
    if failure == "content": records[-1]["sha256"] = records[0]["sha256"]
    if failure == "gt_count": gt["person"]["test/0.jpg"] = gt["person"]["test/0.jpg"][:-1]
    if failure == "gt_duplicate": gt["person"]["test/0.jpg"][-1] = gt["person"]["test/0.jpg"][0]
    if failure == "gt_scope": gt["person"]["val/0.jpg"] = gt["person"].pop("test/0.jpg")
    if failure:
        with pytest.raises(ValueError): quality.validate_dataset(images, gt, records, data)
    else:
        result = quality.validate_dataset(images, gt, records, data)
        assert result["person_gt"] == 236 and len(result["image_keys"]) == 141
        assert len(result["ground_truth_sha256"]) == len(result["image_keys_sha256"]) == 64


def test_postprocess_source_drift_fails_before_any_model_import(tmp_path, monkeypatch):
    file = tmp_path / "models/postprocess.py"
    file.parent.mkdir()
    file.write_text("pinned synthetic source")
    monkeypatch.setattr(quality, "POSTPROCESS_FILES", {"models/postprocess.py": source.sha(file)})
    monkeypatch.setattr(quality.importlib.metadata, "distribution", lambda name: SimpleNamespace(locate_file=lambda item: tmp_path))
    assert file in quality.code_assets()
    file.write_text("changed")
    with pytest.raises(ValueError, match="postprocess source drift"):
        quality.code_assets()


def test_one_official_postprocessor_instance_serves_both_runtimes_without_export_mode(monkeypatch):
    instances = []
    class Model:
        training, _export = False, False
        def __call__(self, tensor):
            raw = raw_outputs()
            return {"pred_boxes": torch.from_numpy(raw[0]), "pred_logits": torch.from_numpy(raw[1])}
    model = Model()
    class PostProcess:
        def __init__(self, num_select):
            assert num_select == 300
            self.calls = 0
            instances.append(self)
        def eval(self): return self
        def __call__(self, outputs, sizes):
            self.calls += 1
            assert sizes.tolist() == [[3, 5]] and sizes.dtype == torch.int64
            value = fake_postprocess([outputs["pred_boxes"].numpy(), outputs["pred_logits"].numpy()], (3, 5))
            return [{name: torch.from_numpy(value[name]) for name in ("boxes", "scores", "labels")}]
    class Session:
        def __init__(self, *args, **kwargs): assert kwargs["providers"] == ["CPUExecutionProvider"]
        def get_providers(self): return ["CPUExecutionProvider"]
        def run(self, names, feed):
            assert names == ["dets", "labels"] and set(feed) == {"images"}
            return raw_outputs()
    monkeypatch.setitem(sys.modules, "rfdetr.models.postprocess", SimpleNamespace(PostProcess=PostProcess))
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace, InferenceSession=Session))
    monkeypatch.setattr(source, "build_native", lambda path: (model, {"config": {"num_select": 300}}))
    calls, postprocess, _ = quality.make_runtimes(Path("unused.pth"), Path("original.onnx"))
    tensor = torch.zeros((1, 3, 2, 2))
    a, b = (postprocess(calls[arm](tensor if arm == "native" else tensor.numpy()), (3, 5)) for arm in quality.ARMS)
    assert len(instances) == 1 and instances[0].calls == 2
    np.testing.assert_array_equal(a["all_scores"], b["all_scores"])
    assert a["all_scores"].shape[-1] == 91 and a["all_scores"][0, 0].sum() > 1  # sigmoid, never softmax
    assert a["labels"][2] == 90 and a["labels"][1] == 77
    model._export = True
    with pytest.raises(ValueError, match="export/training mode"):
        calls["native"](tensor)


def test_reexport_or_debug_graph_pin_is_rejected_before_dataset_or_model_loading(tmp_path, monkeypatch):
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"not the pinned original graph")
    source.write(tmp_path / "manifest.json", {"ok": True, "source_revision": source.SOURCE_REVISION, "onnx": source.asset(graph)})
    monkeypatch.setattr(quality, "load_primary", lambda path: ({}, [], {}))
    monkeypatch.setattr(source, "inspect_graph", lambda path: pytest.fail("Reject wrong bytes before graph/model work"))
    with pytest.raises(ValueError, match="original pinned static ONNX"):
        quality.evaluate(SimpleNamespace(export_dir=tmp_path, primary_parity_dir=tmp_path))


@pytest.mark.parametrize("mutation", [None, "weights", "image", "primary"])
def test_paired_run_checks_all_input_assets_afterward_and_never_scores_drift(tmp_path, monkeypatch, mutation):
    from ml.eval import run_ppe_onnx
    data, images, output = image_scope(tmp_path)
    weights, graph, primary = tmp_path / "weights", tmp_path / "model.onnx", tmp_path / "primary.json"
    for file in (weights, graph, primary): file.write_bytes(b"pinned synthetic asset")
    source.write(tmp_path / "manifest.json", {"ok": True, "source_revision": source.SOURCE_REVISION, "onnx": source.asset(graph)})
    monkeypatch.setattr(quality, "GRAPH_SHA", source.sha(graph))
    monkeypatch.setattr(quality, "GRAPH_BYTES", graph.stat().st_size)
    monkeypatch.setattr(quality, "load_primary", lambda path: ({}, [], {str(primary): source.asset(primary)}))
    monkeypatch.setattr(quality, "code_assets", lambda: [Path(quality.__file__), Path(source.__file__)])
    monkeypatch.setattr(source, "inspect_graph", lambda path: {})
    monkeypatch.setattr(source, "class_layout", class_layout)
    monkeypatch.setattr(source, "verify_environment", lambda: {"synthetic": True})
    monkeypatch.setitem(quality.RECIPE, "images", 2)
    monkeypatch.setitem(quality.RECIPE, "person_gt", 2)
    gt = {"person": {f"test/{image.name}": np.array([[0, 0, 5, 3]], float) for image in images},
          "helmet": {f"test/{images[0].name}": np.array([[0, 0, 5, 3]], float)},
          "vest": {f"test/{images[0].name}": np.array([[0, 0, 5, 3]], float)}}
    records = [{"relative_path": "images/test/" + image.name, **source.asset(image)} for image in images]
    monkeypatch.setattr(run_ppe_onnx, "dataset_inputs", lambda path: (images, gt, records, run_ppe_onnx.DATA_TEST_FINGERPRINT))
    def onnx(batch):
        target = {"weights": weights, "image": images[0], "primary": primary}.get(mutation)
        if target is not None: target.write_bytes(b"changed during paired evaluation")
        return raw_outputs()
    monkeypatch.setattr(quality, "make_runtimes", lambda weights, graph: ({"native": lambda tensor: raw_outputs(), "onnx": onnx},
                                                                         fake_postprocess, {"state_coverage": "synthetic"}))
    args = SimpleNamespace(data=data, weights=weights, export_dir=tmp_path, primary_parity_dir=tmp_path, output_dir=output)
    code = quality.worker(args)
    report = json.loads((output / "manifest.json").read_text())
    assert report["primary_numeric_parity"] == "failed" and report["promotion"] is False
    assert len((output / "images.jsonl").read_text().splitlines()) == 2
    if mutation:
        assert code == 1 and not report["ok"] and "asset changed" in report["error"]
        assert all(arm["metrics"] is None for arm in report["arms"].values())
    else:
        assert code == 0 and report["ok"] and report["images_succeeded_both"] == 2
        assert all(arm["metrics"]["person"]["n_gt"] == 2 for arm in report["arms"].values())


def test_running_manifest_already_contains_failed_parity_if_worker_is_interrupted(tmp_path, monkeypatch):
    def interrupted(): raise KeyboardInterrupt
    monkeypatch.setattr(source, "verify_environment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        quality.worker(SimpleNamespace(output_dir=tmp_path))
    report = json.loads((tmp_path / "manifest.json").read_text())
    assert not report["ok"] and report["status"] == "running" and report["primary_numeric_parity"] == "failed"
    assert report["expected_original_onnx_sha256"] == quality.GRAPH_SHA


def test_late_archived_code_drift_clears_metrics_instead_of_leaving_stale_success(tmp_path, monkeypatch):
    file = tmp_path / "harness.py"
    file.write_text("old")
    initial = source.asset(file)
    monkeypatch.setattr(quality, "archive_sources", lambda output: [{"source": initial}])
    monkeypatch.setattr(source, "verify_environment", lambda: {})
    def changed(args):
        file.write_text("new")
        return {"status": "complete", "arms": {arm: {"metrics": {"person": "stale"}} for arm in quality.ARMS}}
    monkeypatch.setattr(quality, "evaluate", changed)
    assert quality.worker(SimpleNamespace(output_dir=tmp_path)) == 1
    report = json.loads((tmp_path / "manifest.json").read_text())
    assert not report["ok"] and report["status"] == "failed"
    assert all(arm["metrics"] is None for arm in report["arms"].values())


def test_environment_failure_and_bounded_cli_keep_failed_parity_and_no_promotion(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "verify_environment", lambda: (_ for _ in ()).throw(ValueError("pinned dependency drift")))
    monkeypatch.setattr(quality, "evaluate", lambda args: pytest.fail("No inference after environment failure"))
    assert quality.worker(SimpleNamespace(output_dir=tmp_path)) == 1
    report = json.loads((tmp_path / "manifest.json").read_text())
    assert not report["ok"] and report["status"] == "failed" and report["primary_numeric_parity"] == "failed"
    assert not report["promotion"] and all(arm["metrics"] is None for arm in report["arms"].values())
    assert len(report["archived_sources"]) == 6
    assert all(item["source"]["sha256"] == item["archive"]["sha256"] for item in report["archived_sources"])
    argv = ["--" + name + "=" + str(tmp_path / name) for name in ("data", "weights", "export-dir", "primary-parity-dir", "output-dir")]
    captured = []
    monkeypatch.setattr(source, "bounded_run", lambda args, argv, **kw: captured.append(kw["module"]) or 17)
    assert quality.main(argv) == 17 and captured == ["tools.evaluate_rfdetr_candidate"]
