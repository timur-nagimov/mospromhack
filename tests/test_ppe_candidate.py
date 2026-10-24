"""Candidate safety, production class mapping and evaluation failure semantics."""

import json
import stat
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from guard.contracts import Detection
from guard.inference.base import InferenceBackend
from guard.pipeline.detector import Detector
from ml.eval.run_ppe_onnx import dataset_inputs, infer_images, semantic_smoke
from ml.registry import load_specs
from tools.prepare_ppe_candidate import (
    CLASSES, DATA_NAMES, MODEL_NAME, SOURCE_CLASSES, download, get_profile, safe_extract_test,
)

ROOT = Path(__file__).resolve().parents[1]


def test_opt_in_registry_has_explicit_canonical_classes_and_separate_baseline():
    baseline = load_specs(ROOT / "configs/models.cpu-baseline.yaml")
    candidate = load_specs(ROOT / "configs/models.ppe-candidate.yaml")
    assert "ppe" not in baseline
    assert candidate["ppe"].classes == CLASSES
    assert len(CLASSES) == len(SOURCE_CLASSES) == 14
    assert candidate["ppe"].onnx_path == f"candidates/buildvision-hazards-v0.3/{MODEL_NAME}"
    assert candidate["det"].onnx_path == baseline["det"].onnx_path
    assert "respirator" not in CLASSES and "no_respirator" not in CLASSES
    assert CLASSES[5] == "no-mask"


class RawBackend(InferenceBackend):
    def __init__(self, output):
        self.output = output

    def load(self, spec):
        pass

    def infer(self, model_id, batch):
        assert batch.shape == (1, 3, 640, 640)
        return [self.output]

    def stats(self):
        return {}


def _production_detections(classes, profile_name="buildvision"):
    profile = get_profile(profile_name)
    rows = np.zeros((1, 4 + len(profile.classes), len(classes) + 1), dtype=np.float32)
    rows[0, :4, 0] = (320, 320, 400, 600)  # person, head y 20..230; torso y 140..470
    rows[0, 4 + profile.classes.index("person"), 0] = 0.95
    for index, name in enumerate(classes, 1):
        rows[0, :4, index] = (320, 100 if "vest" not in name.lower() else 300, 60, 50)
        rows[0, 4 + profile.source_classes.index(name), index] = 0.9
    spec = load_specs(ROOT / "configs" / profile.registry)["ppe"]
    return Detector(spec, RawBackend(rows)).detect(np.zeros((640, 640, 3), dtype=np.uint8))


@pytest.mark.parametrize(("source", "canonical", "attribute"), [
    ("no-hardhat", "no_helmet", "helmet"),
    ("no-safety vest", "no_vest", "vest"),
])
def test_negative_source_class_maps_canonically_but_one_observation_stays_unknown(source, canonical, attribute):
    detections = _production_detections([source, "no-mask"])
    assert canonical in {det.cls for det in detections}
    result = semantic_smoke(detections, np.zeros((640, 640, 3), np.uint8), 0)
    assert canonical in result["canonical_ppe_classes"]
    assert result["people"][0]["attributes"][attribute] is None
    assert result["people"][0]["attributes"]["respirator"] is None
    assert result["violations"] == []


def test_no_mask_and_missing_predictions_preserve_unknown():
    detections = _production_detections(["no-mask"])
    result = semantic_smoke(detections, np.zeros((640, 640, 3), np.uint8), 0)
    assert result["people"][0]["attributes"] == {"helmet": None, "vest": None, "respirator": None}
    assert result["violations"] == []


def test_positive_ppe_mapping_does_not_report_violation():
    result = semantic_smoke(_production_detections(["hardhat", "safety vest"]),
                            np.zeros((640, 640, 3), np.uint8), 0)
    assert result["people"][0]["attributes"] == {"helmet": True, "vest": True, "respirator": None}
    assert result["violations"] == []


def test_per_image_inference_failure_is_preserved_and_not_counted_as_empty(tmp_path):
    images = []
    for index in range(2):
        image = tmp_path / "images/test" / f"{index}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.zeros((10, 10, 3), np.uint8))
        images.append(image)

    class FailingDetector:
        calls = 0

        def detect(self, frame, timings=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("deliberate backend failure")
            return [Detection(model="det", cls="Person", cls_id=0, conf=0.9, bbox=(0, 0, 8, 8))]

    predictions, rows, errors = infer_images(FailingDetector(), images, tmp_path, tmp_path, include_smoke=False)
    assert len(errors) == 1 and errors[0]["image_key"] == "test/1.jpg"
    assert rows[1]["status"] == "failed" and "detections" not in rows[1]
    assert len(predictions["person"]) == 1 and predictions["person"][0][0] == "test/0.jpg"
    saved = [json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert saved[1]["error"] == "RuntimeError: deliberate backend failure"


def test_cached_corrupt_download_fails_without_network(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download("https://example.invalid/model.onnx", path, max_bytes=20, expected_sha="0" * 64)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "images/../escape", "C:/escape", "images\\test\\escape"])
def test_archive_rejects_unsafe_paths_before_writing(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"x")
    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        safe_extract_test(archive, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_archive_rejects_symlink(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("images/test/evil.jpg")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "../../../outside")
    with pytest.raises(ValueError, match="symlinks"):
        safe_extract_test(archive, tmp_path / "out")


def test_archive_extracts_test_only_and_rejects_overwrite(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for index in range(141):
            z.writestr(f"images/test/{index}.jpg", b"image")
            z.writestr(f"labels/test/{index}.txt", "")
        z.writestr("images/train/not-used.jpg", "not used")
        z.writestr("images/val/not-used.jpg", "not used")
    destination = tmp_path / "out"
    safe_extract_test(archive, destination)
    assert len(list((destination / "images/test").iterdir())) == 141
    assert not (destination / "images/train").exists() and not (destination / "images/val").exists()
    with pytest.raises(ValueError, match="already exists"):
        safe_extract_test(archive, destination)


def test_dataset_preflight_rejects_missing_label_before_inference(tmp_path):
    import yaml

    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": dict(enumerate(DATA_NAMES))}))
    (tmp_path / "images/test").mkdir(parents=True)
    for index in range(141):
        cv2.imwrite(str(tmp_path / "images/test" / f"{index}.jpg"), np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(ValueError, match="Missing label"):
        dataset_inputs(tmp_path)


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("size,grid", [(640, 8400), (1280, 33600)])
def test_generic_baseline_run_resolves_metadata_classes_and_never_scores_failed_images(tmp_path, monkeypatch, failed, size, grid):
    import guard.inference.onnx_cpu as cpu
    import ml.eval.run_ppe_onnx as runner

    model = tmp_path / "models/fixture.onnx"
    model.parent.mkdir()
    model.write_bytes(b"fixture; inference is mocked")
    gt = {name: {"test/example.jpg": np.array([[0, 0, 10, 10]])}
          for name in ("person", "helmet", "vest")}
    monkeypatch.setattr(runner, "dataset_inputs", lambda _: ([Path("example.jpg")], gt, [], "fixture-fingerprint"))
    def fake_inspect(path, names, input_size):
        assert input_size == (size, size)
        return {"classes": names, "grid_count": grid}

    monkeypatch.setattr(runner, "inspect_detector_onnx", fake_inspect)
    import yaml
    config = tmp_path / "models.yaml"
    config.write_text(yaml.safe_dump({"models": [{"model_id": "det", "onnx_path": "fixture.onnx", "input_size": [size, size]}]}))

    class FakeCpu:
        def __init__(self, models_dir, intra_op_threads):
            assert intra_op_threads == 2

        def load(self, spec):
            assert spec.model_id == "det"
            assert spec.input_size == (size, size)

        def class_names(self, model_id):
            return ["person"]

        def stats(self):
            return {"fixture": True}

        def close(self):
            pass

    monkeypatch.setattr(cpu, "OnnxCpuBackend", FakeCpu)

    def fake_infer(detector, images, data, output, *, include_smoke):
        assert isinstance(detector, Detector)
        assert detector.spec.classes == ["person"]
        assert detector.spec.input_size == (size, size)
        assert detector.spec.conf_threshold == 0.001 and detector.spec.iou_threshold == 0.7
        assert include_smoke is False
        predictions = {"person": [("test/example.jpg", .9, np.array([0, 0, 10, 10]))]}
        rows = [{"image_key": "test/example.jpg", "total_ms": 1.0}]
        return predictions, rows, [{"error": "fixture failure"}] if failed else []

    monkeypatch.setattr(runner, "infer_images", fake_infer)
    code, output = runner.run(tmp_path / "data", model.parent, config, tmp_path / "out", "det")
    report = json.loads((output / "report.json").read_text())
    assert report["provenance"]["data_sha256"] == "fixture-fingerprint"
    assert report["provenance"]["class_names_source"] == "onnx_metadata_via_backend"
    assert report["recipe"]["input_size"] == [size, size] and report["recipe"]["grid_count"] == grid
    assert report["recipe"]["resolution_experiment"] == (size == 1280)
    if failed:
        assert code == 1 and report["status"] == "failed" and report["metrics"] is None
    else:
        assert code == 0 and report["status"] == "complete"
        assert report["metrics"]["core_ap50(person,helmet,vest)"] == pytest.approx(1 / 3, abs=.0005)
        assert report["metrics"]["coverage"]["per_class"]["helmet"]["status"] == "unsupported"
        assert json.loads((output / "semantic-smoke.json").read_text())["status"] == "not_applicable"


def test_vyra_profile_preserves_pinned_order_and_legacy_default():
    buildvision, vyra = get_profile(), get_profile("vyra")
    assert buildvision.name == "buildvision" and buildvision.classes == CLASSES
    spec = load_specs(ROOT / "configs/models.ppe-vyra.yaml")["ppe"]
    assert spec.classes == vyra.classes and len(vyra.classes) == 14
    assert [vyra.classes[index] for index in (3, 8, 10, 11, 13)] == ["helmet", "no_helmet", "no_vest", "person", "vest"]
    assert vyra.classes[5] == "Mask" and vyra.classes[9] == "NO-Mask"
    assert all("respirator" not in name for name in vyra.classes)
    assert vyra.model_name == "best.onnx" and vyra.files[vyra.model_name] == (
        103_655_282, "b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246")
    assert vyra.directory != buildvision.directory
    with pytest.raises(ValueError, match="Unknown candidate profile"):
        get_profile("unreviewed")


def test_vyra_negative_classes_decode_in_original_order_and_masks_remain_unknown():
    result = semantic_smoke(_production_detections(["NO-Hardhat", "NO-Safety Vest", "Mask", "NO-Mask"], "vyra"),
                            np.zeros((640, 640, 3), np.uint8), 0)
    assert result["canonical_ppe_classes"] == ["no_helmet", "no_vest"]
    assert result["people"][0]["attributes"] == {"helmet": None, "vest": None, "respirator": None}
    assert result["violations"] == []


def test_still_image_smoke_does_not_fabricate_temporal_continuity():
    detections = _production_detections(["no-hardhat", "no-safety vest"])
    for frame_index in range(3):
        result = semantic_smoke(detections, np.zeros((640, 640, 3), np.uint8), frame_index)
        assert result["observation_scope"] == "single_image_fresh_tracks"
        assert result["temporal_event_evaluation"] is False
        assert result["people"][0]["attributes"] == {"helmet": None, "vest": None, "respirator": None}
        assert result["violations"] == []


@pytest.mark.parametrize("profile_name,source_classes", [
    ("buildvision", ["no-hardhat", "no-safety vest", "no-mask"]),
    ("vyra", ["NO-Hardhat", "NO-Safety Vest", "Mask", "NO-Mask"]),
])
def test_negative_sequence_confirms_attributes_then_duration_confirmed_s2_event(profile_name, source_classes):
    """Synthetic decoder/association/rule/event wiring; no model-quality claim."""
    from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Track
    from guard.events.manager import EventManager
    from guard.rules.plugins.ppe_missing import PpeMissingRule
    from guard.tracking.attributes import assign_attributes

    detections = _production_detections(source_classes, profile_name)
    person = next(det for det in detections if det.cls == "person")
    track = Track(track_id=1, cls="person", bbox=person.bbox, conf=person.conf, first_seen=0., last_seen=0.)
    cfg = RuleConfig(rule_id="s2-temporal-test", type="ppe_missing")
    assert cfg.min_duration_s == 2.0  # retain the real event-confirmation guard
    camera = CameraConfig(camera_id="temporal-ppe", name="PPE test", site="synthetic",
                          source=SourceConfig(type="images", uri="synthetic"), rules=[cfg])
    rule, manager = PpeMissingRule(cfg, camera), EventManager()
    created = []
    for index in range(13):  # distinct 5 FPS observations, then default two-second confirmation
        ts = index / 5
        track.last_seen = ts
        assign_attributes([track], {"ppe": detections}, ts)
        assert track.attributes["respirator"] is None
        violations = rule.evaluate(FrameContext(camera=camera, ts=ts, frame_idx=index,
                                   frame=np.zeros((640, 640, 3), np.uint8), people=[track]))
        if index < 2:
            assert track.attributes["helmet"] is None and track.attributes["vest"] is None
            assert violations == []
        else:
            assert track.attributes["helmet"] is False and track.attributes["vest"] is False
            assert violations[0].details["missing"] == ["helmet", "vest"]
        new_events, _ = manager.process(camera, ts, violations)
        if index < 12:
            assert new_events == []
        created.extend(new_events)
    assert len(created) == 1
    assert created[0].type == "ppe_missing" and created[0].track_id == 1
    assert created[0].started_at == pytest.approx(0.4)
    assert created[0].details["missing"] == ["helmet", "vest"]


@pytest.fixture
def vyra_onnx_fixture(tmp_path, monkeypatch):
    """Tiny local numerical graph; no ORT or remote code/session is involved."""
    import dataclasses
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import tools.prepare_ppe_candidate as prepare

    profile = get_profile("vyra")

    def make_model(*, names=None, channels=18, license_="AGPL-3.0 License (https://ultralytics.com/license)"):
        value = numpy_helper.from_array(np.zeros((1, channels, 8400), np.float32), name="constant")
        graph = helper.make_graph([helper.make_node("Identity", ["constant"], ["output0"])], "local-test",
                                  [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                                  [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, channels, 8400])], [value])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        helper.set_model_props(model, {"names": repr(dict(enumerate(names or profile.source_classes))), "license": license_})
        path = tmp_path / "fixture.onnx"
        onnx.save(model, path)
        files = {**profile.files, "best.onnx": (path.stat().st_size, prepare.sha256(path))}
        monkeypatch.setitem(prepare.PROFILES, "vyra", dataclasses.replace(profile, files=files))
        return path

    return make_model


def test_vyra_onnx_contract_validates_exact_class_order_and_shape(vyra_onnx_fixture):
    from tools.prepare_ppe_candidate import inspect_onnx

    good = inspect_onnx(vyra_onnx_fixture(), candidate_profile="vyra")
    assert good["checker"] == "passed" and good["classes"][8] == "no_helmet"
    wrong_order = get_profile("vyra").source_classes.copy()
    wrong_order[3], wrong_order[8] = wrong_order[8], wrong_order[3]
    with pytest.raises(ValueError, match="class-order mismatch"):
        inspect_onnx(vyra_onnx_fixture(names=wrong_order), candidate_profile="vyra")
    with pytest.raises(ValueError, match="Output/class-count mismatch"):
        inspect_onnx(vyra_onnx_fixture(channels=17), candidate_profile="vyra")
    with pytest.raises(ValueError, match="license metadata"):
        inspect_onnx(vyra_onnx_fixture(license_="unexpected"), candidate_profile="vyra")


def test_prepare_vyra_records_conflicting_licenses_without_ort_or_touching_buildvision(tmp_path, monkeypatch, vyra_onnx_fixture):
    import dataclasses
    import hashlib
    import tools.prepare_ppe_candidate as prepare

    onnx_path = vyra_onnx_fixture()
    profile = get_profile("vyra")
    payloads = {"best.onnx": onnx_path.read_bytes(), "README.md": b"license: cc-by-4.0\n", "args.yaml": b"model: yolov8m.pt\n"}
    files = {name: (len(payload), hashlib.sha256(payload).hexdigest()) for name, payload in payloads.items()}
    profile = dataclasses.replace(profile, files=files)
    monkeypatch.setitem(prepare.PROFILES, "vyra", profile)
    hub = {"id": profile.repository, "sha": profile.revision, "cardData": {"license": profile.hub_license},
           "siblings": [{"rfilename": "best.onnx", "size": files["best.onnx"][0],
                         "lfs": {"sha256": files["best.onnx"][1]}}]}
    requested = []

    def fixture_download(url, destination, **kwargs):
        requested.append(url)
        content = json.dumps(hub).encode() if "/api/models/" in url else payloads[url.rsplit("/", 1)[1]]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return {"url": url, **prepare.asset(destination)}

    monkeypatch.setattr(prepare, "download", fixture_download)
    # If preparation accidentally asks for a backend session, fail immediately.
    import guard.inference.onnx_cpu as cpu
    monkeypatch.setattr(cpu, "OnnxCpuBackend", lambda **_: pytest.fail("ORT session must not be created"))
    original = tmp_path / "models/candidates/buildvision-hazards-v0.3/ppe.json"
    original.parent.mkdir(parents=True)
    original.write_text("original buildvision sidecar")
    report = prepare.prepare_model(tmp_path / "models", candidate_profile="vyra")
    assert report["numerical_smoke"]["status"] == "not_run"
    assert report["license"] is None
    assert report["license_provenance"]["hub"] == "cc-by-4.0"
    assert "AGPL-3.0" in report["license_provenance"]["onnx_metadata"]
    assert report["license_provenance"]["review_required"] is True
    sidecar = json.loads((tmp_path / "models/candidates/vyra-yolo-ppe-detection/ppe.json").read_text())
    assert sidecar["classes"] == profile.classes and sidecar["candidate_profile"] == "vyra"
    assert original.read_text() == "original buildvision sidecar"
    assert all(not url.endswith((".pt", ".py")) for url in requested)


@pytest.mark.parametrize("profile_name", ["buildvision", "vyra"])
def test_runner_cli_resolves_profile_specific_default_registry(tmp_path, monkeypatch, profile_name):
    import ml.eval.run_ppe_onnx as runner

    calls = []
    monkeypatch.setattr(runner, "run", lambda *args: (calls.append(args) or (0, tmp_path)))
    args = ["--data", str(tmp_path)] + (["--candidate-profile", "vyra"] if profile_name == "vyra" else [])
    assert runner.main(args) == 0
    assert calls[0][2] == ROOT / "configs" / get_profile(profile_name).registry
    assert calls[0][-1] == profile_name


def test_runner_rejects_mismatched_profile_before_any_session(tmp_path, monkeypatch):
    import guard.inference.onnx_cpu as cpu
    import ml.eval.run_ppe_onnx as runner

    monkeypatch.setattr(cpu, "OnnxCpuBackend", lambda **_: pytest.fail("No backend session on profile mismatch"))
    with pytest.raises(ValueError, match="canonical classes"):
        runner.run(tmp_path, tmp_path, ROOT / "configs/models.ppe-vyra.yaml", tmp_path / "out")
    report_path = next((tmp_path / "out").glob("*/report.json"))
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["metrics"] is None


def test_high_resolution_recipe_is_explicit_and_does_not_mutate_reference():
    import copy
    from guard.contracts import ModelSpec
    from ml.eval.run_ppe_onnx import RECIPE, effective_recipe

    original = copy.deepcopy(RECIPE)
    low = effective_recipe(ModelSpec(model_id="det", onnx_path="low.onnx", input_size=(640, 640)))
    high = effective_recipe(ModelSpec(model_id="det", onnx_path="high.onnx", input_size=(1280, 1280)))
    assert low["grid_count"] == 8400 and low["overrides_from_640"] == {}
    assert high["grid_count"] == 33600 and high["resolution_experiment"] is True
    assert high["overrides_from_640"]["input_size"] == {"reference": [640, 640], "effective": [1280, 1280]}
    for name in ("expected_images", "conf_ap", "conf_op", "nms_iou", "threads", "ap", "data_sha256"):
        assert high[name] == low[name] == RECIPE[name]
    high["secondary_iou"]["no_helmet"] = 0.99
    assert RECIPE == original
    spec = load_specs(ROOT / "configs/models.yolov8x-1280.yaml")["det"]
    assert spec.input_size == (1280, 1280) and spec.onnx_path == "candidates/yolov8x-1280/yolov8x-1280.onnx"
    assert spec.classes == []  # production metadata fallback is deliberate
    assert spec.conf_threshold == .25 and spec.iou_threshold == .5  # same registry defaults as x640 control


@pytest.mark.parametrize("size", [(960, 960), (640, 1280), (0, 0)])
def test_resolution_recipe_rejects_unreviewed_shapes(size):
    from guard.contracts import ModelSpec
    from ml.eval.run_ppe_onnx import effective_recipe

    with pytest.raises(ValueError, match="640x640 or 1280x1280"):
        effective_recipe(ModelSpec(model_id="det", onnx_path="x.onnx", input_size=size))


@pytest.mark.parametrize("profile_name", ["buildvision", "vyra"])
def test_pinned_ppe_profile_cannot_use_high_resolution(tmp_path, monkeypatch, profile_name):
    import yaml
    import guard.inference.onnx_cpu as cpu
    import ml.eval.run_ppe_onnx as runner

    raw = yaml.safe_load((ROOT / "configs" / get_profile(profile_name).registry).read_text())
    next(item for item in raw["models"] if item["model_id"] == "ppe")["input_size"] = [1280, 1280]
    config = tmp_path / "ppe1280.yaml"
    config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(cpu, "OnnxCpuBackend", lambda **_: pytest.fail("PPE1280 must fail before session creation"))
    with pytest.raises(ValueError, match="PPE profiles require 640x640"):
        runner.run(tmp_path, tmp_path, config, tmp_path / "out", candidate_profile=profile_name)
    report = json.loads(next((tmp_path / "out").glob("*/report.json")).read_text())
    assert report["registry_input_size"] == [1280, 1280] and report["status"] == "failed"
    assert report["metrics"] is None


@pytest.fixture
def generic_detector_onnx(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def make(size, grid, *, metadata=None, input_dtype=TensorProto.FLOAT):
        output = numpy_helper.from_array(np.zeros((1, 5, grid), np.float32), name="constant")
        graph = helper.make_graph([helper.make_node("Identity", ["constant"], ["output0"])], "static-grid",
                                  [helper.make_tensor_value_info("images", input_dtype, [1, 3, size, size])],
                                  [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, grid])], [output])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
        helper.set_model_props(model, metadata or {})
        path = tmp_path / "static.onnx"
        onnx.save(model, path)
        return path

    return make


@pytest.mark.parametrize("size,grid", [(640, 8400), (1280, 33600)])
def test_generic_static_contract_checks_both_grids_and_metadata(generic_detector_onnx, size, grid):
    from ml.eval.run_ppe_onnx import inspect_detector_onnx

    metadata = {"imgsz": repr([size, size]), "names": repr({0: "person"}), "task": "detect"}
    result = inspect_detector_onnx(generic_detector_onnx(size, grid, metadata=metadata), ["person"], (size, size))
    assert result["input_shape"] == [1, 3, size, size] and result["output_shape"] == [1, 5, grid]
    assert result["grid_count"] == grid and result["metadata_checks"] == {"imgsz": "matched", "names": "matched"}
    fallback = inspect_detector_onnx(generic_detector_onnx(size, grid), ["person"], (size, size))
    assert fallback["metadata_checks"]["imgsz"].startswith("absent")


@pytest.mark.parametrize("size,wrong_grid", [(640, 33600), (1280, 8400)])
def test_generic_static_contract_rejects_crossed_grids(generic_detector_onnx, size, wrong_grid):
    from ml.eval.run_ppe_onnx import inspect_detector_onnx

    with pytest.raises(ValueError, match="interface mismatch"):
        inspect_detector_onnx(generic_detector_onnx(size, wrong_grid), ["person"], (size, size))


def test_generic_static_contract_rejects_wrong_input_metadata_or_dtype(generic_detector_onnx):
    import onnx
    from ml.eval.run_ppe_onnx import inspect_detector_onnx

    with pytest.raises(ValueError, match="interface mismatch"):
        inspect_detector_onnx(generic_detector_onnx(640, 8400), ["person"], (1280, 1280))
    with pytest.raises(ValueError, match="imgsz metadata mismatch"):
        inspect_detector_onnx(generic_detector_onnx(1280, 33600, metadata={"imgsz": "[640, 640]"}), ["person"], (1280, 1280))
    with pytest.raises(ValueError, match="names metadata mismatch"):
        inspect_detector_onnx(generic_detector_onnx(1280, 33600, metadata={"names": "{0: 'worker'}"}), ["person"], (1280, 1280))
    with pytest.raises(ValueError, match="float32"):
        inspect_detector_onnx(generic_detector_onnx(1280, 33600, input_dtype=onnx.TensorProto.FLOAT16), ["person"], (1280, 1280))


@pytest.mark.parametrize("size,grid", [(640, 8400), (1280, 33600)])
def test_per_image_validation_rejects_wrong_grid_and_nonfinite_outputs(size, grid):
    from guard.contracts import ModelSpec
    from ml.eval.run_ppe_onnx import effective_recipe, validate_raw_outputs

    recipe = effective_recipe(ModelSpec(model_id="det", onnx_path="x.onnx", input_size=(size, size)))
    good = np.zeros((1, 5, grid), np.float32)
    validate_raw_outputs([good], 1, recipe)
    with pytest.raises(ValueError, match="shape drift"):
        validate_raw_outputs([np.zeros((1, 5, 8400 if size == 1280 else 33600), np.float32)], 1, recipe)
    good[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        validate_raw_outputs([good], 1, recipe)
