"""Path/CLI contracts and fail-visible export plumbing; no pretrained model load."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ml import export_onnx, registry


def test_defaults_use_current_checkout_even_when_models_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("GUARD_MODELS_DIR", raising=False)
    monkeypatch.setattr(export_onnx, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "REPO_ROOT", tmp_path)
    assert export_onnx.models_dir_default() == tmp_path / "models"
    assert registry.models_dir() == tmp_path / "models"
    assert not (tmp_path / "models").exists()


def test_env_paths_expand_home_and_are_read_at_call_time(monkeypatch, tmp_path):
    monkeypatch.setenv("GUARD_MODELS_DIR", "~/shared-models")
    assert export_onnx.models_dir_default() == Path.home() / "shared-models"
    assert registry.models_dir() == Path.home() / "shared-models"
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path))
    assert export_onnx.models_dir_default() == tmp_path
    assert registry.models_dir() == tmp_path


def test_weights_use_existing_relative_file_before_official_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not deserialized")
    assert export_onnx.resolve_weights("best.pt", tmp_path / "models") == weights
    assert export_onnx.resolve_weights(weights, tmp_path / "models") == weights
    assert export_onnx.resolve_weights("yolov8n.pt", tmp_path / "models") == tmp_path / "models/weights/yolov8n.pt"
    with pytest.raises(FileNotFoundError):
        export_onnx.resolve_weights("unknown.pt", tmp_path / "models")


@pytest.mark.parametrize("model_id,filename", [("det", "yolov8n-640.onnx"), ("pose", "yolov8n-pose-640.onnx")])
def test_preset_can_export_just_one_model(monkeypatch, tmp_path, model_id, filename):
    calls = []

    def export(*args, **kwargs):
        calls.append((args, kwargs))
        return {"problems": []}

    monkeypatch.setattr(export_onnx, "export_one", export)
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path / "env"))
    explicit = tmp_path / "cli"
    assert export_onnx.main(["--preset", "coco", "--model-id", model_id, "--models-dir", str(explicit)]) == 0
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == model_id and args[3] == filename
    assert kwargs["models_dir"] == explicit and kwargs["threads"] == 2
    assert kwargs["source"].startswith(export_onnx.OFFICIAL_WEIGHTS_BASE)
    assert "AGPL-3.0" in kwargs["license_"]


def test_full_preset_stays_sequential_and_reports_inspection_failure(monkeypatch):
    calls = []

    def export(model_id, *args, **kwargs):
        calls.append(model_id)
        return {"problems": ["invalid shape"] if model_id == "pose" else []}

    monkeypatch.setattr(export_onnx, "export_one", export)
    assert export_onnx.main(["--preset", "coco"]) == 1
    assert calls == [item[0] for item in export_onnx.COCO_PRESET]


@pytest.mark.parametrize("args", [
    [], ["--preset", "coco", "--model-id", "missing"],
    ["--preset", "coco", "--weights", "best.pt"],
    ["--preset", "coco", "--out", "out.onnx"],
    ["--preset", "coco", "--class-map", "{}"],
    ["--preset", "coco", "--imgsz", "480"],
    ["--preset", "coco", "--task", "pose"],
    ["--preset", "coco", "--threads", "0"],
    ["--weights", "best.pt", "--out", "out.onnx", "--model-id", "x", "--imgsz", "-1"],
    ["--weights", "best.pt", "--out", "out.onnx", "--model-id", "x", "--imgsz", "641"],
    ["--weights", "best.pt", "--out", "out.onnx", "--model-id", "x", "--class-map", "["],
    ["--weights", "best.pt", "--out", "out.onnx", "--model-id", "x", "--class-map", "[]"],
    ["--weights", "best.pt", "--out", "out.onnx", "--model-id", "x", "--class-map", '{"x": 1}'],
])
def test_invalid_cli_does_not_load_models(monkeypatch, args):
    def forbidden(*args, **kwargs):
        raise AssertionError("export must not run for invalid CLI")
    monkeypatch.setattr(export_onnx, "export_one", forbidden)
    with pytest.raises(SystemExit) as error:
        export_onnx.main(args)
    assert error.value.code == 2


def test_custom_cli_propagates_class_map_and_environment(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(export_onnx, "export_one", lambda *args, **kwargs: calls.append((args, kwargs)) or {"problems": []})
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path))
    assert export_onnx.main(["--weights", "local.pt", "--out", "x.onnx", "--model-id", "x",
                             "--class-map", '{"worker": "person"}', "--threads", "1"]) == 0
    assert calls[0][0][-1] == {"worker": "person"}
    assert calls[0][1] == {"models_dir": tmp_path, "threads": 1}


@pytest.mark.parametrize("model_id,out", [("../escape", "x.onnx"), ("det", "../escape.onnx"), ("det", "/tmp/escape.onnx"), ("det", "x.pt")])
def test_export_rejects_unsafe_outputs_before_ml_imports(tmp_path, model_id, out):
    with pytest.raises(ValueError):
        export_onnx.export_one(model_id, "missing.pt", 640, out, "detect", models_dir=tmp_path)


def fake_export_modules(monkeypatch, tmp_path):
    calls = {}
    source = tmp_path / "source.onnx"
    torch = types.SimpleNamespace(
        set_num_threads=lambda n: calls.update(torch_threads=n),
        get_num_threads=lambda: calls["torch_threads"],
        get_num_interop_threads=lambda: calls.get("torch_interop_threads", 8),
        set_num_interop_threads=lambda n: calls.update(torch_interop_threads=n),
    )
    torch_utils = types.SimpleNamespace(NUM_THREADS=8)

    class Model:
        names = {0: "person", 1: "car"}

        def __init__(self, weights):
            calls["weights"] = weights

        def export(self, **kwargs):
            calls["export"] = kwargs
            source.write_bytes(b"raw onnx")
            return str(source)

    def slim(path):
        calls["slim"] = path
        return b"simplified onnx"

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=Model))
    monkeypatch.setitem(sys.modules, "ultralytics.utils", types.SimpleNamespace(torch_utils=torch_utils))
    monkeypatch.setitem(sys.modules, "onnx", types.SimpleNamespace(save=lambda model, path: Path(path).write_bytes(model)))
    monkeypatch.setitem(sys.modules, "onnxslim", types.SimpleNamespace(slim=slim))
    monkeypatch.setattr(export_onnx, "inspect_onnx", lambda path, threads: {"problems": [], "inputs": {}, "outputs": {}})
    monkeypatch.setattr(export_onnx.importlib.metadata, "version", lambda package: "test-version")
    return calls, torch_utils


def test_export_writes_verified_provenance_metadata_and_enforces_thread_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    calls, torch_utils = fake_export_modules(monkeypatch, tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"trusted test fixture")
    models = tmp_path / "models"
    meta = export_onnx.export_one("det", weights, 640, "det.onnx", "detect", source="official", license_="AGPL-3.0",
                                  models_dir=models, class_map={"car": "vehicle"})
    assert calls["torch_threads"] == calls["torch_interop_threads"] == torch_utils.NUM_THREADS == 2
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "2"
    assert calls["export"]["device"] == "cpu" and calls["export"]["batch"] == 1
    assert calls["export"]["simplify"] is False  # explicit slim below cannot silently fail
    assert calls["export"]["nms"] is False and calls["export"]["dynamic"] is False
    assert meta["classes"] == ["person", "vehicle"]
    assert meta["weights_sha256"] == export_onnx.sha256(weights)
    assert meta["onnx_sha256"] == export_onnx.sha256(models / "det.onnx")
    assert meta["simplified_by"] == "onnxslim" and meta["export_device"] == "cpu"
    assert json.loads((models / "det.json").read_text()) == meta


def test_simplifier_failure_is_visible_and_writes_no_success_metadata(monkeypatch, tmp_path):
    fake_export_modules(monkeypatch, tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"trusted test fixture")
    def fail(path):
        raise RuntimeError("incompatible simplifier")
    monkeypatch.setattr(sys.modules["onnxslim"], "slim", fail)
    models = tmp_path / "models"
    with pytest.raises(RuntimeError, match="incompatible simplifier"):
        export_onnx.export_one("det", weights, 640, "det.onnx", "detect", models_dir=models)
    assert not (models / "det.json").exists()


def test_help_and_registry_import_do_not_import_ml_runtime(tmp_path):
    code = '''
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"torch", "torchvision", "ultralytics", "onnxslim", "onnxruntime"}:
        raise AssertionError("heavy import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from ml import registry
sys.argv = ["ml.export_onnx", "--help"]
runpy.run_module("ml.export_onnx", run_name="__main__")
'''
    env = {**os.environ, "GUARD_MODELS_DIR": str(tmp_path / "absent"), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--models-dir" in result.stdout and "--model-id" in result.stdout
    assert not (tmp_path / "absent").exists()
