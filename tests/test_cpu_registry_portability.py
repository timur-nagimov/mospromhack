"""Frozen CPU diagnostics stay independent of the production TPU registry."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CPU = ROOT / "configs/models.cpu-baseline.yaml"


def test_cpu_registry_preserves_premerge_bytes_and_production_tpu_metadata():
    from guard.pipeline.config import load_models

    assert hashlib.sha256(CPU.read_bytes()).hexdigest() == "d0c4c4e3d3ed9150b5d193643f016e6d61eb35f4798bdfa4bda98b05d23e704c"
    cpu = load_models(CPU)
    assert sorted(cpu) == ["det", "pose"] and all(spec.tpu_program is None for spec in cpu.values())
    assert cpu["det"].input_size == (640, 640) and cpu["det"].conf_threshold == .25
    assert cpu["pose"].output_format == "yolov8-pose"
    production = load_models(ROOT / "configs/models.yaml")
    assert all(production[name].tpu_program for name in ("det", "pose", "ppe", "fire"))
    assert len(production["det"].classes) == 80 and production["det"].classes[0] == "person"
    assert "helmet" in production["ppe"].classes and "no_helmet" in production["ppe"].classes
    raw = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    assert raw["meta"]["ppe"]["class_map"]["no-hardhat"] == "no_helmet"
    assert raw["meta"]["ppe"]["class_map"]["safety vest"] == "vest"
    recipe = json.loads((ROOT / "configs/eval-s1.json").read_text())
    assert ROOT / recipe["model_registry"] == CPU


def test_cpu_smoke_settings_ignore_dotenv_and_production_transport(tmp_path, monkeypatch):
    from guard.pipeline.settings import Settings
    from tools import cpu_smoke

    monkeypatch.setattr(Settings, "from_env", lambda: pytest.fail("CPU smoke must not read .env"))
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "shared-data"))
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path / "shared-models"))
    monkeypatch.setenv("GUARD_BACKEND", "linq-tpu")
    monkeypatch.setenv("GUARD_MODE", "edge")
    args = cpu_smoke.parser().parse_args([])
    settings = cpu_smoke._settings(args, tmp_path / "output")
    assert settings.models_config == CPU and settings.backend == "onnx-cpu" and settings.mode == "local"
    assert settings.data_dir == tmp_path / "shared-data" and settings.models_dir == tmp_path / "shared-models"
    assert settings.rabbit_url is None and settings.edge_id is None
    assert not (tmp_path / "shared-data").exists() and not (tmp_path / "output").exists()


def test_multicamera_generates_cpu_det_from_frozen_registry(tmp_path):
    from guard.pipeline.config import load_models
    from tools import cpu_multicamera

    settings = cpu_multicamera._settings(SimpleNamespace(data_dir=tmp_path / "data", models_dir=tmp_path / "models"), tmp_path, [])
    models = load_models(settings.models_config)
    assert sorted(models) == ["det"] and models["det"].tpu_program is None
    assert models["det"] == load_models(CPU)["det"]


def test_ml_output_defaults_follow_checkout_and_guard_data_dir(tmp_path, monkeypatch):
    from ml import build_calib, check_onnx, registry

    monkeypatch.delenv("GUARD_DATA_DIR", raising=False)
    assert check_onnx.default_output_dir() == ROOT / "data/media/ml_checks"
    assert build_calib.data_dir() == ROOT / "data"
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "shared"))
    monkeypatch.setenv("GUARD_MODELS_DIR", str(tmp_path / "models"))
    (tmp_path / "models").mkdir()
    classes = ["person", *[f"class-{index}" for index in range(79)]]
    (tmp_path / "models/det.json").write_text(json.dumps({"classes": classes}))
    check = check_onnx.parser().parse_args(["--model", "det", "--image", "not-opened.jpg"])
    calib = build_calib.parser().parse_args(["--work", str(tmp_path / "downloads")])
    assert Path(check.config) == CPU and Path(check.out) == tmp_path / "shared/media/ml_checks"
    assert Path(calib.out) == tmp_path / "shared/calib"
    specs = check_onnx.resolve_models(["det", "pose"], Path(check.config))
    assert [spec.model_id for spec in specs] == ["det", "pose"]
    assert specs[0].classes == classes  # Same sidecar class loading as the runtime.
    assert registry.onnx_file(specs[0]) == tmp_path / "models/yolov8n-640.onnx"
    assert not (tmp_path / "shared").exists()


@pytest.mark.parametrize("module", ["ml.check_onnx", "ml.build_calib", "ml.eval.run_s1", "ml.eval.compare_s1_models", "tools.cpu_smoke", "tools.cpu_multicamera"])
def test_cpu_ml_help_does_not_run_inference_decode_or_create_outputs(module, tmp_path):
    code = '''
import runpy, sys
import cv2
import onnxruntime
def forbidden(*args, **kwargs):
    raise AssertionError("help must not start a session, decode or write")
cv2.VideoCapture = cv2.VideoWriter = cv2.imwrite = forbidden
onnxruntime.InferenceSession = forbidden
sys.argv = [sys.argv[1], "--help"]
runpy.run_module(sys.argv[0], run_name="__main__")
'''
    data = tmp_path / "never-created"
    result = subprocess.run([sys.executable, "-c", code, module], capture_output=True, text=True, timeout=10,
                            env={**os.environ, "GUARD_DATA_DIR": str(data)})
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and not data.exists()
