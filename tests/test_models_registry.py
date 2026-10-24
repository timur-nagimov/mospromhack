"""configs/models.yaml: каждая запись — валидный ModelSpec, обязательные модели на месте, классы СИЗ нормализованы."""

from pathlib import Path

import numpy as np
import yaml

from guard.contracts import ModelSpec
from ml.registry import load_specs
from ml.yolo_decode import decode, letterbox

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def test_every_entry_validates():
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for item in raw["models"]:
        ModelSpec.model_validate(item)


def test_required_models_and_ppe_classes():
    specs = load_specs(CONFIG)
    assert {"det", "pose", "ppe", "fire"} <= set(specs)
    assert specs["pose"].output_format == "yolov8-pose"
    assert {"person", "helmet", "no_helmet", "vest", "no_vest"} <= set(specs["ppe"].classes)
    assert set(specs["fire"].classes) == {"fire", "smoke"}
    for s in specs.values():
        assert not Path(s.onnx_path).is_absolute()


def test_decode_synthetic_box():
    spec = load_specs(CONFIG)["ppe"]
    img = np.zeros((360, 640, 3), np.uint8)
    x, meta = letterbox(img, spec.input_size)
    assert x.shape == (1, 3, 640, 640)
    nc = len(spec.classes)
    out = np.zeros((1, 4 + nc, 10), np.float32)
    out[0, :4, 3] = [320, 320, 100, 50]  # центр кадра в координатах входа
    out[0, 4 + spec.classes.index("no_helmet"), 3] = 0.9
    dets = decode([out], meta, spec)
    assert len(dets) == 1 and dets[0].cls == "no_helmet"
    x1, y1, x2, y2 = dets[0].bbox
    assert abs((x1 + x2) / 2 - 320) < 1 and abs((y1 + y2) / 2 - 180) < 1
