"""Экспорт YOLOv8 (.pt) в ONNX под LinQ TPU и onnxruntime-CPU.

Требования к ONNX (docs/MODELS.md): opset 12, статический shape с batch=1,
simplify (onnxslim), без NMS внутри — декод и NMS делаем сами.

Примеры:
    python ml/export_onnx.py --preset coco                # det и pose COCO
    python ml/export_onnx.py --weights /abs/best.pt --imgsz 640 --out ppe-640.onnx --model-id ppe

Рядом с ONNX пишется <model_id>.json: классы, вход, выходы, операторы, источник.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = Path("/Users/nagimov-timur/Desktop/mosprom")
MODELS_DIR = Path(os.environ.get("GUARD_MODELS_DIR", MAIN_ROOT / "models"))
WEIGHTS_DIR = MODELS_DIR / "weights"

OPSET = 12

# (model_id, weights, imgsz, onnx-файл, task)
COCO_PRESET = [
    ("det", "yolov8n.pt", 640, "yolov8n-640.onnx", "detect"),
    ("det_n480", "yolov8n.pt", 480, "yolov8n-480.onnx", "detect"),
    ("det_n416", "yolov8n.pt", 416, "yolov8n-416.onnx", "detect"),
    ("det_s640", "yolov8s.pt", 640, "yolov8s-640.onnx", "detect"),
    ("pose", "yolov8n-pose.pt", 640, "yolov8n-pose-640.onnx", "pose"),
    ("pose_n480", "yolov8n-pose.pt", 480, "yolov8n-pose-480.onnx", "pose"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_onnx(path: Path) -> dict:
    """Проверка требований и сводка по графу для риска компиляции под TPU."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    m = onnx.load(str(path))
    onnx.checker.check_model(m)
    opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
    ops = collections.Counter(n.op_type for n in m.graph.node)

    def dims(vi):
        return [d.dim_value if d.dim_value else (d.dim_param or "?") for d in vi.type.tensor_type.shape.dim]

    inputs = {i.name: dims(i) for i in m.graph.input}
    outputs = {o.name: dims(o) for o in m.graph.output}
    problems = []
    if opset.get("ai.onnx") != OPSET:
        problems.append(f"opset {opset.get('ai.onnx')} != {OPSET}")
    for name, shp in {**inputs, **outputs}.items():
        if any(not isinstance(d, int) for d in shp):
            problems.append(f"dynamic dim in {name}: {shp}")
    for shp in inputs.values():
        if shp and shp[0] != 1:
            problems.append(f"batch != 1: {shp}")
    if "NonMaxSuppression" in ops:
        problems.append("NonMaxSuppression inside graph")

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    x = np.zeros(inp.shape, dtype=np.float32)
    outs = sess.run(None, {inp.name: x})
    return {
        "opset": opset,
        "inputs": inputs,
        "outputs": outputs,
        "ort_output_shapes": [list(o.shape) for o in outs],
        "op_counts": dict(sorted(ops.items())),
        "num_nodes": len(m.graph.node),
        "problems": problems,
    }


def export_one(
    model_id: str,
    weights: str | Path,
    imgsz: int,
    out_name: str,
    task: str,
    source: str = "",
    license_: str = "",
    class_map: dict[str, str] | None = None,
) -> dict:
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    w = Path(weights)
    if not w.is_absolute():
        w = WEIGHTS_DIR / w  # официальные веса Ultralytics скачиваются сюда автоматически
    model = YOLO(str(w))
    names = [model.names[i] for i in range(len(model.names))]
    t0 = time.time()
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=OPSET,
        simplify=True,
        dynamic=False,
        batch=1,
        half=False,
        nms=False,
        device="cpu",
    )
    dst = MODELS_DIR / out_name
    shutil.move(str(exported), dst)
    info = inspect_onnx(dst)
    classes = [class_map.get(n, n) for n in names] if class_map else names
    meta = {
        "model_id": model_id,
        "task": task,
        "onnx": out_name,
        "input_size": [imgsz, imgsz],
        "input": "NCHW float32 RGB 0..1, letterbox 114",
        "output_format": "yolov8-pose" if task == "pose" else "yolov8",
        "classes": classes,
        "source_classes": names,
        "weights": str(w),
        "weights_sha256": sha256(w),
        "onnx_sha256": sha256(dst),
        "onnx_bytes": dst.stat().st_size,
        "source": source,
        "license": license_,
        "export_s": round(time.time() - t0, 1),
        **info,
    }
    (MODELS_DIR / f"{model_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    status = "OK" if not info["problems"] else f"PROBLEMS {info['problems']}"
    print(f"[{model_id}] {dst} {meta['onnx_bytes'] / 1e6:.1f} MB in={info['inputs']} out={info['outputs']} {status}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["coco"], help="экспорт стандартного набора COCO det и pose")
    ap.add_argument("--weights", help="путь к .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", help="имя ONNX-файла в каталоге моделей")
    ap.add_argument("--model-id")
    ap.add_argument("--task", default="detect", choices=["detect", "pose"])
    ap.add_argument("--source", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--class-map", help="JSON {исходный класс: нормализованный}")
    args = ap.parse_args()

    if args.preset == "coco":
        for model_id, weights, imgsz, out, task in COCO_PRESET:
            export_one(
                model_id, weights, imgsz, out, task,
                source="https://github.com/ultralytics/assets/releases (COCO)",
                license_="AGPL-3.0 (Ultralytics)",
            )
        return 0
    if not (args.weights and args.out and args.model_id):
        ap.error("нужны --preset или --weights/--out/--model-id")
    class_map = json.loads(args.class_map) if args.class_map else None
    export_one(args.model_id, args.weights, args.imgsz, args.out, args.task, args.source, args.license, class_map)
    return 0


if __name__ == "__main__":
    sys.exit(main())
