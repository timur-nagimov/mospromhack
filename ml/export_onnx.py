"""Экспорт YOLOv8 (.pt) в ONNX под LinQ TPU и onnxruntime-CPU.

Требования к ONNX (docs/MODELS.md): opset 12, статический shape с batch=1,
simplify (onnxslim), без NMS внутри — декод и NMS делаем сами.

Примеры:
    python ml/export_onnx.py --preset coco --model-id det  # только det 640
    python ml/export_onnx.py --preset coco --model-id pose # только pose 640
    python ml/export_onnx.py --weights /abs/best.pt --imgsz 640 --out ppe-640.onnx --model-id ppe

Рядом с ONNX пишется <model_id>.json: классы, вход, выходы, операторы, источник.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

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

OFFICIAL_WEIGHTS_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"


def models_dir_default() -> Path:
    """Current checkout by default; worktrees can explicitly share GUARD_MODELS_DIR."""
    return Path(os.environ.get("GUARD_MODELS_DIR") or ROOT / "models").expanduser()


def resolve_weights(weights: str | Path, models_dir: Path) -> Path:
    """Existing paths are local; recognized preset names use the official cache."""
    path = Path(weights).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and str(path) in {item[1] for item in COCO_PRESET}:
        return models_dir / "weights" / path
    raise FileNotFoundError(f"weights file not found: {path}")


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def image_size(value: str) -> int:
    number = positive_int(value)
    if number % 32:
        raise argparse.ArgumentTypeError("imgsz должен быть кратен 32, чтобы Ultralytics не изменил размер молча")
    return number


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


VERSION_PACKAGES = ("ultralytics", "torch", "torchvision", "onnx", "onnxslim", "onnxruntime", "numpy",
                    "opencv-python", "opencv-python-headless")


def package_versions(packages: Sequence[str] = VERSION_PACKAGES) -> dict[str, str | None]:
    """Версии пакетов окружения экспорта. Не установленный пакет — None: в .venv проекта стоит
    только opencv-python-headless, и экспорт не должен падать на записи метаданных."""
    out: dict[str, str | None] = {}
    for package in packages:
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def inspect_onnx(path: Path, threads: int = 2) -> dict:
    """Проверка требований и сводка по графу для риска компиляции под TPU."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    ort.disable_telemetry_events()
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

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
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
    *,
    models_dir: str | Path | None = None,
    threads: int = 2,
) -> dict:
    directory = Path(models_dir).expanduser() if models_dir is not None else models_dir_default()
    if not model_id or Path(model_id).name != model_id or model_id in {".", ".."}:
        raise ValueError("model_id must be a nonempty filename component")
    if Path(out_name).name != out_name or Path(out_name).suffix.lower() != ".onnx":
        raise ValueError("out must be an ONNX filename inside models_dir")
    if imgsz <= 0 or imgsz % 32 or threads <= 0:
        raise ValueError("imgsz must be positive and divisible by 32; threads must be positive")
    w = resolve_weights(weights, directory)

    # Keep export sequential and CPU-only; importing this module or --help does
    # not import torch/Ultralytics and does not create any directories.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    import onnx
    import onnxslim
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils

    # select_device("cpu") in the pinned Ultralytics exporter resets the torch
    # thread count to its own constant; keep that reset within our CPU budget.
    torch_utils.NUM_THREADS = threads
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != threads:
        torch.set_num_interop_threads(threads)
    directory.mkdir(parents=True, exist_ok=True)
    w.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(w))
    names = [model.names[i] for i in range(len(model.names))]
    t0 = time.time()
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=OPSET,
        # Ultralytics catches simplifier failures and still reports success.
        # Run it explicitly below so incompatible deps fail visibly.
        simplify=False,
        dynamic=False,
        batch=1,
        half=False,
        nms=False,
        device="cpu",
    )
    onnx.save(onnxslim.slim(str(exported)), str(exported))
    dst = directory / out_name
    if Path(exported).resolve() != dst.resolve():
        shutil.move(str(exported), dst)
    info = inspect_onnx(dst, threads=threads)
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
        "export_device": "cpu",
        "export_threads": threads,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "simplified_by": "onnxslim",
        "versions": package_versions(),
        "export_s": round(time.time() - t0, 1),
        **info,
    }
    (directory / f"{model_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "OK" if not info["problems"] else f"PROBLEMS {info['problems']}"
    print(f"[{model_id}] {dst} {meta['onnx_bytes'] / 1e6:.1f} MB in={info['inputs']} out={info['outputs']} {status}")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=["coco"], help="набор COCO; --model-id выбирает одну модель, иначе все варианты последовательно")
    ap.add_argument("--weights", help="путь к .pt")
    ap.add_argument("--imgsz", type=image_size, help="размер для --weights (по умолчанию 640, кратный 32); у preset размер задан его model-id")
    ap.add_argument("--out", help="имя ONNX-файла в каталоге моделей")
    ap.add_argument("--model-id", help="ID модели; с --preset выбирает запись det, pose или другой ID пресета")
    ap.add_argument("--models-dir", type=Path, default=models_dir_default(), help="каталог ONNX/metadata/weights; выше приоритетом, чем GUARD_MODELS_DIR")
    ap.add_argument("--threads", type=positive_int, default=2, help="потоки torch и CPU-проверки ORT (по умолчанию 2)")
    ap.add_argument("--task", choices=["detect", "pose"], help="задача для --weights (по умолчанию detect)")
    ap.add_argument("--source", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--class-map", help="JSON {исходный класс: нормализованный}")
    args = ap.parse_args(argv)

    if args.preset == "coco":
        if args.weights or args.out or args.class_map or args.source or args.license or args.imgsz is not None or args.task:
            ap.error("--preset нельзя смешивать с --weights/--out/--class-map/--source/--license/--imgsz/--task")
        selected = [item for item in COCO_PRESET if not args.model_id or item[0] == args.model_id]
        if not selected:
            ap.error("неизвестный --model-id для COCO: " + ", ".join(item[0] for item in COCO_PRESET))
        failed = False
        for model_id, weights, imgsz, out, task in selected:
            meta = export_one(
                model_id, weights, imgsz, out, task,
                source=f"{OFFICIAL_WEIGHTS_BASE}/{weights}",
                license_="AGPL-3.0 (Ultralytics); https://github.com/ultralytics/ultralytics/blob/main/LICENSE",
                models_dir=args.models_dir, threads=args.threads,
            )
            failed |= bool(meta["problems"])
        return int(failed)
    if not (args.weights and args.out and args.model_id):
        ap.error("нужны --preset или --weights/--out/--model-id")
    try:
        class_map = json.loads(args.class_map) if args.class_map else None
    except json.JSONDecodeError:
        ap.error("--class-map должен быть JSON объектом строка: строка")
    if class_map is not None and (not isinstance(class_map, dict) or
            any(not isinstance(key, str) or not isinstance(value, str) for key, value in class_map.items())):
        ap.error("--class-map должен быть JSON объектом строка: строка")
    meta = export_one(args.model_id, args.weights, args.imgsz or 640, args.out, args.task or "detect",
                      args.source, args.license, class_map, models_dir=args.models_dir, threads=args.threads)
    return int(bool(meta["problems"]))


if __name__ == "__main__":
    sys.exit(main())
