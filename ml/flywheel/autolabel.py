"""Авторазметка сэмплов маховика учителем (docs/FLYWHEEL.md, звено 3). Torch — только через tools/heavy.sh:

    tools/heavy.sh python -m ml.flywheel.autolabel --limit 10 [--teachers coco,world,hardhat] [--status new]
        [--source event,uncertain] [--ids a,b] [--skip-holdout] [--device mps] [--imgsz 960] [--viz DIR]

Учителя — модели крупнее и независимые от моделей конвейера (иначе модель учится на своих ошибках):
- coco — YOLOv8x COCO (models/weights/teacher/yolov8x.pt): люди, техника, телефон, грузы;
- world — YOLO-World v2 x с текстовыми классами (models/weights/teacher_world/yolov8x-worldv2.pt): погрузчик,
  каска, жилет, сигарета, огонь, дым, спецтехника. Нужен пакет clip (ultralytics/CLIP). Без него учитель
  пропускается с предупреждением — автоустановку ultralytics выключаем (YOLO_AUTOINSTALL=false);
- hardhat — keremberke/yolov8m-hard-hat-detection (models/weights/hardhat_keremberke_v8m.pt): каска и голова без каски.

Классы учителей переводятся в нормализованные классы проекта (helmet, no_helmet, vest, forklift …).
Рамки одного класса от разных учителей сливаются (NMS, IoU 0.6). Метки — origin = teacher, conf учителя;
статус new → auto, в meta.autolabel — учителя и время. Метки, поправленные человеком (origin = operator),
и проверенные сэмплы (accepted, rejected) не трогаются, если не указан --overwrite.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # не ставить пакеты из-под скрипта
os.environ.setdefault("YOLO_OFFLINE", "true")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from guard.flywheel.select import box_iou  # noqa: E402
from guard.flywheel.store import FlywheelStore  # noqa: E402
from ml.flywheel.export import default_flywheel_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COCO_KEEP = {
    "person": "person", "bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "bus": "bus",
    "truck": "truck", "train": "train", "backpack": "backpack", "handbag": "handbag", "suitcase": "suitcase",
    "cell phone": "cell phone",
}
# текстовый класс YOLO-World → нормализованный класс проекта
WORLD_PROMPTS = {
    "forklift": "forklift",
    "hard hat": "helmet",
    "safety vest": "vest",
    "cigarette": "cigarette",
    "fire": "fire",
    "smoke": "smoke",
    "excavator": "excavator",
    "wheel loader": "wheel_loader",
    "dump truck": "dump_truck",
}
HARDHAT_MAP = {"hardhat": "helmet", "helmet": "helmet", "hard-hat": "helmet", "no-hardhat": "no_helmet",
               "no_hardhat": "no_helmet", "no-helmet": "no_helmet", "head": "no_helmet"}
TEACHERS = {
    "coco": {"weights": "models/weights/teacher/yolov8x.pt", "conf": 0.35, "map": COCO_KEEP},
    "world": {"weights": "models/weights/teacher_world/yolov8x-worldv2.pt", "conf": 0.2, "map": WORLD_PROMPTS},
    "hardhat": {"weights": "models/weights/hardhat_keremberke_v8m.pt", "conf": 0.4, "map": HARDHAT_MAP},
}
MERGE_IOU = 0.6


def weights_path(teacher: str) -> Path:
    rel = TEACHERS[teacher]["weights"]
    env = os.environ.get("GUARD_MODELS_DIR")
    if env:  # GUARD_MODELS_DIR указывает на каталог models
        return Path(env).expanduser() / Path(rel).relative_to("models")
    return ROOT / rel


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def teacher_available(teacher: str) -> tuple[bool, str]:
    path = weights_path(teacher)
    if not path.is_file():
        return False, f"нет весов {path}"
    if teacher == "world" and importlib.util.find_spec("clip") is None:
        return False, ("нет пакета clip для текстовых классов YOLO-World; установка: uv pip install --python "
                       ".venv/bin/python ftfy regex && uv pip install --python .venv/bin/python --no-deps "
                       "git+https://github.com/ultralytics/CLIP.git")
    return True, ""


def load_teacher(teacher: str) -> Any:
    from ultralytics import YOLO, YOLOWorld

    path = str(weights_path(teacher))
    if teacher == "world":
        model = YOLOWorld(path)
        model.set_classes(list(WORLD_PROMPTS))
        return model
    return YOLO(path)


def map_class(teacher: str, name: str) -> str | None:
    mapping = TEACHERS[teacher]["map"]
    key = str(name).strip()
    if teacher == "hardhat":
        key = key.lower()
    return mapping.get(key)


def run_teacher(teacher: str, model: Any, image: np.ndarray, *, conf: float, imgsz: int, device: str) -> list[dict]:
    # agnostic NMS: одна рамка на объект (иначе фура получает и truck, и bus, а голова — helmet и no_helmet)
    result = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False, agnostic_nms=True)[0]
    names = result.names
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    h, w = image.shape[:2]
    out = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls_ids, strict=False):
        cls = map_class(teacher, names[int(k)])
        if cls is None:
            continue
        x1, x2 = float(max(0.0, min(w, x1))), float(max(0.0, min(w, x2)))
        y1, y2 = float(max(0.0, min(h, y1))), float(max(0.0, min(h, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"cls": cls, "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "origin": "teacher", "conf": round(float(c), 4), "teacher": teacher})
    return out


def merge_labels(labels: list[dict]) -> list[dict]:
    """NMS по классу между учителями: из пересекающихся (IoU ≥ 0.6) рамок одного класса остаётся уверенная."""
    kept: list[dict] = []
    for lb in sorted(labels, key=lambda x: -(x.get("conf") or 0.0)):
        if any(k["cls"] == lb["cls"] and box_iou(k["bbox"], lb["bbox"]) >= MERGE_IOU for k in kept):
            continue
        kept.append(lb)
    return kept


def select_samples(store: FlywheelStore, args: argparse.Namespace) -> list[dict]:
    if args.ids:
        items, _ = store.query(ids=[i for i in args.ids.split(",") if i], limit=None)
    else:
        items, _ = store.query(status=args.status, source=args.source, camera_id=args.camera,
                               holdout=False if args.skip_holdout else None, limit=args.limit)
    out = []
    for s in items:
        if not args.overwrite:
            if s["status"] in ("accepted", "rejected"):
                continue
            if any(lb.get("origin") == "operator" for lb in s["labels"]):
                continue
        out.append(s)
    return out[: args.limit] if args.limit else out


def draw(image: np.ndarray, labels: list[dict], detections: list[dict]) -> np.ndarray:
    img = image.copy()
    for d in detections:  # детекции конвейера — тонкие серые
        x1, y1, x2, y2 = (int(v) for v in d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
    colors = {"coco": (0, 200, 0), "world": (0, 140, 255), "hardhat": (255, 0, 200)}
    for lb in labels:
        x1, y1, x2, y2 = (int(v) for v in lb["bbox"])
        color = colors.get(lb.get("teacher", ""), (0, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{lb['cls']} {lb.get('conf') or 0:.2f}"
        cv2.putText(img, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def autolabel(store: FlywheelStore, args: argparse.Namespace) -> dict[str, Any]:
    samples = select_samples(store, args)
    report: dict[str, Any] = {"selected": len(samples), "teachers": {}, "labels_by_class": {}, "labeled": 0}
    if not samples:
        return report
    images: dict[str, np.ndarray] = {}
    for s in samples:
        path = store.resolve_frame(s["frame_path"])
        img = cv2.imread(str(path)) if path is not None else None
        if img is not None:
            images[s["id"]] = img
    proposals: dict[str, list[dict]] = {sid: [] for sid in images}
    device = pick_device(args.device)
    used = []
    for teacher in [t.strip() for t in args.teachers.split(",") if t.strip()]:
        if teacher not in TEACHERS:
            report["teachers"][teacher] = {"skipped": "неизвестный учитель"}
            continue
        ok, why = teacher_available(teacher)
        if not ok:
            report["teachers"][teacher] = {"skipped": why}
            print(f"autolabel: учитель {teacher} пропущен — {why}", file=sys.stderr)
            continue
        t0 = time.time()
        model = load_teacher(teacher)
        conf = args.conf if args.conf is not None else TEACHERS[teacher]["conf"]
        n_boxes = 0
        for sid, img in images.items():
            boxes = run_teacher(teacher, model, img, conf=conf, imgsz=args.imgsz, device=device)
            proposals[sid].extend(boxes)
            n_boxes += len(boxes)
        report["teachers"][teacher] = {"boxes": n_boxes, "seconds": round(time.time() - t0, 1), "conf": conf,
                                       "weights": str(weights_path(teacher))}
        used.append(teacher)
        del model
        gc.collect()
        try:
            import torch

            if device == "mps":
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    if not used:
        report["error"] = "ни один учитель недоступен"
        return report
    counts: Counter[str] = Counter()
    viz_dir = Path(args.viz) if args.viz else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        if s["id"] not in proposals:
            continue
        labels = merge_labels(proposals[s["id"]])
        counts.update(lb["cls"] for lb in labels)
        meta = {**s["meta"], "autolabel": {"teachers": used, "at": round(time.time(), 1), "device": device,
                                           "imgsz": args.imgsz, "boxes": len(labels)}}
        store.update(s["id"], labels=[{k: v for k, v in lb.items() if k != "teacher"} for lb in labels],
                     status="auto", meta=meta)
        report["labeled"] += 1
        if viz_dir is not None:
            cv2.imwrite(str(viz_dir / f"{s['source']}_{s['camera_id']}_{s['id']}.jpg"),
                        draw(images[s["id"]], labels, s["detections"]))
    report["labels_by_class"] = dict(counts)
    report["device"] = device
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m ml.flywheel.autolabel", description="Авторазметка сэмплов учителем")
    p.add_argument("--flywheel-dir", type=Path, default=None, help="хранилище (GUARD_FLYWHEEL_DIR или data/flywheel)")
    p.add_argument("--teachers", default="coco,world,hardhat", help="через запятую: coco, world, hardhat")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--status", default="new", help="какие сэмплы размечать (через запятую)")
    p.add_argument("--source", default=None, help="источники через запятую")
    p.add_argument("--camera", default=None)
    p.add_argument("--ids", default=None, help="id сэмплов через запятую")
    p.add_argument("--skip-holdout", action="store_true", help="не размечать кадры отложенных роликов")
    p.add_argument("--overwrite", action="store_true", help="переразметить и проверенные человеком")
    p.add_argument("--device", default=None, help="mps | cpu (по умолчанию mps, если есть)")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=None, help="порог учителей (по умолчанию свой у каждого)")
    p.add_argument("--viz", default=None, help="каталог для кадров с нарисованными метками")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = FlywheelStore(args.flywheel_dir or default_flywheel_dir())
    try:
        report = autolabel(store, args)
    finally:
        store.close()
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if report.get("labeled") or not report.get("selected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
