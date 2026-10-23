"""Сравнение детекторов СИЗ на размеченном наборе в формате YOLO (по умолчанию Ultralytics Construction-PPE).

Классы моделей и разметки приводятся к общим: person, helmet, no_helmet, vest, no_vest.
Метрика — AP50 (COCO, 101 точка) по каждому классу; для no_helmet дополнительно AP при IoU 0.3,
потому что в Construction-PPE рамка no_helmet нарисована по лицу, а модели рисуют голову.
Плюс precision/recall в рабочей точке --conf-op.

Пример:
    python ml/eval_ppe.py --data /tmp/a2_dl/cppe/extracted --splits val test \
        --model constructelligence=/tmp/a2_w/constructelligence/buildvision-hazards-v0.3.pt \
        --model det_s=/Users/.../models/weights/yolov8s.pt --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

NORM = ("person", "helmet", "no_helmet", "vest", "no_vest")


def normalize(name: str) -> str | None:
    """Имя класса модели → общий класс или None (не участвует в оценке)."""
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    neg = bool(re.match(r"^(no|not|without)\b", n))
    if re.search(r"hard ?hat|helmet", n):
        return "no_helmet" if neg else "helmet"
    if n == "head":
        return "no_helmet"
    if "vest" in n:
        return "no_vest" if neg else "vest"
    if n in ("person", "worker", "human", "people"):
        return "person"
    return None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def ap_for_class(preds, gts, thr: float, conf_op: float):
    """preds: list[(img, conf, box)], gts: dict[img -> np.ndarray(N,4)]. → (AP, P@op, R@op, n_gt)."""
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return None, None, None, 0
    preds = sorted(preds, key=lambda p: -p[1])
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(preds))
    for i, (img, _, box) in enumerate(preds):
        g = gts.get(img)
        if g is None or len(g) == 0:
            continue
        ious = iou_matrix(box[None], g)[0]
        ious[used[img]] = -1
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            used[img][j] = True
            tp[i] = 1
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    # COCO: огибающая precision и 101 точка по recall
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(x, mrec, mpre), x))
    confs = np.array([p[1] for p in preds])
    k = int((confs >= conf_op).sum())
    p_op = float(ctp[k - 1] / k) if k else 0.0
    r_op = float(ctp[k - 1] / n_gt) if k else 0.0
    return ap, p_op, r_op, n_gt


def load_gt(data: Path, splits, names: dict[int, str]):
    imgs, gts = [], defaultdict(lambda: defaultdict(list))
    for s in splits:
        for p in sorted((data / "images" / s).glob("*")):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            imgs.append(p)
            lbl = data / "labels" / s / (p.stem + ".txt")
            if not lbl.exists():
                continue
            import cv2

            h, w = cv2.imread(str(p)).shape[:2]
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = normalize(names[int(parts[0])])
                if c is None:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                gts[c][p.name].append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    gts = {c: {k: np.array(v) for k, v in d.items()} for c, d in gts.items()}
    return imgs, gts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="корень набора YOLO: images/<split>, labels/<split>, data.yaml")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--model", action="append", required=True, help="имя=путь к .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-op", type=float, default=0.35)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json")
    args = ap.parse_args()

    import yaml
    from ultralytics import YOLO

    data = Path(args.data)
    meta = yaml.safe_load((data / "data.yaml").read_text())
    names = meta["names"] if isinstance(meta["names"], dict) else dict(enumerate(meta["names"]))
    imgs, gts = load_gt(data, args.splits, names)
    print(f"images={len(imgs)} gt: " + ", ".join(f"{c}={sum(len(v) for v in gts.get(c, {}).values())}" for c in NORM))

    results = {}
    for spec in args.model:
        name, path = spec.split("=", 1)
        model = YOLO(path)
        cmap = {i: normalize(n) for i, n in model.names.items()}
        print(f"\n[{name}] classes: " + ", ".join(f"{n}->{cmap[i]}" for i, n in model.names.items() if cmap[i]))
        preds = defaultdict(list)
        t0 = time.time()
        for r in model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=0.001, iou=0.7, device=args.device,
                               stream=True, verbose=False, max_det=300):
            b = r.boxes
            for box, conf, cls in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)):
                c = cmap.get(int(cls))
                if c:
                    preds[c].append((Path(r.path).name, float(conf), box))
        dt = (time.time() - t0) / max(len(imgs), 1) * 1000
        row = {"ms_per_img": round(dt, 1)}
        for c in NORM:
            if c not in set(cmap.values()):
                row[c] = None
                continue
            a, p, rcl, n = ap_for_class(preds[c], gts.get(c, {}), 0.5, args.conf_op)
            row[c] = None if a is None else {"ap50": round(a, 3), "p": round(p, 3), "r": round(rcl, 3), "n_gt": n}
            if c == "no_helmet":
                a3, p3, r3, _ = ap_for_class(preds[c], gts.get(c, {}), 0.3, args.conf_op)
                if a3 is not None:
                    row["no_helmet@iou0.3"] = {"ap": round(a3, 3), "p": round(p3, 3), "r": round(r3, 3)}
        core = [row[c]["ap50"] for c in ("person", "helmet", "vest") if row.get(c)]
        row["core_ap50(person,helmet,vest)"] = round(float(np.mean(core)), 3) if core else None
        results[name] = row
        print(json.dumps(row, ensure_ascii=False))

    print("\nmodel | person | helmet | vest | no_helmet@0.5 | no_helmet@0.3 | ms/img")
    for name, row in results.items():
        def g(k, f="ap50"):
            v = row.get(k)
            return "-" if not v else f"{v[f]:.3f}"
        print(f"{name} | {g('person')} | {g('helmet')} | {g('vest')} | {g('no_helmet')} | "
              f"{g('no_helmet@iou0.3', 'ap')} | {row['ms_per_img']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
