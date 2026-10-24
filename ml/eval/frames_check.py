"""Покадровая проверка детектора на роликах оценки: на каких кадрах модель видит класс и с какой уверенностью.

Ролики из ml/eval/scenarios.yaml — только для оценки, на них не обучаем. Скрипт ничего не пишет в набор:
он печатает по ролику и интервалам разметки долю кадров с классом и максимум conf, а с --sheet сохраняет
контрольные кадры с рамками.

Пример (погрузчик, s4_02 — тёмный проём):
    python ml/eval/frames_check.py --model forklift_v1=models/forklift_v1-480.onnx:person,forklift \\
        --model keremberke=models/forklift-640.onnx:forklift,person --cls forklift \\
        --videos s4_01,s4_02,s4_03,clean_01,clean_02 --fps 2 --out data/flywheel/runs/forklift-v1/frames.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_model(value: str) -> tuple[str, Path, list[str]]:
    name, rest = value.split("=", 1)
    path, classes = rest.rsplit(":", 1)
    return name, Path(path), [c.strip() for c in classes.split(",") if c.strip()]


def load_session(path: Path, classes: list[str], conf: float):
    import onnxruntime as ort

    from guard.contracts import ModelSpec

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    size = tuple(int(v) for v in inp.shape[2:4])
    spec = ModelSpec(model_id=path.stem, onnx_path=str(path), input_size=size, classes=classes,
                     conf_threshold=conf, iou_threshold=0.5)
    return sess, inp.name, size, spec


def label_intervals(labels_dir: Path, video: str, event_type: str) -> list[tuple[float, float, bool]]:
    path = labels_dir / f"{video}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(e["t_start"], e["t_end"], bool(e.get("uncertain"))) for e in data.get("events", [])
            if e.get("type") == event_type]


def check_video(video: Path, models: list[tuple[str, Any]], cls: str, fps: float, conf: float,
                sheet_dir: Path | None = None) -> list[dict[str, Any]]:
    from ml.yolo_decode import decode, letterbox

    cap = cv2.VideoCapture(str(video))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(native / fps))
    rows = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            t = idx / native
            row: dict[str, Any] = {"t": round(t, 2)}
            vis = frame.copy() if sheet_dir else None
            for k, (name, (sess, inp, size, spec)) in enumerate(models):
                x, meta = letterbox(frame, size)
                dets = [d for d in decode(sess.run(None, {inp: x}), meta, spec, conf=conf) if d.cls == cls]
                row[name] = round(max((d.conf for d in dets), default=0.0), 3)
                row[f"{name}_boxes"] = [[round(v, 1) for v in d.bbox] + [round(d.conf, 3)] for d in dets]
                if vis is not None:
                    color = [(0, 200, 255), (255, 120, 0), (0, 255, 0)][k % 3]
                    for d in dets:
                        x1, y1, x2, y2 = (int(v) for v in d.bbox)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, f"{name} {d.conf:.2f}", (x1, max(12, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if vis is not None:
                sheet_dir.mkdir(parents=True, exist_ok=True)
                cv2.putText(vis, f"{video.stem} t={t:.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(sheet_dir / f"{video.stem}_t{t:05.1f}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
            rows.append(row)
        idx += 1
    cap.release()
    return rows


def summarize(rows: list[dict[str, Any]], names: list[str], intervals, thr: float) -> dict[str, Any]:
    """Доля кадров с классом (conf ≥ thr) внутри интервалов разметки и вне их."""
    out: dict[str, Any] = {}
    for name in names:
        inside = [r for r in rows if any(a <= r["t"] <= b for a, b, _ in intervals)]
        outside = [r for r in rows if not any(a - 1.0 <= r["t"] <= b + 1.0 for a, b, _ in intervals)]
        per_event = []
        for a, b, unc in intervals:
            ev = [r for r in rows if a <= r["t"] <= b]
            per_event.append({"t": [a, b], "uncertain": unc, "frames": len(ev),
                              "hit": sum(r[name] >= thr for r in ev),
                              "max_conf": max((r[name] for r in ev), default=0.0)})
        out[name] = {
            "frames_in_events": len(inside),
            "hit_in_events": sum(r[name] >= thr for r in inside),
            "frames_outside": len(outside),
            "hit_outside": sum(r[name] >= thr for r in outside),
            "events": per_event,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="имя=путь.onnx:класс1,класс2")
    ap.add_argument("--cls", default="forklift")
    ap.add_argument("--event-type", default="vehicle_proximity", help="тип разметки для интервалов")
    ap.add_argument("--videos", required=True, help="ролики без расширения через запятую")
    ap.add_argument("--videos-dir", type=Path, default=ROOT / "data/videos")
    ap.add_argument("--labels-dir", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.1, help="порог декодера (что сохраняется в рамках)")
    ap.add_argument("--thr", type=float, default=0.45, help="порог «модель видит класс» (рождение трека)")
    ap.add_argument("--sheet", type=Path, help="каталог для кадров с рамками")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    models = []
    for value in args.model:
        name, path, classes = parse_model(value)
        path = path if path.is_absolute() else ROOT / path
        models.append((name, load_session(path, classes, args.conf)))
    names = [m[0] for m in models]
    report: dict[str, Any] = {"cls": args.cls, "fps": args.fps, "thr": args.thr, "models": args.model, "videos": {}}
    for video in args.videos.split(","):
        rows = check_video(args.videos_dir / f"{video}.mp4", models, args.cls, args.fps, args.conf,
                           args.sheet / video if args.sheet else None)
        intervals = label_intervals(args.labels_dir, video, args.event_type)
        summary = summarize(rows, names, intervals, args.thr)
        report["videos"][video] = {"summary": summary, "frames": rows}
        for name in names:
            s = summary[name]
            ev = "; ".join(f"{e['t'][0]:.1f}–{e['t'][1]:.1f} с: {e['hit']}/{e['frames']}, max {e['max_conf']:.2f}"
                           for e in s["events"])
            print(f"{video:9s} {name:12s} в событиях {s['hit_in_events']}/{s['frames_in_events']} кадров "
                  f"(≥{args.thr}), вне событий {s['hit_outside']}/{s['frames_outside']}  [{ev}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
