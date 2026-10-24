"""Прогон ONNX-моделей на onnxruntime-CPU по картинкам или видео: рамки, скелет, время инференса.

    python ml/check_onnx.py --model det --image data/images/samples/09_cctv_warehouse_workers_ppe.jpg
    python ml/check_onnx.py --model det pose --image data/images/samples/          # каталог
    python ml/check_onnx.py --model all --image data/images/samples/
    python ml/check_onnx.py --config configs/models.ppe-candidate.yaml --model ppe --video data/videos/x.mp4 --fps 5 --max-frames 100

--model — model_id из CPU-реестра configs/models.cpu-baseline.yaml или путь к .onnx
(тогда классы берутся из <stem>.json рядом). Другой реестр задаётся --config.
Печатает число детекций по классам и время: infer p50/p95 за --runs прогонов (по умолчанию 20),
плюс препроцессинг и декод. Визуализации — в data/media/ml_checks/<model_id>__<кадр>.jpg (или .mp4).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from guard.contracts import ModelSpec  # noqa: E402
from guard.pipeline.config import load_models  # noqa: E402
from ml.registry import models_dir, onnx_file  # noqa: E402
from ml.yolo_decode import decode, draw, letterbox  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_output_dir() -> Path:
    return Path(os.environ.get("GUARD_DATA_DIR") or ROOT / "data").expanduser() / "media" / "ml_checks"


def resolve_models(names: list[str], config: Path) -> list[ModelSpec]:
    specs = load_models(config, models_dir()) if config.exists() else {}
    if names == ["all"]:
        return list(specs.values())
    out = []
    for n in names:
        if n in specs:
            out.append(specs[n])
            continue
        p = Path(n)
        if p.suffix == ".onnx" and p.exists():
            meta_path = p.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
            shp = sess.get_inputs()[0].shape
            out.append(
                ModelSpec(
                    model_id=p.stem,
                    task=meta.get("task", "detect"),
                    onnx_path=str(p.resolve()),
                    input_size=(shp[2], shp[3]),
                    classes=meta.get("classes", []),
                    output_format=meta.get("output_format", "yolov8"),
                )
            )
            continue
        raise SystemExit(f"неизвестная модель: {n} (нет в {config} и не путь к .onnx)")
    return out


def make_session(spec: ModelSpec, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    path = onnx_file(spec)
    if not path.exists():
        raise SystemExit(f"[{spec.model_id}] нет файла {path}")
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    shp = sess.get_inputs()[0].shape
    if tuple(shp[2:]) != tuple(spec.input_size):
        raise SystemExit(f"[{spec.model_id}] input_size {spec.input_size} != ONNX {shp}")
    return sess


def pct(xs, q):
    return float(np.percentile(np.asarray(xs), q)) if xs else float("nan")


def run_image(spec, sess, img_path: Path, args) -> dict:
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"не читается {img_path}")
    inp = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    x, meta = letterbox(img, spec.input_size)
    t_pre = (time.perf_counter() - t0) * 1000
    outs = sess.run(None, {inp: x})  # прогрев
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        outs = sess.run(None, {inp: x})
        times.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    dets = decode(outs, meta, spec, conf=args.conf, iou=args.iou)
    t_post = (time.perf_counter() - t0) * 1000
    counts = collections.Counter(d.cls for d in dets)
    vis = draw(img, dets)
    cv2.putText(vis, f"{spec.model_id} {spec.input_size[1]}x{spec.input_size[0]} p50 {pct(times, 50):.1f} ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(vis, f"{spec.model_id} {spec.input_size[1]}x{spec.input_size[0]} p50 {pct(times, 50):.1f} ms",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    out = Path(args.out) / f"{spec.model_id}__{img_path.stem}.jpg"
    cv2.imwrite(str(out), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    res = {
        "model": spec.model_id,
        "image": img_path.name,
        "detections": len(dets),
        "by_class": dict(counts.most_common()),
        "infer_p50_ms": round(pct(times, 50), 2),
        "infer_p95_ms": round(pct(times, 95), 2),
        "pre_ms": round(t_pre, 2),
        "post_ms": round(t_post, 2),
        "vis": str(out),
    }
    print(
        f"[{spec.model_id}] {img_path.name}: {len(dets)} детекций {dict(counts.most_common())} | "
        f"infer p50 {res['infer_p50_ms']} ms p95 {res['infer_p95_ms']} ms ({args.runs} прогонов) | "
        f"pre {res['pre_ms']} ms post {res['post_ms']} ms → {out}"
    )
    if args.verbose:
        for d in sorted(dets, key=lambda d: -d.conf):
            print(f"    {d.cls:14s} {d.conf:.2f} [{', '.join(f'{v:.0f}' for v in d.bbox)}]")
    return res


def run_video(spec, sess, video: Path, args) -> dict:
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(src_fps / args.fps))
    out = Path(args.out) / f"{spec.model_id}__{video.stem}.mp4"
    writer = None
    inp = sess.get_inputs()[0].name
    times, counts, n = [], collections.Counter(), 0
    i = 0
    while n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step:
            i += 1
            continue
        i += 1
        x, meta = letterbox(frame, spec.input_size)
        t0 = time.perf_counter()
        outs = sess.run(None, {inp: x})
        times.append((time.perf_counter() - t0) * 1000)
        dets = decode(outs, meta, spec, conf=args.conf, iou=args.iou)
        counts.update(d.cls for d in dets)
        vis = draw(frame, dets)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(vis)
        n += 1
    if writer:
        writer.release()
    res = {"model": spec.model_id, "video": video.name, "frames": n, "by_class_total": dict(counts.most_common()),
           "infer_p50_ms": round(pct(times, 50), 2), "infer_p95_ms": round(pct(times, 95), 2), "vis": str(out)}
    print(f"[{spec.model_id}] {video.name}: {n} кадров, детекций по классам {dict(counts.most_common())} | "
          f"infer p50 {res['infer_p50_ms']} ms p95 {res['infer_p95_ms']} ms → {out}")
    return res


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, help="model_id из CPU-реестра/--config, путь к .onnx или all")
    ap.add_argument("--image", nargs="*", default=[], help="файлы или каталоги с картинками")
    ap.add_argument("--video", nargs="*", default=[])
    ap.add_argument("--config", default=str(ROOT / "configs" / "models.cpu-baseline.yaml"))
    ap.add_argument("--conf", type=float, default=None, help="порог уверенности (по умолчанию из ModelSpec)")
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--runs", type=int, default=20, help="прогонов для p50")
    ap.add_argument("--threads", type=int, default=0, help="intra_op потоков ORT (0 — по умолчанию)")
    ap.add_argument("--fps", type=float, default=5.0, help="частота выборки кадров видео")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--out", default=str(default_output_dir()))
    ap.add_argument("--json", help="сохранить сводку в JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="печатать каждую детекцию")
    return ap


def main(argv=None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    if not args.image and not args.video:
        ap.error("нужен --image или --video")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    images: list[Path] = []
    for s in args.image:
        p = Path(s)
        images += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]

    results = []
    for spec in resolve_models(args.model, Path(args.config)):
        sess = make_session(spec, args.threads)
        print(f"== {spec.model_id}: {onnx_file(spec).name} in={spec.input_size} classes={len(spec.classes)} "
              f"conf={args.conf or spec.conf_threshold} ORT {ort.__version__} CPU")
        for img in images:
            results.append(run_image(spec, sess, img, args))
        for v in args.video:
            results.append(run_video(spec, sess, Path(v), args))
        p50s = [r["infer_p50_ms"] for r in results if r["model"] == spec.model_id]
        if len(p50s) > 1:
            print(f"   итог {spec.model_id}: медиана p50 по кадрам {np.median(p50s):.1f} ms")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
