#!/usr/bin/env python3
"""YOLOv8 на LinQ TPU против CPU: сверка рамок, задержки p50/p95, кадры/с при окне задач в полёте.

Запуск на Jupyter-стенде (брокер рядом), окружение ~/linq312 (onnxruntime, pika, protobuf, pydantic, pillow):
    cd ~/work/guard_repo && GUARD_MODELS_DIR=$HOME/work/models OMP_NUM_THREADS=2 \
      ~/linq312/bin/python tpu/yolo_check.py --model yolov8n_640 --onnx ~/work/models/yolov8n-640.onnx \
      --images ~/work/models/calib_frames --n-images 20 -n 50 --windows 1,4,8 --out ~/work/metrics/yolo_tpu.json

CPU-эталон — исходная ONNX в onnxruntime. TPU — LinqTpuBackend (голова на TPU + хвост *_tail.onnx на CPU),
выход в формате ONNX (output0 [1,84,N]). Декод: conf >= --conf, NMS по классам IoU 0.5 (numpy).
Совпадение: та же метка и IoU >= 0.5 (жадно по убыванию conf).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from compile import preprocess  # noqa: E402  (letterbox 114, RGB 0..1, NCHW)
from guard.contracts import ModelSpec  # noqa: E402
from guard.inference.linq_tpu import LinqTpuBackend  # noqa: E402


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


NC = 80  # число классов; для pose = 1, остальное — keypoints (17x3)


def decode(out0: np.ndarray, conf: float, iou_thr: float = 0.5):
    p = out0[0].T  # [N, 4+C(+K*3)]
    scores = p[:, 4:4 + NC]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    keep = sc >= conf
    xywh, cls, sc, extra = p[keep, :4], cls[keep], sc[keep], p[keep, 4 + NC:]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    order = np.argsort(-sc)
    boxes, cls, sc, extra = boxes[order], cls[order], sc[order], extra[order]
    sel = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        while len(idx):
            i = idx[0]
            sel.append(i)
            if len(idx) == 1:
                break
            ov = iou(boxes[i:i + 1], boxes[idx[1:]])[0]
            idx = idx[1:][ov < iou_thr]
    sel = sorted(sel, key=lambda i: -sc[i])
    return boxes[sel], cls[sel], sc[sel], extra[sel]


def match(ref, got, thr: float = 0.5) -> dict:
    rb, rc, rs, rk = ref
    gb, gc, gs, gk = got
    used = set()
    ious, dconf, dkpt = [], [], []
    for i in range(len(rb)):
        if len(gb) == 0:
            break
        ov = iou(rb[i:i + 1], gb)[0]
        ov[[j for j in range(len(gb)) if j in used or gc[j] != rc[i]]] = 0
        j = int(ov.argmax())
        if ov[j] >= thr:
            used.add(j)
            ious.append(float(ov[j]))
            dconf.append(float(gs[j]) - float(rs[i]))  # знаковый: TPU - CPU
            if rk.shape[1] >= 51:  # pose: ошибка xy видимых keypoint'ов, пиксели входа
                r3, g3 = rk[i, :51].reshape(17, 3), gk[j, :51].reshape(17, 3)
                vis = r3[:, 2] > 0.5
                if vis.any():
                    dkpt.append(float(np.linalg.norm(r3[vis, :2] - g3[vis, :2], axis=1).mean()))
    return {"ref": len(rb), "got": len(gb), "matched": len(ious), "ious": ious, "dconf": dconf, "dkpt": dkpt}


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolov8n_640", help="каталог models/tpu/<model> с manifest.json")
    ap.add_argument("--onnx", required=True, help="исходная ONNX (CPU-эталон)")
    ap.add_argument("--program", help="URI программы (по умолчанию file:///auto/cache/<login>/<имя из manifest>)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tpu-conf-delta", type=float, default=0.1, help="на сколько ниже порог для TPU во 2-м варианте")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--windows", default="1,4,8")
    ap.add_argument("--host", default=os.environ.get("LINQ_RABBIT_HOST", "10.0.114.30"))
    ap.add_argument("--out", default="yolo_tpu.json")
    ap.add_argument("--nc", type=int, default=80, help="число классов (pose: 1)")
    ap.add_argument("--sweep", default="0,0.05,0.1,0.15", help="снижения порога TPU относительно --conf")
    ap.add_argument("--glob", default="*.jpg,*.jpeg,*.png", help="маски файлов в --images")
    ap.add_argument("--raw", action="store_true", help="raw=True: упаковка на клиенте (RawCodec)")
    a = ap.parse_args()
    global NC
    NC = a.nc
    sweep = [float(s) for s in a.sweep.split(",") if s]

    import onnxruntime as ort

    models_dir = Path(os.environ.get("GUARD_MODELS_DIR", Path.home() / "work/models"))
    man = json.loads((models_dir / "tpu" / a.model / "manifest.json").read_text())
    shape = next(iter(man["inputs"].values()))["shape"]
    hw = (shape[2], shape[3])
    login = os.environ.get("LINQ_LOGIN", "hackathon10864")
    uri = a.program or f"file:///auto/cache/{login}/{Path(man['program']).name}"
    spec = ModelSpec(model_id=a.model, onnx_path=a.onnx, tpu_program=uri, input_size=hw)
    be = LinqTpuBackend(host=a.host, port=5672, models_dir=models_dir, timeout_s=60, raw=a.raw)
    be.load(spec)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    cpu = ort.InferenceSession(a.onnx, so, providers=["CPUExecutionProvider"])
    cpu_in = cpu.get_inputs()[0].name

    files = sorted({f for g in a.glob.split(",") for f in glob.glob(os.path.join(a.images, g))})
    files = files[:: max(1, len(files) // a.n_images)][: a.n_images]
    agg = {"ref": 0, "got": 0, "matched": 0, "ious": [], "dconf": [], "dkpt": []}
    raw_cos = []
    cpu_ms = []
    per_image = []
    for f in files:
        x = preprocess(f, "letterbox", hw)
        t0 = time.perf_counter()
        ref_out = cpu.run(None, {cpu_in: x})[0]
        cpu_ms.append((time.perf_counter() - t0) * 1000)
        tpu_out = be.infer(a.model, x)[0]
        # косинус по вероятностям классов (строки 4:) и по рамкам (0:4)
        for sl in (slice(0, 4), slice(4, 4 + NC)):
            r, g = ref_out[0, sl].ravel().astype(np.float64), tpu_out[0, sl].ravel().astype(np.float64)
            raw_cos.append(float(r @ g / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-12)))
        m = match(decode(ref_out, a.conf), decode(tpu_out, a.conf))
        m_lo = match(decode(ref_out, a.conf), decode(tpu_out, a.conf - a.tpu_conf_delta))
        ref_dec = decode(ref_out, a.conf)
        for d in sweep:
            ms = match(ref_dec, decode(tpu_out, round(a.conf - d, 3)))
            sw = agg.setdefault("sweep", {}).setdefault(round(a.conf - d, 3), [0, 0])
            sw[0] += ms["matched"]
            sw[1] += ms["got"]
        agg["matched_lo"] = agg.get("matched_lo", 0) + m_lo["matched"]
        agg["got_lo"] = agg.get("got_lo", 0) + m_lo["got"]
        per_image.append({"image": Path(f).name, "cpu": m["ref"], "tpu": m["got"], "matched": m["matched"]})
        for k in ("ref", "got", "matched"):
            agg[k] += m[k]
        agg["ious"] += m["ious"]
        agg["dconf"] += m["dconf"]
        agg["dkpt"] += m["dkpt"]
    acc = {
        "images": len(files), "conf": a.conf, "cpu_dets": agg["ref"], "tpu_dets": agg["got"], "matched": agg["matched"],
        "recall_vs_cpu": round(agg["matched"] / max(1, agg["ref"]), 3),
        "precision_vs_cpu": round(agg["matched"] / max(1, agg["got"]), 3),
        "iou_mean": round(float(np.mean(agg["ious"])), 3) if agg["ious"] else None,
        "iou_p05": pct(agg["ious"], 5), "dconf_abs_mean": round(float(np.mean(np.abs(agg["dconf"]))), 3) if agg["dconf"] else None,
        "dconf_signed_mean": round(float(np.mean(agg["dconf"])), 3) if agg["dconf"] else None,
        "tpu_conf_lowered": round(a.conf - a.tpu_conf_delta, 3),
        "recall_vs_cpu_lowered": round(agg.get("matched_lo", 0) / max(1, agg["ref"]), 3),
        "precision_vs_cpu_lowered": round(agg.get("matched_lo", 0) / max(1, agg.get("got_lo", 0)), 3),
        "cos_boxes_mean": round(float(np.mean(raw_cos[0::2])), 4), "cos_scores_mean": round(float(np.mean(raw_cos[1::2])), 4),
        "kpt_err_px_mean": round(float(np.mean(agg["dkpt"])), 2) if agg["dkpt"] else None,
        "kpt_err_px_p95": pct(agg["dkpt"], 95),
        "tpu_threshold_sweep": {str(k): {"recall_vs_cpu": round(v[0] / max(1, agg["ref"]), 3),
                                         "precision_vs_cpu": round(v[0] / max(1, v[1]), 3), "tpu_dets": v[1]}
                                for k, v in sorted(agg.get("sweep", {}).items(), reverse=True)},
        "per_image": per_image,
    }
    print("accuracy:", json.dumps({k: v for k, v in acc.items() if k != "per_image"}), flush=True)

    x = preprocess(files[0], "letterbox", hw)
    perf: dict = {"cpu_ort_2threads_ms": {"p50": pct(cpu_ms, 50), "p95": pct(cpu_ms, 95)}}
    for w in [int(s) for s in a.windows.split(",")]:
        xb = np.repeat(x, w, axis=0)
        stages: dict[str, list] = {}
        rounds = max(3, a.n // w)
        be.infer_timed(a.model, xb)
        t0 = time.perf_counter()
        for _ in range(rounds):
            _, t = be.infer_timed(a.model, xb)
            for k, v in t.items():
                stages.setdefault(k, []).append(v)
        dt = time.perf_counter() - t0
        perf[f"window_{w}"] = {"frames": rounds * w, "fps": round(rounds * w / dt, 2),
                               "call_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95)} for k, v in stages.items()}}
        print(f"window {w}: {perf[f'window_{w}']['fps']} fps; total p50 {pct(stages['total'], 50)} p95 "
              f"{pct(stages['total'], 95)} ms; infer p50 {pct(stages['infer'], 50)} ms", flush=True)
    res = {"model": a.model, "program": uri, "input": [1, 3, *hw], "accuracy_vs_cpu": acc, "perf": perf,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(a.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    res["raw"] = a.raw
    data[a.model + ("_raw" if a.raw else "")] = res
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    be.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
