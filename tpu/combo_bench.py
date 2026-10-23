#!/usr/bin/env python3
"""Совместная нагрузка: на каждый кадр — все модели (det + pose), окно кадров в полёте. Кадров/с конвейера.

    cd ~/work/guard_repo && GUARD_MODELS_DIR=$HOME/work/models OMP_NUM_THREADS=2 \
      ~/linq312/bin/python tpu/combo_bench.py --models yolov8n_480,yolov8n_pose_480 --frames 64 --windows 1,2,4 --raw
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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
from compile import preprocess  # noqa: E402
from guard.contracts import ModelSpec  # noqa: E402
from guard.inference.linq_tpu import LinqTpuBackend  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="yolov8n_480,yolov8n_pose_480")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--windows", default="1,2,4")
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/work/metrics/combo.json"))
    a = ap.parse_args()
    md = Path(os.environ.get("GUARD_MODELS_DIR", Path.home() / "work/models"))
    names = a.models.split(",")
    be = LinqTpuBackend(models_dir=md, raw=a.raw, window=max(int(w) for w in a.windows.split(",")) * len(names))
    xs = {}
    for n in names:
        man = json.loads((md / "tpu" / n / "manifest.json").read_text())
        shp = next(iter(man["inputs"].values()))["shape"]
        be.load(ModelSpec(model_id=n, onnx_path=man["source_onnx"], tpu_program=f"file:///auto/cache/{os.environ.get('LINQ_LOGIN', 'hackathon10864')}/{Path(man['program']).name}",
                          input_size=(shp[2], shp[3])))
        img = sorted(glob.glob(str(md / "calib_frames/*.jpg")))[5]
        xs[n] = preprocess(img, "letterbox", (shp[2], shp[3]))
        be.infer(n, xs[n])  # прогрев
    res = {"models": names, "raw": a.raw, "host": be.host, "windows": {}}
    for w in [int(s) for s in a.windows.split(",")]:
        lat = []
        t0 = time.perf_counter()
        inflight: list[tuple[float, list]] = []
        done = 0
        for i in range(a.frames):
            inflight.append((time.perf_counter(), [be.submit(n, xs[n]) for n in names]))
            while len(inflight) >= w or (i == a.frames - 1 and inflight):
                ts, futs = inflight.pop(0)
                cf.wait(futs)
                for f in futs:
                    f.result()
                lat.append((time.perf_counter() - ts) * 1000)
                done += 1
        dt = time.perf_counter() - t0
        res["windows"][w] = {"frames": done, "frames_per_s": round(done / dt, 2),
                             "frame_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                                                  "p95": round(float(np.percentile(lat, 95)), 1)}}
        print(f"window {w}: {res['windows'][w]}", flush=True)
    res["stats"] = {k: {"fps": v["fps"], "errors": v["errors"]} for k, v in be.stats()["models"].items()}
    be.close()
    out = Path(a.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data["+".join(names) + ("_raw" if a.raw else "")] = res
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
