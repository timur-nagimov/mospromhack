#!/usr/bin/env python3
"""Смоук LinqTpuBackend в окружении конвейера: модели по относительному пути программы, raw, батч, submit.

    cd ~/work/guard_repo && GUARD_MODELS_DIR=$HOME/work/models OMP_NUM_THREADS=2 ~/guard311/bin/python tpu/smoke_backend.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from guard.contracts import ModelSpec  # noqa: E402
from guard.inference import get_backend  # noqa: E402

MODELS = {
    "det": ("yolov8n-480.onnx", "tpu/yolov8n_480/yolov8n_480_b1_o5_h1m.tpu", (1, 84, 4725)),
    "pose": ("yolov8n-pose-480.onnx", "tpu/yolov8n_pose_480/yolov8n_pose_480_b1_o5_h1m.tpu", (1, 56, 4725)),
}

be = get_backend("linq-tpu", raw=os.environ.get("LINQ_RAW", "1") == "1")
print("backend", be.name, "host", be.host, "raw", be.raw, flush=True)
for mid, (onnx_path, prog, shape) in MODELS.items():
    be.load(ModelSpec(model_id=mid, onnx_path=onnx_path, tpu_program=prog, input_size=(480, 480)))
x = np.random.rand(4, 3, 480, 480).astype(np.float32)
for mid, (_, _, shape) in MODELS.items():
    outs, t = be.infer_timed(mid, x)
    assert outs[0].shape == (4,) + shape[1:], outs[0].shape
    print(mid, be._models[mid].uri, outs[0].shape, {k: round(v, 1) for k, v in t.items()}, flush=True)
t0 = time.perf_counter()
futs = [be.submit(m, x) for m in MODELS]
print("submit det+pose batch4:", [f.result()[0][0].shape for f in futs], f"{(time.perf_counter() - t0) * 1000:.0f} ms")
st = be.stats()
print("stats:", {m: (v["items"], v["errors"]) for m, v in st["models"].items()}, st["bytes"])
be.close()
print("SMOKE_OK")
