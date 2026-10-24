#!/usr/bin/env python3
"""Проверка raw-режима: numpy RawCodec против tpu_io (побитово) и задача ?raw=True на облачном воркере.

    cd ~/work/guard_repo && ~/linq312/bin/python tpu/raw_probe.py /auto/cache/hackathon10864/yolov8n_640_b1_o5_h1m.tpu \
        ~/work/models/calib_frames/worker-zone-detection_001000.jpg
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from compile import preprocess  # noqa: E402
from guard.inference.linq_tpu import RabbitClient, RawCodec, serialize_task  # noqa: E402


def mask_for(t: dict, padded_shape: list[int]) -> tuple:
    m = []
    for ax, (a, b) in enumerate(t["padding"]):
        n = padded_shape[ax]
        before, after = abs(a), abs(b)
        m.append(tuple([False] * before + [True] * (n - before - after) + [False] * after))
    return tuple(m)


def main() -> int:
    prog = sys.argv[1]
    img = sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob(os.path.expanduser("~/work/models/calib_frames/*.jpg")))[10]
    meta = json.loads(zipfile.ZipFile(prog).read("metadata.json"))
    ins = {n: t for d in meta["inputs"].values() for n, t in d.items()}
    outs = {n: t for d in meta["outputs"].values() for n, t in d.items()}
    codec = RawCodec(ins, outs)
    name, t = next(iter(ins.items()))
    hw = tuple(t["user_shape"][2:4])
    x = preprocess(img, "letterbox", hw)
    ours = codec.pack(name, x)
    res: dict = {"program": prog, "packed_bytes": int(ours.size), "meta_size": t["size"]}
    try:
        import tpu_io

        padded = [s + a + b for s, (a, b) in zip(t["user_shape"], t["padding"])]
        ref = tpu_io.pack(x, cache_word_length=128, ddr_word_length=meta["hardware_parameters"]["ddr_word_len"],
                          user_shape_mask=mask_for(t, padded), user_order=tuple(t["user_order"]),
                          tpu_order=tuple(t["tpu_order"]), tpu_dtype=np.int8, scales=tuple(t["scales"]),
                          scale_shape=tuple(t["scale_shape"]), zero_points=tuple(t["zero_points"]))
        ref = np.asarray(ref)
        res["tpu_io_pack"] = {"shape": list(ref.shape), "dtype": str(ref.dtype),
                              "equal": bool(ref.size == ours.size and np.array_equal(ref.reshape(-1).view(np.uint8), ours))}
        if not res["tpu_io_pack"]["equal"] and ref.size == ours.size:
            d = ref.reshape(-1).view(np.int8).astype(int) - ours.view(np.int8).astype(int)
            res["tpu_io_pack"]["max_abs_diff"] = int(np.abs(d).max())
            res["tpu_io_pack"]["n_diff"] = int((d != 0).sum())
        for on, ot in outs.items():
            raw = np.random.randint(-128, 128, size=ot["size"], dtype=np.int8)
            mine = codec.unpack(on, raw.view(np.uint8))
            tpu_shape = ot["tpu_shape"]
            user_padded = [s - a - b if (a < 0 or b < 0) else s for s, (a, b) in zip(ot["user_shape"], ot["padding"])]
            refo = tpu_io.unpack(raw.reshape(tpu_shape), cache_word_length=128,
                                 user_shape_mask=mask_for(ot, user_padded), tpu_order=tuple(ot["tpu_order"]),
                                 user_order=tuple(ot["user_order"]), tpu_dtype=np.int8, user_dtype=np.float32,
                                 scales=tuple(ot["scales"]), scale_shape=tuple(ot["scale_shape"]))
            refo = np.asarray(refo)
            res.setdefault("tpu_io_unpack", {})[on] = {
                "ref_shape": list(refo.shape), "ours_shape": list(mine.shape),
                "max_abs_diff": float(np.abs(refo.reshape(mine.shape) - mine).max()) if refo.size == mine.size else None}
    except Exception as e:  # noqa: BLE001
        res["tpu_io_error"] = repr(e)

    # облачный воркер: raw=False против raw=True
    c = RabbitClient(os.environ.get("LINQ_RABBIT_HOST", "10.0.114.30"), 5672)
    c.connect()
    base_uri = "file://" + prog
    t0 = time.perf_counter()
    r_norm = c.collect([c.submit({name: x}, base_uri + "?raw=False")], 60)
    t1 = time.perf_counter()
    norm = next(iter(r_norm.values()))
    res["raw_false"] = {"ms": round((t1 - t0) * 1000, 1), "req_bytes": norm.req_bytes, "resp_bytes": norm.resp_bytes,
                        "status": norm.worker["status_code"], "outputs": {k: [list(v.shape), str(v.dtype)] for k, v in norm.data.items()}}
    variants = {"u8_flat": ours, "u8_2d": ours.reshape(-1, 32), "i8_tpu_shape": ours.view(np.int8).reshape(t["tpu_shape"])}
    for vn, arr in variants.items():
        try:
            t0 = time.perf_counter()
            rr = next(iter(c.collect([c.submit({name: arr}, base_uri + "?raw=True")], 60).values()))
            dt = (time.perf_counter() - t0) * 1000
            v = {"ms": round(dt, 1), "req_bytes": rr.req_bytes, "resp_bytes": rr.resp_bytes, "status": rr.worker["status_code"],
                 "error": rr.worker["error_message"][:300], "outputs": {k: [list(a.shape), str(a.dtype)] for k, a in rr.data.items()}}
            if rr.data and rr.worker["status_code"] == 0:
                cmp = {}
                for on in outs:
                    if on in rr.data:
                        u = codec.unpack(on, rr.data[on].reshape(-1).view(np.uint8))
                        ref = norm.data[on]
                        cmp[on] = {"max_abs_diff_vs_raw_false": float(np.abs(u.reshape(ref.shape) - ref).max())}
                v["compare"] = cmp
            res.setdefault("raw_true", {})[vn] = v
        except Exception as e:  # noqa: BLE001
            res.setdefault("raw_true", {})[vn] = {"error": repr(e)[:300]}
            c.close()
            c.connect()
    c.close()
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
