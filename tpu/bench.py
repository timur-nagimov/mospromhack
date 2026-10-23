#!/usr/bin/env python3
"""Замеры транспорта и TPU: RabbitMQ-петля (пропускная способность, RTT) и инференс моделей.

    # с ноутбука (через туннель 127.0.0.1:5672):
    python tpu/bench.py --where laptop --host 127.0.0.1 net tpu --model resnet50 --model yolov8n_640
    # из Jupyter-стенда (брокер рядом):
    python tpu/bench.py --where jupyter --host 10.0.114.30 net tpu --model resnet50
    # без стенда — моки:
    python tpu/bench.py --dry-run net tpu --model resnet50

Что меряется:
  net  — сообщение 0.1/1/5 МБ публикуется в tpu.direct с ключом своей очереди и принимается обратно
         (туда и обратно через брокер = upload + download); RTT — пустое сообщение; TCP connect.
  tpu  — N последовательных вызовов LinqTpuBackend.infer_timed (p50/p95: serialize, transfer = rtt - infer,
         infer = inference_time воркера, tail, total) + конвейер: окно из K задач в полёте -> кадров/с.
Результат дописывается в .orchestra/metrics/tpu-baseline.json под ключом <where>.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from guard.contracts import ModelSpec  # noqa: E402
from guard.inference import linq_tpu  # noqa: E402

ROOT = Path(os.environ.get("GUARD_ROOT", "/Users/nagimov-timur/Desktop/mosprom"))
METRICS = Path(os.environ.get("TPU_METRICS", ROOT / ".orchestra" / "metrics" / "tpu-baseline.json"))
MODELS = Path(os.environ.get("GUARD_MODELS_DIR", ROOT / "models"))
LOGIN = os.environ.get("LINQ_LOGIN", "hackathon10864")

# Известные модели: программа на стенде, вход, препроцессинг теста
MODEL_PRESETS = {
    "resnet50": dict(tpu_program=f"file:///auto/cache/{LOGIN}/torch_resnet50_b1_o5_h1m.tpu",
                     onnx_path=str(ROOT / "vendor/linq_sdk/tpu_framework/torch_resnet50.onnx"),
                     input_size=(224, 224), task="classify", output_format="raw"),
}


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def summary(xs: list[float]) -> dict:
    return {"p50": pct(xs, 50), "p95": pct(xs, 95), "mean": round(statistics.fmean(xs), 2) if xs else None,
            "min": round(min(xs), 2) if xs else None, "max": round(max(xs), 2) if xs else None, "n": len(xs)}


# ------------------------------------------------------------------ моки (dry-run)


class MockClient:
    """Имитация брокера+воркера: задержка = rtt + размер/скорость, ответ — нули нужной формы."""

    def __init__(self, mbps: float = 2.0, rtt_ms: float = 240.0, infer_ms: float = 8.0, out_shapes=None):
        self.mbps, self.rtt, self.infer_ms = mbps, rtt_ms / 1000, infer_ms / 1000
        self.out_shapes = out_shapes or {"495": (1, 1000)}
        self.reply_queue = "mock"
        self._q: dict[str, tuple[float, bytes]] = {}
        self.is_open = True
        self.connect_ms = 1.0

    def connect(self):
        pass

    def close(self):
        pass

    def submit(self, tensors, uri, body=None):
        body = body if body is not None else linq_tpu.serialize_task(tensors)
        mid = uuid.uuid4().hex
        if uri == "loopback":
            ready = time.monotonic() + self.rtt + 2 * len(body) / (self.mbps * 1e6)
            self._q[mid] = (ready, body)
        else:
            out = {k: np.zeros(s, np.float32) for k, s in self.out_shapes.items()}
            msg = linq_tpu._task_cls()()
            for k, v in out.items():
                a = msg.dict_data[k]
                a.dtype, a.byte_data = v.dtype.name, v.tobytes()
                a.shape.extend(v.shape)
            msg.inference_time = self.infer_ms
            rb = msg.SerializeToString()
            ready = time.monotonic() + self.rtt + (len(body) + len(rb)) / (self.mbps * 1e6) + self.infer_ms
            self._q[mid] = (ready, rb)
        return mid

    def collect(self, ids, timeout):
        out = {}
        for i in ids:
            ready, body = self._q.pop(i)
            time.sleep(max(0.0, ready - time.monotonic()))
            if body[:1] == b"L":
                out[i] = linq_tpu.Reply({}, {}, 0, time.monotonic(), len(body), len(body))
            else:
                data, worker = linq_tpu.deserialize_task(body)
                out[i] = linq_tpu.Reply(data, worker, 0, time.monotonic(), 0, len(body))
        return out


# ------------------------------------------------------------------ net


def bench_net(host: str, port: int, sizes_mb: list[float], n_rtt: int, n_size: int, dry: bool) -> dict:
    res: dict = {"host": f"{host}:{port}"}
    if not dry:
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            s = socket.create_connection((host, port), timeout=10)
            ts.append((time.perf_counter() - t0) * 1000)
            s.close()
        res["tcp_connect_ms"] = summary(ts)
    if dry:
        c = MockClient()
    else:
        c = linq_tpu.RabbitClient(host, port)
        c.connect()
        res["amqp_connect_ms"] = round(c.connect_ms, 1)

    got: list[float] = []
    if not dry:  # петля со своим постоянным консьюмером вместо консьюмера бэкенда
        for tag in list(c.ch.consumer_tags):
            c.ch.basic_cancel(tag)
        c.ch.basic_consume(c.reply_queue, lambda ch, m, p, body: got.append(time.perf_counter()), auto_ack=True)

    def loop(payload: bytes) -> tuple[float, float]:
        """-> (publish_ms, roundtrip_ms). publish блокируется, пока тело не уйдёт в сокет (≈ upload)."""
        if dry:
            t0 = time.perf_counter()
            mid = c.submit({}, "loopback", body=b"L" + payload)
            c.collect([mid], 120)
            return 0.0, (time.perf_counter() - t0) * 1000
        got.clear()
        t0 = time.perf_counter()
        c.ch.basic_publish(exchange=linq_tpu.EXCHANGE, routing_key=c.reply_queue, body=payload)
        t1 = time.perf_counter()
        deadline = time.monotonic() + 180
        while not got and time.monotonic() < deadline:
            c.conn.process_data_events(time_limit=0.05)
        if not got:
            raise TimeoutError("loopback timeout")
        return (t1 - t0) * 1000, (got[0] - t0) * 1000

    rtt = [loop(b"x" * 64)[1] for _ in range(n_rtt)]
    res["rtt_ms"] = summary(rtt)
    res["payload"] = {}
    for mb in sizes_mb:
        payload = os.urandom(int(mb * 1e6))
        runs = [loop(payload) for _ in range(n_size)]
        rt = [r[1] for r in runs]
        p50 = pct(rt, 50)
        res["payload"][f"{mb}MB"] = {
            "roundtrip_ms": summary(rt),
            "publish_ms": summary([r[0] for r in runs]),
            # туда и обратно через брокер: 2*size за (roundtrip - rtt)
            "effective_MBps_avg_two_way": round(2 * mb / max(1e-6, (p50 - pct(rtt, 50)) / 1000), 3),
        }
        print(f"net {mb} MB: roundtrip p50 {p50} ms", flush=True)
    c.close()
    return res


# ------------------------------------------------------------------ tpu


def make_input(name: str, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    img = ROOT / "vendor/linq_api/usage_example_python_api/ILSVRC2012_val_00000045.JPEG"
    from PIL import Image

    im = Image.open(img).convert("RGB")
    if name.startswith("resnet"):
        iw, ih = im.size
        m = int(min(iw, ih) * 0.85)
        im = im.crop(((iw - m) // 2, (ih - m) // 2, (iw - m) // 2 + m, (ih - m) // 2 + m)).resize((w, h))
        arr = (np.asarray(im, np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    else:
        arr = np.asarray(im.resize((w, h)), np.float32) / 255
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], np.float32)


def spec_for(name: str) -> ModelSpec:
    if name in MODEL_PRESETS:
        return ModelSpec(model_id=name, **MODEL_PRESETS[name])
    man = json.loads((MODELS / "tpu" / name / "manifest.json").read_text())
    shape = next(iter(man["inputs"].values()))["shape"]
    return ModelSpec(model_id=name, onnx_path=man["source_onnx"], tpu_program=man["program"],
                     input_size=(shape[2], shape[3]))


def bench_tpu(host: str, port: int, names: list[str], n: int, window: int, dry: bool) -> dict:
    out: dict = {}
    for name in names:
        spec = spec_for(name)
        be = linq_tpu.LinqTpuBackend(host=host, port=port, timeout_s=60)
        if dry:
            be._local.client = MockClient()
            if name != "resnet50":
                spec = ModelSpec(**{**spec.model_dump(), "tpu_program": "file:///mock.tpu"})
        be.load(spec)
        x = make_input(name, spec.input_size)
        outs, _ = be.infer_timed(name, x)  # прогрев (подключение, кеш программы на воркере)
        stages: dict[str, list[float]] = {}
        t_wall = time.perf_counter()
        for _ in range(n):
            _, t = be.infer_timed(name, x)
            for k, v in t.items():
                stages.setdefault(k, []).append(v)
        seq_fps = n / (time.perf_counter() - t_wall)
        r = {k: summary(v) for k, v in stages.items()}
        res = {"program": be._models[name].uri, "input": list(x.shape), "request_bytes": len(linq_tpu.serialize_task({be._models[name].input_name: x})),
               "n": n, "stages_ms": r, "sequential_fps": round(seq_fps, 2),
               "outputs": [list(o.shape) for o in outs]}
        if window > 1:  # конвейер: batch из window кадров -> window задач в полёте
            xb = np.repeat(x, window, axis=0)
            rounds = max(2, n // window)
            t0 = time.perf_counter()
            for _ in range(rounds):
                be.infer_timed(name, xb)
            res["pipelined"] = {"window": window, "frames": rounds * window,
                                "fps": round(rounds * window / (time.perf_counter() - t0), 2)}
        if name == "resnet50":
            res["top1_class"] = int(np.argmax(outs[0])) + 1
        out[name] = res
        print(f"tpu {name}: total p50 {r['total']['p50']} ms p95 {r['total']['p95']} ms; infer p50 {r['infer']['p50']} ms;"
              f" seq {seq_fps:.2f} fps; pipelined {res.get('pipelined')}", flush=True)
        be.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["net", "tpu"])
    ap.add_argument("--where", default="laptop", help="метка места запуска: laptop | jupyter | ...")
    ap.add_argument("--host", default=os.environ.get("LINQ_RABBIT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("LINQ_RABBIT_PORT", "5672")))
    ap.add_argument("--model", action="append", default=[], help="resnet50 или имя каталога models/tpu/<name>")
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--window", type=int, default=4, help="задач в полёте для конвейерного теста")
    ap.add_argument("--sizes", default="0.1,1,5", help="МБ для net")
    ap.add_argument("--n-size", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="без стенда: моки брокера и воркера")
    ap.add_argument("--out", default=str(METRICS))
    a = ap.parse_args()
    where = a.where + ("-dryrun" if a.dry_run else "")
    rep: dict = {"where": where, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname()}
    if "net" in a.what:
        rep["net"] = bench_net(a.host, a.port, [float(s) for s in a.sizes.split(",")], 10, a.n_size, a.dry_run)
    if "tpu" in a.what:
        rep["tpu"] = bench_tpu(a.host, a.port, a.model or ["resnet50"], a.n, a.window, a.dry_run)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    data.setdefault(where, {}).update({k: v for k, v in rep.items() if k in ("net", "tpu")})
    data[where].update({"ts": rep["ts"], "host": rep["host"]})
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False)[:3000])
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
