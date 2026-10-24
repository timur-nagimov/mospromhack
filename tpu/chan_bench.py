#!/usr/bin/env python3
"""Канал «сервер Jupyter -> ноутбук» через RabbitMQ: свои exchange и очереди, латентность и пропускная способность.

На сервере (ответчик; живёт --duration секунд):
    python tpu/chan_bench.py serve --host 10.0.114.30 --duration 300
На ноутбуке (через туннель):
    python tpu/chan_bench.py client --host 127.0.0.1 --sizes 2048,61440 -n 20 --burst 50

Схема: exchange `zazerkalye` (direct, не durable, не auto_delete).
  * `zazerkalye.req` — запросы клиента {size, count, reply};
  * `zazerkalye.events` — очередь клиента (auto_delete=false, как будущая очередь событий).
Ответчик на каждый запрос публикует count сообщений по size байт в routing key = reply.
Латентность = время «запрос (64 Б) -> первый ответ»; запрос идёт ноутбук -> сервер, ответ — сервер -> ноутбук.
Пропускная способность = count*size / (последний ответ - первый ответ).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import pika

EXCHANGE = "zazerkalye"
REQ_QUEUE = "zazerkalye.req"
EVENTS_QUEUE = "zazerkalye.events"


def connect(host: str, port: int) -> pika.BlockingConnection:
    return pika.BlockingConnection(pika.ConnectionParameters(
        host, port, credentials=pika.PlainCredentials("guest", "guest"), heartbeat=60))


def declare(ch, queue: str) -> None:
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=False, auto_delete=False)
    ch.queue_declare(queue=queue, durable=False, auto_delete=False)
    ch.queue_bind(exchange=EXCHANGE, queue=queue, routing_key=queue)


def serve(a) -> int:
    conn = connect(a.host, a.port)
    ch = conn.channel()
    declare(ch, REQ_QUEUE)
    ch.queue_purge(REQ_QUEUE)
    deadline = time.time() + a.duration
    print(f"serve: {REQ_QUEUE} на {a.host}:{a.port}, до {time.strftime('%H:%M:%S', time.localtime(deadline))}", flush=True)
    n = 0
    while time.time() < deadline:
        method, props, body = ch.basic_get(REQ_QUEUE, auto_ack=True)
        if method is None:
            conn.process_data_events(time_limit=0.01)
            continue
        req = json.loads(body)
        payload = os.urandom(int(req["size"]))
        for i in range(int(req.get("count", 1))):
            ch.basic_publish(EXCHANGE, req["reply"], payload,
                             pika.BasicProperties(headers={"i": i, "server_ts": repr(time.time())}))
        n += 1
    print(f"serve: обработано {n} запросов", flush=True)
    conn.close()
    return 0


def client(a) -> int:
    conn = connect(a.host, a.port)
    ch = conn.channel()
    declare(ch, EVENTS_QUEUE)
    ch.queue_purge(EVENTS_QUEUE)
    got: list[tuple[float, int]] = []
    ch.basic_consume(EVENTS_QUEUE, lambda c, m, p, b: got.append((time.perf_counter(), len(b))), auto_ack=True)

    def request(size: int, count: int, timeout: float = 120) -> tuple[float, float, list]:
        got.clear()
        t0 = time.perf_counter()
        ch.basic_publish(EXCHANGE, REQ_QUEUE, json.dumps({"size": size, "count": count, "reply": EVENTS_QUEUE}))
        while len(got) < count and time.perf_counter() - t0 < timeout:
            conn.process_data_events(time_limit=0.01)
        if len(got) < count:
            raise TimeoutError(f"получено {len(got)}/{count}")
        return t0, got[0][0], list(got)

    request(64, 1)  # прогрев / проверка, что ответчик жив
    res: dict = {"exchange": EXCHANGE, "queue": EVENTS_QUEUE, "declare_ok": True, "sizes": {}}
    base = [1000 * (r[1] - r[0]) for r in (request(64, 1) for _ in range(a.n))]
    res["latency_64B_ms"] = {"p50": round(statistics.median(base), 1), "p95": round(sorted(base)[int(0.95 * (len(base) - 1))], 1)}
    for size in [int(s) for s in a.sizes.split(",")]:
        lat = [1000 * (r[1] - r[0]) for r in (request(size, 1) for _ in range(a.n))]
        t0, t_first, msgs = request(size, a.burst)
        span = msgs[-1][0] - t_first
        res["sizes"][f"{size}B"] = {
            "latency_ms": {"p50": round(statistics.median(lat), 1), "p95": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
                           "n": len(lat)},
            "burst": {"count": a.burst, "first_ms": round(1000 * (t_first - t0), 1), "span_s": round(span, 3),
                      "msgs_per_s": round((a.burst - 1) / max(span, 1e-9), 1),
                      "MBps": round(size * (a.burst - 1) / 1e6 / max(span, 1e-9), 3)},
        }
        print(f"{size} B: latency p50 {res['sizes'][f'{size}B']['latency_ms']['p50']} ms; burst {res['sizes'][f'{size}B']['burst']}", flush=True)
    conn.close()
    out = Path(a.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data.setdefault(a.where, {})["channel_server_to_client"] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **res}
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(json.dumps(res, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["serve", "client"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5672)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--sizes", default="2048,61440")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--burst", type=int, default=50)
    ap.add_argument("--where", default="laptop")
    ap.add_argument("--out", default="/Users/nagimov-timur/Desktop/mosprom/.orchestra/metrics/tpu-baseline.json")
    a = ap.parse_args()
    return serve(a) if a.role == "serve" else client(a)


if __name__ == "__main__":
    raise SystemExit(main())
