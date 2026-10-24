"""Замеряющий подписчик на стороне шлюза: что и с какой задержкой доходит с edge (docs/EDGE.md).

    python -m guard.edge_probe --duration 300 --out edge-tpu-<время>.json

Читает очередь шлюза zazerkalye.gateway (по умолчанию через VPN-контейнер, 127.0.0.1:5672) и считает:
- сквозную задержку оповещения: момент получения на ноутбуке − Event.started_at (первый нарушающий
  кадр) с поправкой на разность часов; отдельно — от confirmed_at (кадр подтверждения) и от
  details.edge.sent_at (передача в транспорт на edge), то есть задержку канала;
- задержку frame_meta (захват кадра на edge → ноутбук) и долю потерянных frame_meta: сравнение
  полученных с приростом frames_processed и счётчиком отправленных из metrics.edge;
- FPS по камерам, загрузку CPU контейнера, задержки по этапам и статистику инференса — из metrics;
- объём и темп сообщений по видам.

Разность часов стенда и ноутбука D = часы ноутбука − часы стенда оценивается двумя способами:
- NTP: edge публикует свой сдвиг в metrics.edge.clock, подписчик меряет свой; D = off_стенд − off_ноутбук;
- RTT: min(t_получения − ts heartbeat/metrics) − RTT_min/2, где RTT_min — круг ноутбук ↔ брокер.
Используется NTP (если есть обе стороны), RTT — для сверки. Погрешность — в отчёте.

prefetch: RabbitSource шлюза берёт по одному сообщению (prefetch_count=1, ack на каждое). Через VPN
(RTT ~235 мс) это ~4 сообщения/с. --prefetch 1 --ack-every 1 повторяет поведение шлюза.
Очередь общая: при работающем шлюзе подписчик делит с ним сообщения — смотрите consumers в отчёте.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from guard.sysinfo import ntp_best
from guard.transport.codec import decode
from guard.transport.messages import EventMsg, FrameMetaMsg, HeartbeatMsg, MetricsMsg

DEFAULT_URL = "amqp://guest:guest@127.0.0.1:5672/"
GATEWAY_QUEUE = "zazerkalye.gateway"
EXCHANGE = "zazerkalye"
BINDINGS = ("edge.event", "edge.frame_meta", "edge.metrics")


def stats(values: list[float], scale: float = 1.0, digits: int = 1) -> dict[str, Any] | None:
    xs = sorted(v * scale for v in values if v is not None)
    if not xs:
        return None

    def pct(p: float) -> float:
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    return {
        "n": len(xs),
        "p50": round(pct(0.5), digits),
        "p95": round(pct(0.95), digits),
        "mean": round(statistics.fmean(xs), digits),
        "min": round(xs[0], digits),
        "max": round(xs[-1], digits),
    }


class Recorder:
    def __init__(self) -> None:
        self.t_start = time.time()
        self.counts: dict[str, int] = defaultdict(int)
        self.bytes: dict[str, int] = defaultdict(int)
        self.events: dict[str, dict[str, Any]] = {}
        self.event_duplicates = 0
        self.updates = 0
        self.frame_meta: dict[str, list[tuple[float, float]]] = defaultdict(list)  # (ts edge, recv)
        self.metrics: list[tuple[float, dict[str, Any]]] = []
        self.clock_samples: list[float] = []  # recv − ts edge для heartbeat и metrics
        self.decode_errors = 0
        self.first_recv: float | None = None
        self.last_recv: float | None = None
        self.consume_started: float | None = None  # часы ноутбука: начало приёма

    def add(self, msg: Any, recv: float, size: int) -> None:
        kind = getattr(msg, "kind", "?")
        self.counts[kind] += 1
        self.bytes[kind] += size
        self.first_recv = self.first_recv or recv
        self.last_recv = recv
        if isinstance(msg, EventMsg):
            if msg.kind == "event_update":
                self.updates += 1
                return
            ev = msg.event
            if ev.id in self.events:
                self.event_duplicates += 1
                return
            edge = (ev.details or {}).get("edge") or {}
            self.events[ev.id] = {
                "id": ev.id,
                "camera_id": ev.camera_id,
                "type": ev.type,
                "track_id": ev.track_id,
                "started_at": ev.started_at,
                "confirmed_at": ev.confirmed_at,
                "sent_at": edge.get("sent_at"),
                "recv": recv,
                "latency_ms": dict(ev.latency_ms or {}),
                "snapshot_kb": round(len(msg.snapshot_jpeg) / 1024, 1) if msg.snapshot_jpeg else None,
                "body_kb": round(size / 1024, 1),
            }
        elif isinstance(msg, FrameMetaMsg):
            self.frame_meta[msg.camera_id].append((msg.ts, recv))
        elif isinstance(msg, MetricsMsg):
            self.metrics.append((recv, msg.data))
            ts = (msg.data.get("edge") or {}).get("ts")
            if isinstance(ts, (int, float)):
                self.clock_samples.append(recv - ts)
        elif isinstance(msg, HeartbeatMsg):
            self.clock_samples.append(recv - msg.ts)


def broker_rtt(url: str, queue: str, n: int = 15) -> dict[str, Any]:
    """Круг ноутбук ↔ брокер (passive declare) и состояние очереди перед замером."""
    import pika

    conn = pika.BlockingConnection(pika.URLParameters(url))
    try:
        ch = conn.channel()
        samples = []
        res = None
        for _ in range(n):
            t0 = time.perf_counter()
            res = ch.queue_declare(queue=queue, passive=True)
            samples.append((time.perf_counter() - t0) * 1000.0)
        assert res is not None
        return {
            "rtt_ms": stats(samples),
            "rtt_min_ms": round(min(samples), 1),
            "consumers_before": res.method.consumer_count,
            "messages_before": res.method.message_count,
        }
    finally:
        conn.close()


def consume(url: str, queue: str, duration: float, prefetch: int, ack_every: int, rec: Recorder,
            purge: bool = False, tap: bool = False) -> None:
    import pika

    conn = pika.BlockingConnection(pika.URLParameters(url))
    ch = conn.channel()
    if tap:
        # Своя временная очередь на тех же ключах: замер получает копию потока, а не делит очередь шлюза
        # (у общей очереди брокер раздаёт сообщения потребителям по очереди, и шлюз теряет часть событий).
        queue = ch.queue_declare(queue="", exclusive=True, auto_delete=True,
                                 arguments={"x-max-length": 5000}).method.queue
        for routing_key in BINDINGS:
            ch.queue_bind(queue=queue, exchange=EXCHANGE, routing_key=routing_key)
    elif purge:
        ch.queue_purge(queue)
    ch.basic_qos(prefetch_count=prefetch)
    deadline = time.monotonic() + duration
    rec.consume_started = time.time()
    unacked = 0
    last_tag = None
    last_ack = time.monotonic()
    try:
        for method, props, body in ch.consume(queue, auto_ack=False, inactivity_timeout=0.25):
            now = time.time()
            if method is not None:
                try:
                    msg = decode(body, props.headers or {})
                except (ValueError, TypeError):
                    rec.decode_errors += 1
                    msg = None
                if msg is not None:
                    rec.add(msg, now, len(body))
                last_tag = method.delivery_tag
                unacked += 1
            if last_tag is not None and unacked and (unacked >= ack_every or time.monotonic() - last_ack > 0.2):
                ch.basic_ack(delivery_tag=last_tag, multiple=True)
                unacked = 0
                last_ack = time.monotonic()
            if time.monotonic() >= deadline:
                break
        if last_tag is not None and unacked:
            ch.basic_ack(delivery_tag=last_tag, multiple=True)
        ch.cancel()
    finally:
        conn.close()


def _edge(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("edge") or {}


def summarize(rec: Recorder, laptop_ntp: dict[str, Any] | None, rtt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    metrics = rec.metrics
    edge_last = _edge(metrics[-1][1]) if metrics else {}
    # --- часы
    stand_clock = None
    for _, data in reversed(metrics):
        clock = _edge(data).get("clock") or {}
        if "offset_ms" in clock:
            stand_clock = clock
            break
    clock: dict[str, Any] = {"stand_ntp": stand_clock, "laptop_ntp": laptop_ntp}
    d_ntp = None
    if stand_clock and laptop_ntp and "offset_ms" in laptop_ntp:
        d_ntp = (stand_clock["offset_ms"] - laptop_ntp["offset_ms"]) / 1000.0
        clock["ntp"] = {
            "laptop_minus_stand_ms": round(d_ntp * 1000, 1),
            "uncertainty_ms": round(stand_clock.get("uncertainty_ms", 0) + laptop_ntp.get("uncertainty_ms", 0), 1),
        }
    d_rtt = None
    if rec.clock_samples and rtt.get("rtt_min_ms") is not None:
        d_rtt = min(rec.clock_samples) - rtt["rtt_min_ms"] / 2000.0
        clock["rtt"] = {
            "laptop_minus_stand_ms": round(d_rtt * 1000, 1),
            "uncertainty_ms": round(rtt["rtt_min_ms"] / 2, 1),
            "min_one_way_plus_offset_ms": round(min(rec.clock_samples) * 1000, 1),
            "samples": len(rec.clock_samples),
        }
    if d_ntp is not None:
        d, clock["used"] = d_ntp, "ntp"
    elif d_rtt is not None:
        d, clock["used"] = d_rtt, "rtt"
    else:
        d, clock["used"] = 0.0, "none (часы считаются синхронными)"
    if d_ntp is not None and d_rtt is not None:
        clock["ntp_vs_rtt_ms"] = round((d_ntp - d_rtt) * 1000, 1)
    clock["laptop_minus_stand_used_ms"] = round(d * 1000, 1)

    # --- события
    all_events = sorted(rec.events.values(), key=lambda e: e["recv"])
    start_stand = (rec.consume_started or rec.t_start) - d
    # метрики из очереди до начала приёма могут быть от прошлого запуска edge — FPS, CPU и потери только по свежим
    current = [m for m in rec.metrics if (_edge(m[1]).get("ts") or 0) >= start_stand]
    metrics_backlog = len(rec.metrics) - len(current)
    if len(current) >= 2:
        metrics = current
        edge_last = _edge(metrics[-1][1])
    for e in all_events:
        # событие ушло с edge до подключения подписчика — ждало в очереди, в задержку не входит
        e["backlog"] = bool(e.get("sent_at") and e["sent_at"] < start_stand)
    events = [e for e in all_events if not e["backlog"]]
    for e in all_events:
        recv_stand = e["recv"] - d
        e["e2e_from_started_s"] = round(recv_stand - e["started_at"], 3)
        e["e2e_from_confirmed_s"] = round(recv_stand - e["confirmed_at"], 3)
        e["confirm_window_s"] = round(e["confirmed_at"] - e["started_at"], 3)
        if e.get("sent_at"):
            e["edge_confirmed_to_sent_s"] = round(e["sent_at"] - e["confirmed_at"], 3)
            e["transport_s"] = round(recv_stand - e["sent_at"], 3)
    ev_summary = {
        "count": len(events),
        "backlog_excluded": len(all_events) - len(events),
        "duplicates": rec.event_duplicates,
        "updates": rec.updates,
        "by_type": dict(sorted({t: sum(1 for e in events if e["type"] == t) for t in {e["type"] for e in events}}.items())),
        "e2e_from_started_s": stats([e["e2e_from_started_s"] for e in events], digits=3),
        "e2e_from_confirmed_s": stats([e["e2e_from_confirmed_s"] for e in events], digits=3),
        "confirm_window_s": stats([e["confirm_window_s"] for e in events], digits=3),
        "edge_confirmed_to_sent_s": stats([e.get("edge_confirmed_to_sent_s") for e in events], digits=3),
        "transport_s": stats([e.get("transport_s") for e in events], digits=3),
        "pipeline_total_ms": stats([e["latency_ms"].get("total") for e in events]),
        "snapshot_kb": stats([e["snapshot_kb"] for e in events]),
        "list": all_events,
    }

    # --- frame_meta: задержка и потери
    fm: dict[str, Any] = {"by_camera": {}}
    m_first = metrics[0] if metrics else None
    m_last = metrics[-1] if metrics else None
    all_lat: list[float] = []
    for cid, items in sorted(rec.frame_meta.items()):
        # кадры, снятые до начала приёма, ждали в очереди — в задержку не входят (как backlog событий)
        lat = [recv - d - ts for ts, recv in items if ts >= start_stand]
        all_lat.extend(lat)
        entry: dict[str, Any] = {"received": len(items), "latency_s": stats(lat, digits=3)}
        if m_first and m_last and m_last is not m_first:
            t0 = _edge(m_first[1]).get("ts")
            t1 = _edge(m_last[1]).get("ts")
            p0 = (_edge(m_first[1]).get("frames_processed") or {}).get(cid)
            p1 = (_edge(m_last[1]).get("frames_processed") or {}).get(cid)
            s0 = ((_edge(m_first[1]).get("transport") or {}).get("frame_meta_by_camera") or {}).get(cid)
            s1 = ((_edge(m_last[1]).get("transport") or {}).get("frame_meta_by_camera") or {}).get(cid)
            if t0 and t1:
                got = sum(1 for ts, _ in items if t0 <= ts <= t1)
                entry["window_s"] = round(t1 - t0, 1)
                entry["received_in_window"] = got
                if p0 is not None and p1 is not None and p1 > p0:
                    entry["processed_in_window"] = p1 - p0
                    entry["loss_vs_processed_pct"] = round(100.0 * max(0, (p1 - p0) - got) / (p1 - p0), 2)
                if s0 is not None and s1 is not None and s1 > s0:
                    entry["sent_in_window"] = s1 - s0
                    entry["loss_vs_sent_pct"] = round(100.0 * max(0, (s1 - s0) - got) / (s1 - s0), 2)
        fm["by_camera"][cid] = entry
    fm["latency_s"] = stats(all_lat, digits=3)
    processed = sum(e.get("processed_in_window", 0) for e in fm["by_camera"].values())
    got_total = sum(e.get("received_in_window", 0) for e in fm["by_camera"].values() if "processed_in_window" in e)
    if processed:
        fm["loss_vs_processed_pct"] = round(100.0 * max(0, processed - got_total) / processed, 2)
    if m_first and m_last:
        t_first = _edge(m_first[1]).get("transport") or {}
        t_last = _edge(m_last[1]).get("transport") or {}

        def delta(key: str) -> dict[str, int]:
            a, b = t_first.get(key) or {}, t_last.get(key) or {}
            return {k: int(b.get(k, 0)) - int(a.get(k, 0)) for k in set(a) | set(b) if int(b.get(k, 0)) - int(a.get(k, 0))}

        fm["edge_held_by_guard"] = delta("held_by_guard")
        fm["edge_outbox_dropped"] = delta("outbox_dropped")
        fm["edge_rejected"] = delta("rejected")

    # --- FPS, CPU, этапы, инференс
    cams: dict[str, Any] = {}
    cpu_cores, cpu_pct, proc_cores, rss = [], [], [], []
    for _, data in metrics:
        e = _edge(data)
        cpu = e.get("cpu") or {}
        if cpu.get("container_cores") is not None:
            cpu_cores.append(cpu["container_cores"])
        if cpu.get("container_pct_of_limit") is not None:
            cpu_pct.append(cpu["container_pct_of_limit"])
        if cpu.get("process_cores") is not None:
            proc_cores.append(cpu["process_cores"])
        mem = e.get("memory") or {}
        if mem.get("process_rss_mb") is not None:
            rss.append(mem["process_rss_mb"])
        for cid, cm in (data.get("cameras") or {}).items():
            c = cams.setdefault(cid, {"fps_processed": [], "fps_in": [], "lag_s": []})
            c["fps_processed"].append(cm.get("fps_processed"))
            c["fps_in"].append(cm.get("fps_in"))
            c["lag_s"].append(cm.get("lag_s"))
    fps = {
        cid: {
            "fps_processed": stats(c["fps_processed"], digits=2),
            "fps_in": stats(c["fps_in"], digits=2),
            "lag_s": stats(c["lag_s"], digits=3),
        }
        for cid, c in sorted(cams.items())
    }
    last_data = metrics[-1][1] if metrics else {}
    stages = {cid: cm.get("timings_ms") for cid, cm in (last_data.get("cameras") or {}).items()}
    duration = (rec.last_recv - rec.first_recv) if rec.first_recv and rec.last_recv else 0.0
    return {
        "label": args.label,
        "probe": {
            "started_at": round(rec.t_start, 3),
            "duration_s": round(duration, 1),
            "url_host": args.url.split("@")[-1],
            "queue": args.queue,
            "prefetch": args.prefetch,
            "ack_every": args.ack_every,
            "broker": rtt,
            "decode_errors": rec.decode_errors,
            "metrics_backlog_excluded": metrics_backlog,
        },
        "edge": {
            "node_id": edge_last.get("node_id"),
            "backend": edge_last.get("backend") or last_data.get("backend"),
            "uptime_s": edge_last.get("uptime_s"),
            "events_created": edge_last.get("events_created"),
            "sink_errors": edge_last.get("sink_errors"),
        },
        "clock": clock,
        "events": ev_summary,
        "frame_meta": fm,
        "fps": fps,
        "cpu": {
            "container_cores": stats(cpu_cores, digits=3),
            "container_pct_of_limit": stats(cpu_pct),
            "process_cores": stats(proc_cores, digits=3),
            "limit_cores": ((edge_last.get("cpu") or {}).get("limit_cores")),
            "process_rss_mb": stats(rss),
            "memory_last": edge_last.get("memory"),
        },
        "stages_ms_last_window": stages,
        "inference_last": last_data.get("inference"),
        "channel": {
            "messages": dict(rec.counts),
            "kb": {k: round(v / 1024, 1) for k, v in rec.bytes.items()},
            "msgs_per_s": round(sum(rec.counts.values()) / duration, 2) if duration else None,
            "kb_per_s": round(sum(rec.bytes.values()) / 1024 / duration, 2) if duration else None,
        },
        "transport_last": edge_last.get("transport"),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m guard.edge_probe", description="Замеры канала edge → шлюз")
    p.add_argument("--url", default=DEFAULT_URL, help="AMQP URL брокера (с ноутбука — 127.0.0.1:5672 через VPN-контейнер)")
    p.add_argument("--queue", default=GATEWAY_QUEUE)
    p.add_argument("--duration", type=float, default=120.0, help="секунд приёма")
    p.add_argument("--prefetch", type=int, default=200, help="1 — как RabbitSource шлюза")
    p.add_argument("--ack-every", type=int, default=50, help="групповой ack; 1 — как RabbitSource")
    p.add_argument("--purge", action="store_true", help="очистить очередь перед приёмом (старые сообщения)")
    p.add_argument("--tap", action="store_true",
                   help="копия потока через свою очередь: не мешает работающему шлюзу. frame_meta и metrics "
                        "edge шлёт, только пока у очереди шлюза есть потребитель (docs/EDGE.md §4)")
    p.add_argument("--no-ntp", action="store_true")
    p.add_argument("--label", default="edge-tpu")
    p.add_argument("--out", help="JSON с результатами")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ntp_servers = ("time.google.com", "pool.ntp.org", "time.cloudflare.com")
    laptop_ntp = None if args.no_ntp else ntp_best(ntp_servers, samples=6)
    rtt = broker_rtt(args.url, args.queue)
    print(f"broker rtt {rtt['rtt_ms']}, consumers before {rtt['consumers_before']}, "
          f"messages before {rtt['messages_before']}; laptop ntp {laptop_ntp}", flush=True)
    if rtt["consumers_before"] and not args.tap:
        print("ВНИМАНИЕ: у очереди уже есть потребитель (шлюз?) — сообщения делятся между ним и замером", flush=True)
    rec = Recorder()
    consume(args.url, args.queue, args.duration, args.prefetch, args.ack_every, rec, purge=args.purge, tap=args.tap)
    if not args.no_ntp:
        again = ntp_best(ntp_servers, samples=6)
        if again and "offset_ms" in again and (not laptop_ntp or "offset_ms" not in laptop_ntp
                                               or again["rtt_ms"] < laptop_ntp["rtt_ms"]):
            laptop_ntp = again
    result = summarize(rec, laptop_ntp, rtt, args)
    ev = result["events"]
    fm = result["frame_meta"]
    print(json.dumps({
        "events": ev["count"],
        "e2e_from_started_s": ev["e2e_from_started_s"],
        "e2e_from_confirmed_s": ev["e2e_from_confirmed_s"],
        "transport_s": ev["transport_s"],
        "frame_meta_latency_s": fm.get("latency_s"),
        "frame_meta_loss_pct": fm.get("loss_vs_processed_pct"),
        "clock": {k: result["clock"].get(k) for k in ("used", "laptop_minus_stand_used_ms", "ntp_vs_rtt_ms")},
        "channel": result["channel"],
    }, ensure_ascii=False, indent=1))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
