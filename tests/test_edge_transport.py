"""EdgeSink, защита очереди шлюза, метрики edge и расчёты замеряющего подписчика (A10)."""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pytest

from guard import edge
from guard.contracts import Event, Severity
from guard.edge_probe import Recorder, stats, summarize
from guard.sysinfo import CpuSampler, cgroup_cpu_limit, cgroup_cpu_usage_s, cgroup_memory
from guard.transport import EventMsg, FrameMetaMsg, HeartbeatMsg, MemoryTransport, MetricsMsg


def _event(eid: str = "e1", started: float = 100.0, confirmed: float = 102.0) -> Event:
    return Event(id=eid, camera_id="cam_s1", site="s", rule_id="r", type="zone_intrusion", severity=Severity.high,
                 started_at=started, confirmed_at=confirmed, bbox=(0, 0, 10, 10), latency_ms={"total": 80.0})


def _drain(transport: MemoryTransport) -> list:
    transport.close()
    return list(transport.iter())


def test_sink_stamps_events_and_counts():
    transport = MemoryTransport()
    sink = edge.EdgeSink(transport, node_id="n1")
    before = time.time()
    sink.publish(EventMsg(kind="event", event=_event(), snapshot_jpeg=b"\xff\xd8" + b"x" * 1000))
    sink.publish(FrameMetaMsg(camera_id="cam_s1", frame_idx=5, ts=1.0, w=640, h=360))
    sink.publish(FrameMetaMsg(camera_id="cam_s2", frame_idx=5, ts=1.0, w=640, h=360))
    sink.publish(HeartbeatMsg(node_id="n1"))
    msgs = _drain(transport)
    stamped = msgs[0].event.details["edge"]
    assert stamped["node"] == "n1" and before - 0.01 <= stamped["sent_at"] <= time.time() + 0.01
    st = sink.stats()
    assert st["sent"] == {"event": 1, "frame_meta": 2, "heartbeat": 1}
    assert st["frame_meta_by_camera"] == {"cam_s1": 1, "cam_s2": 1}
    assert st["snapshots"] == 1 and st["snapshot_kb_avg"] == pytest.approx(1002 / 1024, abs=0.1)


def test_guard_holds_telemetry_without_gateway_but_not_events(monkeypatch):
    guard = edge.GatewayQueueGuard(url=None, backlog_limit=10)
    transport = MemoryTransport()
    sink = edge.EdgeSink(transport, guard=guard)
    meta = FrameMetaMsg(camera_id="c", frame_idx=0, ts=1.0, w=2, h=2)

    assert sink.publish(meta) is not False  # состояние очереди неизвестно — ничего не блокируем
    guard.update(consumers=0, messages=3)
    assert sink.publish(meta) is False
    assert sink.publish(MetricsMsg(data={})) is False
    assert sink.publish(EventMsg(kind="event", event=_event())) is not False
    assert sink.publish(HeartbeatMsg()) is not False  # первый heartbeat без шлюза проходит
    assert sink.publish(HeartbeatMsg()) is False  # следующий — не раньше чем через 30 с
    assert sink.stats()["held_by_guard"] == {"frame_meta": 1, "metrics": 1, "heartbeat": 1}
    assert sink.stats()["gateway_queue"]["frame_meta_paused"] is True

    guard.update(consumers=1, messages=11)  # шлюз есть, но очередь выше порога — frame_meta на паузе
    assert sink.publish(meta) is False and sink.publish(MetricsMsg(data={})) is not False
    guard.update(consumers=1, messages=7)  # гистерезис: выше половины порога пауза держится
    assert sink.publish(meta) is False
    guard.update(consumers=1, messages=5)
    assert sink.publish(meta) is not False
    kinds = [m.kind for m in _drain(transport)]
    assert kinds.count("event") == 1 and kinds.count("frame_meta") == 2


def test_json_safe_metrics():
    data = {"a": float("nan"), "b": np.float32(1.5), "c": [np.int64(3), float("inf")], "d": {"e": None}}
    assert edge._json_safe(data) == {"a": None, "b": 1.5, "c": [3, None], "d": {"e": None}}


def test_disable_ort_spinning(tmp_path, monkeypatch):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    monkeypatch.setattr(ort, "InferenceSession", ort.InferenceSession)  # вернуть класс после теста
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "id.onnx"
    onnx.save(model, str(path))

    monkeypatch.setenv("GUARD_ORT_SPINNING", "1")
    assert edge.disable_ort_spinning() is False
    monkeypatch.delenv("GUARD_ORT_SPINNING")
    assert edge.disable_ort_spinning() is True
    assert edge.disable_ort_spinning() is True  # повторный вызов не оборачивает класс второй раз
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    opts = sess.get_session_options()
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    out = sess.run(None, {"x": np.ones((1, 2), np.float32)})[0]
    assert out.tolist() == [[1.0, 1.0]]


def test_cgroup_readers(tmp_path):
    (tmp_path / "cpu.stat").write_text("usage_usec 2500000\nuser_usec 1\n")
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    (tmp_path / "memory.current").write_text("1048576\n")
    (tmp_path / "memory.max").write_text("max\n")
    assert cgroup_cpu_usage_s(tmp_path) == 2.5
    assert cgroup_cpu_limit(tmp_path) == 1.5
    assert cgroup_memory(tmp_path) == (1048576, None)
    sampler = CpuSampler(tmp_path)
    time.sleep(0.02)
    (tmp_path / "cpu.stat").write_text("usage_usec 2530000\n")
    s = sampler.sample()
    assert s["limit_cores"] == 1.5 and s["container_cores"] > 0 and s["container_pct_of_limit"] > 0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(tmp_path) is None


def _metrics(ts: float, processed: int, sent: int, offset_ms: float = 3600.0) -> MetricsMsg:
    return MetricsMsg(data={
        "backend": "linq-tpu",
        "cameras": {"cam_s1": {"fps_processed": 5.0, "fps_in": 25.0, "lag_s": 0.1, "timings_ms": {}}},
        "edge": {
            "ts": ts, "node_id": "n", "backend": "linq-tpu",
            "cpu": {"container_cores": 1.0, "container_pct_of_limit": 66.7, "process_cores": 0.9, "limit_cores": 1.5},
            "memory": {"process_rss_mb": 300.0},
            "clock": {"offset_ms": offset_ms, "uncertainty_ms": 10.0},
            "frames_processed": {"cam_s1": processed},
            "transport": {"frame_meta_by_camera": {"cam_s1": sent}, "held_by_guard": {}, "outbox_dropped": {}},
        },
    })


def test_probe_summary_clock_latency_and_loss():
    # часы ноутбука впереди часов стенда на 3.5 с (NTP: стенд +3600 мс, ноутбук +100 мс)
    d = 3.5
    rec = Recorder()
    rec.consume_started = 1000.0 + d
    # метрика прошлого запуска edge, пролежавшая в очереди: в FPS и потери не входит
    rec.add(_metrics(990.0, processed=5000, sent=5000), recv=1000.0 + d + 0.05, size=500)
    for i, ts in enumerate((1000.0, 1002.0, 1004.0)):
        rec.add(_metrics(ts, processed=10 * i, sent=10 * i), recv=ts + d + 0.12, size=500)
    # 18 из 20 frame_meta за окно 1000..1004
    for k in range(18):
        ts = 1000.1 + k * 0.2
        rec.add(FrameMetaMsg(camera_id="cam_s1", frame_idx=k, ts=ts, w=640, h=360), recv=ts + d + 0.3, size=150)
    ev = _event("live", started=1001.0, confirmed=1003.0)
    ev.details = {"edge": {"node": "n", "sent_at": 1003.1}}
    rec.add(EventMsg(kind="event", event=ev, snapshot_jpeg=b"j" * 2048), recv=1003.1 + d + 0.2, size=4096)
    old = _event("old", started=990.0, confirmed=992.0)
    old.details = {"edge": {"node": "n", "sent_at": 992.1}}  # ушло до подключения подписчика
    rec.add(EventMsg(kind="event", event=old), recv=1000.5 + d, size=1000)
    rec.add(EventMsg(kind="event", event=ev), recv=1003.5 + d, size=4096)  # повтор (at-least-once)

    args = argparse.Namespace(label="t", url="amqp://g:g@127.0.0.1:5672/", queue="q", prefetch=200, ack_every=50)
    out = summarize(rec, {"offset_ms": 100.0, "uncertainty_ms": 50.0}, {"rtt_min_ms": 240.0}, args)
    assert out["clock"]["used"] == "ntp"
    assert out["clock"]["laptop_minus_stand_used_ms"] == pytest.approx(3500.0)
    evs = out["events"]
    assert evs["count"] == 1 and evs["backlog_excluded"] == 1 and evs["duplicates"] == 1
    assert evs["e2e_from_started_s"]["p50"] == pytest.approx(2.3, abs=1e-3)
    assert evs["e2e_from_confirmed_s"]["p50"] == pytest.approx(0.3, abs=1e-3)
    assert evs["transport_s"]["p50"] == pytest.approx(0.2, abs=1e-3)
    cam = out["frame_meta"]["by_camera"]["cam_s1"]
    assert cam["processed_in_window"] == 20 and cam["received_in_window"] == 18
    assert cam["loss_vs_processed_pct"] == pytest.approx(10.0)
    assert out["frame_meta"]["latency_s"]["p50"] == pytest.approx(0.3, abs=1e-3)
    assert out["cpu"]["container_cores"]["mean"] == pytest.approx(1.0)
    assert out["probe"]["metrics_backlog_excluded"] == 1


def test_stats_helper():
    s = stats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and s["p50"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0
    assert stats([]) is None and stats([None]) is None
    assert math.isclose(stats([0.5], scale=1000)["p50"], 500.0)


def test_probe_bindings_match_transport():
    from guard import edge_probe
    from guard.transport import rabbit

    assert (edge_probe.EXCHANGE, edge_probe.GATEWAY_QUEUE) == (rabbit.EXCHANGE, rabbit.GATEWAY_QUEUE)
    assert edge_probe.BINDINGS == rabbit.BINDINGS


def test_probe_tap_uses_own_queue_and_leaves_gateway_queue(monkeypatch):
    """--tap: своя exclusive-очередь на ключах edge, очередь шлюза не трогается (ни purge, ни consume)."""
    import sys
    from unittest import mock

    from guard import edge_probe

    ch = mock.MagicMock()
    ch.queue_declare.return_value.method.queue = "amq.gen-tap"
    ch.consume.return_value = iter([(None, None, None)])
    pika = mock.MagicMock()
    pika.BlockingConnection.return_value.channel.return_value = ch
    monkeypatch.setitem(sys.modules, "pika", pika)

    edge_probe.consume("amqp://x", edge_probe.GATEWAY_QUEUE, 0.0, 200, 50, Recorder(), purge=True, tap=True)

    kwargs = ch.queue_declare.call_args.kwargs
    assert kwargs["queue"] == "" and kwargs["exclusive"] and kwargs["auto_delete"]
    bound = {c.kwargs["routing_key"] for c in ch.queue_bind.call_args_list}
    assert bound == set(edge_probe.BINDINGS)
    assert all(c.kwargs["queue"] == "amq.gen-tap" for c in ch.queue_bind.call_args_list)
    ch.queue_purge.assert_not_called()
    assert ch.consume.call_args.args[0] == "amq.gen-tap"
