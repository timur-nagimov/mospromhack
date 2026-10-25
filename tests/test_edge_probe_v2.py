"""edge_probe для topology v2: копия потока edge через свою очередь (--tap), разность часов из --clock-offset-ms."""

import base64
import io
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from guard import edge_probe
from guard.contracts import Event
from guard.transport.protocol import Envelope, encode, event_payload


def jpeg():
    out = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(out, format="JPEG")
    return out.getvalue()


def envelope(kind, payload, *, sequence=1, sent_at=1000.0, producer=None):
    producer = producer or str(uuid4())
    return Envelope(
        edge_id="stand-linq", producer_id=producer, run_id=str(uuid4()), sequence=sequence,
        message_id=f"{producer}:{sequence}", kind=kind, sent_at=float(sent_at), payload=payload,
    )


def violation(**overrides):
    return Event(**{
        "id": "ev-1", "camera_id": "cam_s1", "site": "Цех", "zone_id": None, "rule_id": "r", "type": "zone_intrusion",
        "severity": "high", "started_at": 996.0, "confirmed_at": 998.0, "bbox": (1, 2, 20, 23), **overrides,
    })


def preview(capture_ts=999.5):
    return {
        "camera_id": "cam_s2", "capture_id": str(uuid4()), "frame_seq": 7, "capture_ts": capture_ts, "source_pos_s": 1.0,
        "width": 32, "height": 24, "jpeg_b64": base64.b64encode(jpeg()).decode(),
        "people": [], "objects": [], "highlight": [], "zones": [],
    }


def test_recorder_maps_v2_envelopes():
    rec = edge_probe.Recorder()
    snapshot = jpeg()
    rec.add_envelope(envelope("event", event_payload(violation(), snapshot_jpeg=snapshot), sent_at=998.2), 1003.0, 5000)
    rec.add_envelope(envelope("event_update", event_payload(violation(ended_at=999.0)), sent_at=999.1), 1004.0, 4000)
    rec.add_envelope(envelope("event", event_payload(violation()), sent_at=998.2), 1005.0, 5000)  # повтор доставки
    rec.add_envelope(envelope("frame_meta", preview(), sent_at=999.6), 1004.5, 900)
    rec.add_envelope(envelope("metrics", {"data": {"backend": "linq-tpu", "cameras": {}}}, sent_at=1000.0), 1004.6, 300)
    event = rec.events["ev-1"]
    assert (event["started_at"], event["confirmed_at"], event["sent_at"], event["recv"]) == (996.0, 998.0, 998.2, 1003.0)
    assert event["snapshot_kb"] == round(len(snapshot) / 1024, 1)
    assert rec.updates == 1 and rec.event_duplicates == 1
    assert rec.frame_meta["cam_s2"] == [(999.5, 1004.5)]
    assert rec.metrics == [(1004.6, {"backend": "linq-tpu", "cameras": {}})]
    # часы — по телеметрии (frame_meta, metrics); события со снимком — отдельно, как запасной вариант
    assert rec.clock_samples == [pytest.approx(4.9), pytest.approx(4.6)]
    assert rec.event_clock_samples == [pytest.approx(4.8), pytest.approx(4.9)]
    assert dict(rec.counts) == {"event": 2, "event_update": 1, "frame_meta": 1, "metrics": 1}


def test_explicit_clock_offset_is_used_for_alert_latency():
    rec = edge_probe.Recorder()
    rec.consume_started = 1004.0  # часы ноутбука; по часам стенда — 999.5
    rec.add_envelope(envelope("event", event_payload(violation()), sent_at=998.2), 1004.2, 5000)
    live = violation(id="ev-2", started_at=999.0, confirmed_at=1000.8)
    rec.add_envelope(envelope("event", event_payload(live), sent_at=1000.9), 1006.0, 5000)
    args = edge_probe.parse_args(["--tap", "--clock-offset-ms", "4500"])
    result = edge_probe.summarize(rec, None, {"rtt_min_ms": 300.0}, args)
    assert result["clock"]["used"] == "cli"
    assert result["clock"]["laptop_minus_stand_used_ms"] == 4500.0
    # событие, отправленное до начала приёма (по часам стенда), — хвост, а не живая задержка
    assert result["events"]["backlog_excluded"] == 1 and result["events"]["count"] == 1
    assert result["events"]["e2e_from_started_s"]["p50"] == pytest.approx(1006.0 - 4.5 - 999.0)
    assert result["events"]["transport_s"]["p50"] == pytest.approx(1006.0 - 4.5 - 1000.9)


class FakeChannel:
    def __init__(self, deliveries):
        self.deliveries = deliveries
        self.calls = []

    def queue_declare(self, **kwargs):
        self.calls.append(("queue_declare", kwargs))
        return SimpleNamespace(method=SimpleNamespace(queue="amq.gen-tap"))

    def queue_bind(self, **kwargs):
        self.calls.append(("queue_bind", kwargs))

    def basic_qos(self, **kwargs):
        self.calls.append(("basic_qos", kwargs))

    def queue_purge(self, *args, **kwargs):
        raise AssertionError("tap never purges a shared queue")

    def consume(self, queue, auto_ack, inactivity_timeout):
        self.calls.append(("consume", {"queue": queue}))
        for index, body in enumerate(self.deliveries, start=1):
            yield SimpleNamespace(delivery_tag=index), SimpleNamespace(headers=None), body
        while True:
            yield None, None, None

    def basic_ack(self, **kwargs):
        self.calls.append(("basic_ack", kwargs))

    def cancel(self):
        pass


def test_v2_tap_binds_own_queue_to_edge_exchange(monkeypatch):
    import pika

    body = encode(envelope("event", event_payload(violation()), sent_at=998.2))
    channel = FakeChannel([body, b"not json"])
    connection = SimpleNamespace(channel=lambda: channel, close=lambda: None)
    monkeypatch.setattr(pika, "BlockingConnection", lambda params: connection)
    rec = edge_probe.Recorder()
    edge_probe.consume_v2("amqp://guest:guest@127.0.0.1:5672/", "stand-linq", 0.3, 50, 10, rec)
    declare = next(args for name, args in channel.calls if name == "queue_declare")
    assert declare["exclusive"] is True and declare["auto_delete"] is True and declare["queue"] == ""
    binds = [args for name, args in channel.calls if name == "queue_bind"]
    # По умолчанию — только события: копия телеметрии (~50–75 КиБ/с) удвоила бы нагрузку на туннель демо.
    assert binds == [{"queue": "amq.gen-tap", "exchange": "guard.edge.v1", "routing_key": "stand-linq.events"}]
    assert ("consume", {"queue": "amq.gen-tap"}) in channel.calls
    assert list(rec.events) == ["ev-1"] and rec.decode_errors == 1


def test_v2_tap_with_telemetry_binds_both_keys(monkeypatch):
    import pika

    channel = FakeChannel([])
    monkeypatch.setattr(pika, "BlockingConnection", lambda params: SimpleNamespace(channel=lambda: channel, close=lambda: None))
    edge_probe.consume_v2("amqp://localhost/", "stand-linq", 0.1, 50, 10, edge_probe.Recorder(), telemetry=True)
    keys = [args["routing_key"] for name, args in channel.calls if name == "queue_bind"]
    assert keys == ["stand-linq.events", "stand-linq.telemetry"]


def test_v2_without_tap_is_refused(capsys):
    # Очередь guard.gateway.<edge>.events принадлежит шлюзу: замер без --tap отнял бы у него события.
    assert edge_probe.main(["--duration", "1"]) == 2
    assert "--tap" in capsys.readouterr().err
