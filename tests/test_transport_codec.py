import json
import struct

import pytest
from pydantic import ValidationError

from guard.contracts import TIMING_KEYS, Event, Severity
from guard.transport.codec import decode, encode
from guard.transport.messages import (
    EventMsg,
    FrameMetaMsg,
    HeartbeatMsg,
    MetricsMsg,
)


def make_event():
    return Event(
        id="12345678-1234-5678-9012-123456789012",
        camera_id="camera-01",
        site="Цех № 1",
        zone_id="danger-zone",
        rule_id="ppe-rule",
        type="ppe_missing",
        severity=Severity.high,
        track_id=1234,
        started_at=1729730000.125,
        confirmed_at=1729730002.5,
        bbox=(100.5, 200.0, 300.25, 700.0),
        snapshot_path="media/events/12345678/snapshot.jpg",
        clip_path="media/events/12345678/clip.mp4",
        details={
            "missing": ["helmet", "vest"],
            "camera_name": "Камера у входа",
            "confirm_window_s": 2.375,
        },
        latency_ms={key: 12.34 for key in TIMING_KEYS},
    )


def make_frame():
    return FrameMetaMsg(
        camera_id="camera-01",
        frame_idx=123456,
        ts=1729730002.5,
        w=1920,
        h=1080,
        people=[
            [1000 + i, 100 + i, 200, 500 + i, 1000, 7]
            for i in range(10)
        ],
        objects=[
            [2001, "forklift", 100, 200, 600, 800],
            [2002, "truck", 1000, 200, 1800, 900],
        ],
        violators=[1001, 1004],
    )


@pytest.mark.parametrize("kind", ["event", "event_update"])
@pytest.mark.parametrize(
    "snapshot",
    [None, b"", b"\xff\xd8\x00\x80\xff\xfe\xff\xd9" * 100],
)
def test_event_roundtrip_and_binary_snapshot(kind, snapshot):
    message = EventMsg(
        kind=kind,
        event=make_event(),
        snapshot_jpeg=snapshot,
    )
    body, headers = encode(message)
    assert decode(body, headers) == message
    assert struct.unpack("!I", body[:4])[0] == headers["json_length"]

    json_end = 4 + headers["json_length"]
    document = json.loads(body[4:json_end])
    assert "snapshot_jpeg" not in document
    assert document["event"]["severity"] == "high"
    assert body[json_end:] == (snapshot or b"")
    assert headers["snapshot_length"] == (
        -1 if snapshot is None else len(snapshot)
    )
    assert len(body) == json_end + len(snapshot or b"")


@pytest.mark.parametrize(
    "message",
    [
        make_frame(),
        MetricsMsg(
            data={
                "backend": "linq-tpu",
                "cameras": {"cam": {"fps_processed": 5.0}},
                "inference": {"utilization": None},
            }
        ),
        HeartbeatMsg(ts=1729730000.5, node_id="edge-01"),
    ],
)
def test_other_message_roundtrips(message):
    body, headers = encode(message)
    assert decode(body, headers) == message
    assert headers["snapshot_length"] == -1


def test_frame_and_event_size_budgets():
    frame_body, frame_headers = encode(make_frame())
    event_body, event_headers = encode(EventMsg(event=make_event()))
    # Decimal KB, including binary prefix and a conservative JSON rendering
    # of the codec's AMQP headers.
    assert len(frame_body) + len(json.dumps(frame_headers).encode()) <= 1500
    assert len(event_body) + len(json.dumps(event_headers).encode()) <= 2500

    frame_json = frame_body[4:]
    assert b": " not in frame_json
    assert b", " not in frame_json
    assert isinstance(json.loads(frame_json)["people"][0], list)


def test_pipeline_dictionary_is_accepted():
    event = make_event()
    message = {"kind": "event_update", "event": event.model_dump(mode="json")}
    body, headers = encode(message)
    assert decode(body, headers) == EventMsg(kind="event_update", event=event)

    metrics = {"kind": "metrics", "data": {"backend": "linq-tpu"}}
    assert decode(*encode(metrics)) == MetricsMsg(data=metrics["data"])


@pytest.mark.parametrize(
    "body_change,header_change",
    [
        (lambda body: body[:2], {}),
        (lambda body: body[:-1], {}),
        (lambda body: body + b"x", {}),
        (lambda body: struct.pack("!I", 1) + body[4:], {}),
        (lambda body: body, {"guard_version": 2}),
        (lambda body: body, {"guard_version": True}),
        (lambda body: body, {"json_length": "123"}),
        (lambda body: body, {"snapshot_length": -2}),
    ],
)
def test_corrupt_envelopes_are_rejected(body_change, header_change):
    body, headers = encode(MetricsMsg(data={"fps": 5}))
    with pytest.raises(ValueError):
        decode(body_change(body), {**headers, **header_change})


def test_missing_headers_are_rejected():
    body, headers = encode(HeartbeatMsg(ts=1))
    for key in headers:
        incomplete = dict(headers)
        del incomplete[key]
        with pytest.raises(ValueError):
            decode(body, incomplete)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {"kind": "unknown"},
        {"kind": "event", "snapshot_jpeg": "base64-is-forbidden"},
        {"kind": "metrics", "data": {}, "unexpected": 1},
    ],
)
def test_invalid_documents_are_rejected(document):
    raw = json.dumps(document).encode()
    with pytest.raises(ValueError):
        decode(
            struct.pack("!I", len(raw)) + raw,
            {
                "guard_version": 1,
                "json_length": len(raw),
                "snapshot_length": -1,
            },
        )


def test_snapshot_on_non_event_is_rejected():
    body, headers = encode(MetricsMsg(data={}))
    with pytest.raises(ValueError):
        decode(body + b"jpeg", {**headers, "snapshot_length": 4})


@pytest.mark.parametrize(
    "row",
    [
        [1, 1.5, 2, 3, 4, 0],
        [1, True, 2, 3, 4, 0],
        [1, 1, 2, 3, 4],
    ],
)
def test_person_rows_require_six_integers(row):
    with pytest.raises(ValidationError):
        FrameMetaMsg(
            camera_id="cam",
            frame_idx=0,
            ts=1,
            w=1920,
            h=1080,
            people=[row],
        )


def test_nonfinite_json_is_rejected():
    # pydantic 2 (mode="json") сериализует NaN/Inf как null: на выходе всегда
    # валидный JSON, поток отправки не падает на метрике NaN.
    body, headers = encode(MetricsMsg(data={"fps": float("nan"), "lat": float("inf")}))
    assert b"NaN" not in body and b"Infinity" not in body
    assert decode(body, headers).data == {"fps": None, "lat": None}

    # Входящий нестандартный JSON с NaN отвергается.
    raw = b'{"kind":"metrics","data":{"fps":NaN}}'
    with pytest.raises(ValueError):
        decode(
            struct.pack("!I", len(raw)) + raw,
            {
                "guard_version": 1,
                "json_length": len(raw),
                "snapshot_length": -1,
            },
        )
