"""Wire validation must happen before a gateway can touch storage/media."""

import base64
import json
from uuid import uuid4

import pytest

from guard.contracts import CameraConfig, Event, Severity, SourceConfig
from guard.transport.protocol import (
    MAX_MESSAGE_BYTES, MAX_SNAPSHOT_BYTES, Envelope, decode, encode,
    event_payload, snapshot_bytes,
)

JPEG = b"\xff\xd8test-jpeg\xff\xd9"


def make_event(**changes):
    data = dict(id="event-1", camera_id="camera-1", site="Цех", rule_id="s1",
                type="zone_intrusion", severity=Severity.critical, started_at=100.,
                confirmed_at=102., bbox=(1., 2., 30., 40.))
    data.update(changes)
    return Event(**data)


def message(kind="event", payload=None, **changes):
    producer = str(uuid4())
    data = dict(edge_id="edge-1", producer_id=producer, run_id=str(uuid4()),
                message_id=f"{producer}:1", kind=kind, sequence=1, sent_at=100.,
                payload=event_payload(make_event()) if payload is None else payload)
    data.update(changes)
    return Envelope(**data)


def frame_payload(**changes):
    data = dict(camera_id="camera-1", capture_id=str(uuid4()), frame_seq=0,
                capture_ts=100., source_pos_s=0., width=640, height=480,
                jpeg_b64=base64.b64encode(JPEG).decode("ascii"),
                people=[dict(track_id=1, bbox=[0., 0., .5, 1.], cls="person", conf=.9)],
                objects=[], highlight=[1], zones=[])
    data.update(changes)
    return data


def test_event_roundtrip_and_snapshot_bytes():
    envelope = message(payload=event_payload(make_event(), snapshot_jpeg=JPEG))
    wire = encode(envelope)
    assert isinstance(wire, bytes)
    assert decode(wire, expected_edge_id="edge-1") == envelope
    assert decode(wire.decode()) == envelope
    assert snapshot_bytes(envelope) == JPEG
    assert snapshot_bytes(message()) is None


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"kind": "execute"}, {"sequence": 0}, {"sequence": True},
    {"sequence": 1.5}, {"sent_at": float("nan")}, {"sent_at": float("inf")},
    {"edge_id": "../edge"}, {"producer_id": "unknown"}, {"run_id": "unknown"},
    {"message_id": "arbitrary"}, {"extra": "forbidden"},
])
def test_reject_invalid_envelope(change):
    with pytest.raises(ValueError):
        message(**change)


def test_origin_validation():
    with pytest.raises(ValueError, match="origin"):
        decode(encode(message()), expected_edge_id="another-edge")


@pytest.mark.parametrize("raw", [b"\xff", b"{}", b"[]", b"null", b"not json", b'{"version":1,"version":1}'])
def test_malformed_wire_and_duplicate_keys(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_raw_and_encoded_message_size_limits():
    with pytest.raises(ValueError, match="size"):
        decode(b" " * (MAX_MESSAGE_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        message("metrics", {"data": {"padding": "я" * MAX_MESSAGE_BYTES}})
    with pytest.raises(ValueError, match="size"):
        decode("я" * MAX_MESSAGE_BYTES)


def test_mutating_nested_payload_cannot_bypass_validation():
    original = message()
    original.payload["event"]["camera_id"] = "../../outside"
    with pytest.raises(ValueError):
        encode(original)
    bypassed = message().model_copy(update={"version": 100})
    with pytest.raises(ValueError):
        encode(bypassed)


@pytest.mark.parametrize("changes", [
    {"camera_id": "../../camera"}, {"id": "/tmp/event"}, {"rule_id": ""},
    {"bbox": [0, 0, float("nan"), 10]}, {"bbox": [10, 0, 0, 10]},
    {"started_at": -1}, {"confirmed_at": 99}, {"ended_at": 99},
    {"details": {"bad": float("inf")}},
])
def test_event_schema_and_identity_fields_validated(changes):
    raw = make_event().model_dump(mode="json")
    raw.update(changes)
    with pytest.raises(ValueError):
        message(payload={"event": raw})


def test_event_payload_cannot_silently_convert_nonfinite_values_to_null():
    with pytest.raises(ValueError, match="finite"):
        event_payload(make_event(details={"bad": float("inf")}))
    with pytest.raises(ValueError, match="finite"):
        message(payload={"event": {**make_event().model_dump(mode="json"), "latency_ms": {"infer": "nan"}}})


@pytest.mark.parametrize("value", ["not base64!", base64.b64encode(b"not jpeg").decode(),
                                    base64.b64encode(b"\xff\xd8" + b"a" * MAX_SNAPSHOT_BYTES + b"\xff\xd9").decode()])
def test_snapshot_encoding_and_size(value):
    payload = event_payload(make_event())
    payload["snapshot_jpeg"] = value
    with pytest.raises(ValueError):
        message(payload=payload)


def test_telemetry_roundtrips():
    frame = message("frame_meta", frame_payload())
    assert decode(encode(frame)) == frame
    metrics = message("metrics", {"data": {"cameras": {"camera-1": {"infer": 10.5}}, "ready": True}})
    assert decode(encode(metrics)) == metrics
    camera = CameraConfig(camera_id="camera-1", name="Камера", site="Цех",
                          source=SourceConfig(type="rtsp", uri="edge://edge-1/camera-1"))
    cameras = message("cameras", {"cameras": [camera.model_dump(mode="json")],
                                   "statuses": {"camera-1": {"running": True}}})
    assert decode(encode(cameras)) == cameras


@pytest.mark.parametrize("changes", [
    {"capture_id": "not-uuid"}, {"frame_seq": -1}, {"width": 4097},
    {"height": True}, {"width": 0}, {"capture_ts": float("inf")},
    {"source_pos_s": -1}, {"jpeg_b64": ""}, {"camera_id": "../x"},
    {"highlight": ["1"]}, {"people": [{"track_id": 1, "bbox": [0, 0, 2, 1], "cls": "person", "conf": .5}]},
    {"objects": [{"track_id": 1, "bbox": [0, 0, 1, 1], "cls": "forklift", "conf": 1.1}]},
    {"zones": [{"zone_id": "z", "name": "Z", "polygon": [[2, 1]]}]},
])
def test_frame_meta_bounds(changes):
    with pytest.raises(ValueError):
        message("frame_meta", frame_payload(**changes))


def test_source_uri_cannot_escape_edge_reference():
    camera = CameraConfig(camera_id="camera-1", name="Камера", site="Цех",
                          source=SourceConfig(type="rtsp", uri="rtsp://user:secret@camera"))
    with pytest.raises(ValueError, match="edge reference"):
        message("cameras", {"cameras": [camera.model_dump(mode="json")], "statuses": {}})


@pytest.mark.parametrize("payload", [{"data": []}, {"data": {}, "extra": 1}, {"other": {}}])
def test_metrics_shape(payload):
    with pytest.raises(ValueError):
        message("metrics", payload)


def test_json_complexity_and_non_json_objects():
    nested = {}
    for _ in range(40):
        nested = {"nested": nested}
    for invalid in (nested, {"items": list(range(100001))}, {"nan": float("nan")}, {"tuple": (1, 2)}, {"key": {1: "bad"}}):
        with pytest.raises(ValueError):
            message("metrics", {"data": invalid})


def test_unknown_event_payload_fields_rejected():
    payload = event_payload(make_event())
    payload["command"] = "ignored"
    with pytest.raises(ValueError):
        message(payload=payload)
