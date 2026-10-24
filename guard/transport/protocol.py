"""Bounded JSON wire protocol. This module never opens paths from messages."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from guard.contracts import CameraConfig, Event, Zone

MessageKind = Literal["event", "event_update", "frame_meta", "metrics", "cameras", "flywheel_sample",
                      "camera_config", "camera_config_ack"]
# camera_config / camera_config_ack — управляющий канал gateway → edge (docs/EDGE_PROTOCOL.md)
CONTROL_KINDS = frozenset({"camera_config", "camera_config_ack"})
MAX_MESSAGE_BYTES = 2 * 1024 * 1024
# Data-flywheel sample part (docs/FLYWHEEL.md): a JPEG chunk; part 0 also carries metadata.
MAX_FLYWHEEL_PARTS = 64
MAX_FLYWHEEL_CHUNK_BYTES = 64 * 1024
MAX_FLYWHEEL_DETECTIONS = 4096
FLYWHEEL_SAMPLE_FIELDS = frozenset({
    "id", "created_at", "camera_id", "source", "event_id", "reason", "holdout",
    "width", "height", "detections", "meta",
})
MAX_SNAPSHOT_BYTES = 1024 * 1024
MAX_FRAME_DIMENSION = 4096
MAX_FRAME_PIXELS = 16_777_216
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_SEQUENCE = 2**63 - 1
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def identifier(value: Any, name: str = "identifier") -> str:
    """Safe bounded identifiers for origins, cameras and event references."""
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def _uuid(value: str, name: str) -> None:
    try:
        if str(UUID(value)) != value:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ValueError(f"{name} must be a canonical UUID") from None


def _integer(value: Any, name: str, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}")


def _number(value: Any, name: str, *, minimum: float | None = None) -> None:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"Invalid {name}")
    if minimum is not None and value < minimum:
        raise ValueError(f"Invalid {name}")


def _json_tree(value: Any, depth: int = 0, budget: list[int] | None = None) -> None:
    """Reject non-JSON objects, huge trees and nonfinite numeric values."""
    if budget is None:
        budget = [MAX_JSON_NODES]
    budget[0] -= 1
    if depth > MAX_JSON_DEPTH or budget[0] < 0:
        raise ValueError("JSON payload is too complex")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_tree(item, depth + 1, budget)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_tree(item, depth + 1, budget)
        return
    raise ValueError("Payload must contain finite JSON values only")


def _jpeg(value: Any, name: str) -> bytes:
    if not isinstance(value, str) or len(value) > 4 * ((MAX_SNAPSHOT_BYTES + 2) // 3):
        raise ValueError(f"Invalid or oversized {name}")
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError(f"Invalid base64 {name}") from None
    if not 4 <= len(data) <= MAX_SNAPSHOT_BYTES or not (
        data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")
    ):
        raise ValueError(f"Invalid or oversized JPEG {name}")
    return data


def _event_payload(payload: dict[str, Any]) -> None:
    if set(payload) - {"event", "snapshot_jpeg"}:
        raise ValueError("Unknown event payload fields")
    raw_event = payload.get("event")
    if not isinstance(raw_event, dict) or set(raw_event) - Event.model_fields.keys():
        raise ValueError("event must be an Event object")
    event = Event.model_validate(raw_event)
    # Pydantic's JSON mode turns nonfinite floats into null; check Python
    # values first so coercion (e.g. latency='nan') cannot hide invalid data.
    _finite_event(event)
    for name in ("id", "camera_id", "rule_id", "type"):
        identifier(getattr(event, name), f"event.{name}")
    if event.zone_id is not None:
        identifier(event.zone_id, "event.zone_id")
    if len(event.site) > 512:
        raise ValueError("Event site is too long")
    for name in ("started_at", "confirmed_at", "ended_at"):
        value = getattr(event, name)
        if value is not None:
            _number(value, name, minimum=0)
    if event.confirmed_at < event.started_at or (
        event.ended_at is not None and event.ended_at < event.started_at
    ):
        raise ValueError("Invalid event time ordering")
    if len(event.bbox) != 4:
        raise ValueError("Invalid event bbox")
    for value in event.bbox:
        _number(value, "event.bbox")
    if event.bbox[2] < event.bbox[0] or event.bbox[3] < event.bbox[1]:
        raise ValueError("Invalid event bbox")
    if "snapshot_jpeg" in payload:
        _jpeg(payload["snapshot_jpeg"], "snapshot_jpeg")


def _frame_payload(payload: dict[str, Any]) -> None:
    required = {"camera_id", "capture_id", "frame_seq", "capture_ts", "source_pos_s",
                "width", "height", "jpeg_b64", "people", "objects", "highlight", "zones"}
    if set(payload) != required:
        raise ValueError("Invalid frame_meta fields")
    identifier(payload["camera_id"], "camera_id")
    _uuid(payload["capture_id"], "capture_id")
    _integer(payload["frame_seq"], "frame_seq", 0, MAX_SEQUENCE)
    _number(payload["capture_ts"], "capture_ts", minimum=0)
    if payload["source_pos_s"] is not None:
        _number(payload["source_pos_s"], "source_pos_s", minimum=0)
    for dimension in ("width", "height"):
        _integer(payload[dimension], dimension, 1, MAX_FRAME_DIMENSION)
    if payload["width"] * payload["height"] > MAX_FRAME_PIXELS:
        raise ValueError("Frame dimensions are too large")
    _jpeg(payload["jpeg_b64"], "jpeg_b64")
    for group in ("people", "objects"):
        if not isinstance(payload[group], list) or len(payload[group]) > 4096:
            raise ValueError(f"Invalid {group}")
        for track in payload[group]:
            if not isinstance(track, dict) or set(track) != {"track_id", "bbox", "cls", "conf"}:
                raise ValueError("Invalid frame track")
            _integer(track["track_id"], "track_id", 0, MAX_SEQUENCE)
            if not isinstance(track["cls"], str) or not 1 <= len(track["cls"]) <= 128:
                raise ValueError("Invalid track class")
            _number(track["conf"], "track.conf", minimum=0)
            if track["conf"] > 1:
                raise ValueError("Invalid track confidence")
            bbox = track["bbox"]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("Invalid track bbox")
            for value in bbox:
                _number(value, "track.bbox", minimum=0)
                if value > 1:
                    raise ValueError("Track bbox must be normalized")
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise ValueError("Invalid track bbox ordering")
    if not isinstance(payload["highlight"], list) or len(payload["highlight"]) > 8192:
        raise ValueError("Invalid highlighted tracks")
    for value in payload["highlight"]:
        _integer(value, "highlight", 0, MAX_SEQUENCE)
    if not isinstance(payload["zones"], list) or len(payload["zones"]) > 256:
        raise ValueError("Invalid zones")
    for raw in payload["zones"]:
        zone = Zone.model_validate(raw)
        identifier(zone.zone_id, "zone_id")
        if len(zone.polygon) > 4096:
            raise ValueError("Zone polygon is too large")
        for point in zone.polygon:
            if any(not 0 <= coordinate <= 1 for coordinate in point):
                raise ValueError("Zone polygon must be normalized")


def _flywheel_payload(payload: dict[str, Any]) -> None:
    """No paths are accepted: the gateway derives the frame name from sample_id."""
    if set(payload) != {"sample_id", "part", "parts", "sample", "jpeg_b64"}:
        raise ValueError("Invalid flywheel_sample fields")
    identifier(payload["sample_id"], "sample_id")
    _integer(payload["parts"], "parts", 1, MAX_FLYWHEEL_PARTS)
    _integer(payload["part"], "part", 0, payload["parts"] - 1)
    sample = payload["sample"]
    if payload["part"] > 0:
        if sample is not None:
            raise ValueError("Only the first flywheel part carries sample metadata")
    else:
        if not isinstance(sample, dict) or set(sample) - FLYWHEEL_SAMPLE_FIELDS:
            raise ValueError("Invalid flywheel sample metadata")
        if sample.get("id", payload["sample_id"]) != payload["sample_id"]:
            raise ValueError("Flywheel sample id mismatch")
        identifier(sample.get("camera_id"), "sample.camera_id")
        if sample.get("event_id") is not None:
            identifier(sample["event_id"], "sample.event_id")
        if not isinstance(sample.get("source"), str) or not 1 <= len(sample["source"]) <= 32:
            raise ValueError("Invalid flywheel sample source")
        for dimension in ("width", "height"):
            _integer(sample.get(dimension), f"sample.{dimension}", 1, MAX_FRAME_DIMENSION)
        if "created_at" in sample:
            _number(sample["created_at"], "sample.created_at", minimum=0)
        if not isinstance(sample.get("reason", ""), str) or len(sample.get("reason", "")) > 512:
            raise ValueError("Invalid flywheel sample reason")
        if not isinstance(sample.get("meta", {}), dict):
            raise ValueError("Invalid flywheel sample meta")
        detections = sample.get("detections", [])
        if not isinstance(detections, list) or len(detections) > MAX_FLYWHEEL_DETECTIONS:
            raise ValueError("Invalid flywheel sample detections")
        if any(not isinstance(item, dict) for item in detections):
            raise ValueError("Invalid flywheel sample detection")
    chunk = payload["jpeg_b64"]
    if not isinstance(chunk, str) or len(chunk) > 4 * ((MAX_FLYWHEEL_CHUNK_BYTES + 2) // 3):
        raise ValueError("Invalid or oversized flywheel chunk")
    try:
        data = base64.b64decode(chunk, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid base64 flywheel chunk") from None
    if not 1 <= len(data) <= MAX_FLYWHEEL_CHUNK_BYTES:
        raise ValueError("Invalid or oversized flywheel chunk")


class Envelope(BaseModel):
    """Version 1 envelope; validation also runs for messages built in Python."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal[1] = 1
    edge_id: str
    producer_id: str
    run_id: str
    message_id: str
    kind: MessageKind
    sequence: int = Field(ge=1, le=MAX_SEQUENCE)
    sent_at: float = Field(ge=0, allow_inf_nan=False)
    payload: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def validate_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            version = value.get("version", 1)
            if type(version) is not int or version != 1:
                raise ValueError("Unsupported protocol version")
        return value

    @model_validator(mode="after")
    def validate_message(self) -> Envelope:
        identifier(self.edge_id, "edge_id")
        _uuid(self.producer_id, "producer_id")
        _uuid(self.run_id, "run_id")
        if self.message_id != f"{self.producer_id}:{self.sequence}":
            raise ValueError("message_id must be producer_id:sequence")
        _json_tree(self.payload)
        if self.kind in {"event", "event_update"}:
            _event_payload(self.payload)
        elif self.kind == "frame_meta":
            _frame_payload(self.payload)
        elif self.kind == "flywheel_sample":
            _flywheel_payload(self.payload)
        elif self.kind == "metrics":
            if set(self.payload) != {"data"} or not isinstance(self.payload["data"], dict):
                raise ValueError("metrics payload must contain a data object")
        elif self.kind == "cameras":
            if set(self.payload) != {"cameras", "statuses"} or not isinstance(self.payload["cameras"], list):
                raise ValueError("Invalid cameras payload")
            if not isinstance(self.payload["statuses"], dict) or len(self.payload["cameras"]) > 256:
                raise ValueError("Invalid camera statuses")
            seen = set()
            for raw in self.payload["cameras"]:
                camera = CameraConfig.model_validate(raw)
                identifier(camera.camera_id, "camera_id")
                if camera.camera_id in seen:
                    raise ValueError("Duplicate camera_id")
                seen.add(camera.camera_id)
                if camera.source.uri != f"edge://{self.edge_id}/{camera.camera_id}":
                    raise ValueError("Camera source must be an edge reference")
            for camera_id in self.payload["statuses"]:
                identifier(camera_id, "camera status id")
        elif self.kind == "camera_config":
            _camera_config_payload(self.payload)
        elif self.kind == "camera_config_ack":
            _camera_config_ack_payload(self.payload)
        if len(_dump(self)) > MAX_MESSAGE_BYTES:
            raise ValueError("Message exceeds maximum size")
        return self


def _dump(envelope: Envelope) -> bytes:
    return json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False,
                      allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def encode(envelope: Envelope) -> bytes:
    """Validate again, including nested dictionaries changed after construction."""
    return _dump(Envelope.model_validate(envelope.model_dump()))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def decode(data: bytes | str, *, expected_edge_id: str | None = None) -> Envelope:
    """Reject oversized/unknown-version/malformed messages before application."""
    if not isinstance(data, (bytes, str)):
        raise ValueError("Message must be bytes or text")
    if len(data) > MAX_MESSAGE_BYTES or (isinstance(data, str) and len(data.encode("utf-8")) > MAX_MESSAGE_BYTES):
        raise ValueError("Message exceeds maximum size")
    try:
        raw = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError("Invalid JSON message") from None
    envelope = Envelope.model_validate(raw)
    if expected_edge_id is not None:
        identifier(expected_edge_id, "expected_edge_id")
        if envelope.edge_id != expected_edge_id:
            raise ValueError("Unexpected edge origin")
    return envelope


def event_payload(event: Event, *, snapshot_jpeg: bytes | None = None) -> dict[str, Any]:
    validated = Event.model_validate(event.model_dump())
    _finite_event(validated)
    payload = {"event": validated.model_dump(mode="json")}
    if snapshot_jpeg is not None:
        payload["snapshot_jpeg"] = base64.b64encode(snapshot_jpeg).decode("ascii")
    _json_tree(payload)
    _event_payload(payload)
    return payload


def _finite_event(event: Event) -> None:
    try:
        json.dumps(event.model_dump(), allow_nan=False)
    except (ValueError, TypeError, OverflowError):
        raise ValueError("Event must contain finite JSON values only") from None


def snapshot_bytes(envelope: Envelope) -> bytes | None:
    """Return validated JPEG bytes, never a remote filesystem path."""
    value = envelope.payload.get("snapshot_jpeg")
    return None if value is None else _jpeg(value, "snapshot_jpeg")


# ---------------------------------------------------------------- управляющий канал gateway → edge

MAX_CONFIG_ERROR = 1000
MAX_SOURCE_URI = 2048


def _camera_config_payload(payload: dict[str, Any]) -> None:
    """Желаемая конфигурация одной камеры: upsert с полным CameraConfig или delete без него.

    source.uri равен edge://<edge_id>/<camera_id>, если источник не меняется: шлюз не знает
    настоящий URI, edge подставляет свой. ID зон и правил — те же идентификаторы, что в событиях."""
    if set(payload) != {"camera_id", "version", "op", "config"}:
        raise ValueError("Invalid camera_config fields")
    camera_id = identifier(payload["camera_id"], "camera_id")
    _integer(payload["version"], "config version", 1, MAX_SEQUENCE)
    if payload["op"] == "delete":
        if payload["config"] is not None:
            raise ValueError("camera_config delete carries no config")
        return
    if payload["op"] != "upsert" or not isinstance(payload["config"], dict):
        raise ValueError("Invalid camera_config operation")
    camera = CameraConfig.model_validate(payload["config"])
    if camera.camera_id != camera_id:
        raise ValueError("camera_config identity mismatch")
    if not camera.source.uri or len(camera.source.uri) > MAX_SOURCE_URI:
        raise ValueError("Invalid camera source uri")
    if len(camera.zones) > 256 or len(camera.rules) > 256:
        raise ValueError("Too many zones or rules")
    for zone in camera.zones:
        identifier(zone.zone_id, "zone_id")
        if len(zone.polygon) > 4096:
            raise ValueError("Zone polygon is too large")
    for rule in camera.rules:
        identifier(rule.rule_id, "rule_id")
        identifier(rule.type, "rule type")


def _camera_config_ack_payload(payload: dict[str, Any]) -> None:
    """Ответ edge на camera_config: ok и текст ошибки; current_version — действующая версия камеры."""
    if set(payload) != {"camera_id", "version", "ok", "error", "current_version"}:
        raise ValueError("Invalid camera_config_ack fields")
    identifier(payload["camera_id"], "camera_id")
    _integer(payload["version"], "config version", 1, MAX_SEQUENCE)
    _integer(payload["current_version"], "current config version", 0, MAX_SEQUENCE)
    if type(payload["ok"]) is not bool:
        raise ValueError("Invalid camera_config_ack ok flag")
    error = payload["error"]
    if payload["ok"]:
        if error is not None:
            raise ValueError("Successful camera_config_ack has no error")
    elif not isinstance(error, str) or not 1 <= len(error) <= MAX_CONFIG_ERROR:
        raise ValueError("Failed camera_config_ack requires an error text")
