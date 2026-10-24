"""Version 1 wire format: uint32 JSON length (network order), JSON, raw JPEG.

AMQP headers:
    guard_version: 1
    json_length: number of UTF-8 JSON bytes, also stored in the body prefix
    snapshot_length: -1 for None; otherwise the exact JPEG byte count

The distinction between None and an empty byte string survives a roundtrip.
The snapshot_jpeg field is omitted from JSON entirely.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from typing import Any

from .messages import EventMsg, MessageInput, Msg, parse_message

_PREFIX = struct.Struct("!I")
_VERSION = 1


def encode(message: MessageInput) -> tuple[bytes, dict[str, Any]]:
    msg = parse_message(message)
    snapshot = msg.snapshot_jpeg if isinstance(msg, EventMsg) else None
    document = msg.model_dump(mode="json", exclude={"snapshot_jpeg"})
    raw_json = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(raw_json) > 0xFFFFFFFF:
        raise ValueError("JSON exceeds the wire format length limit")
    headers = {
        "guard_version": _VERSION,
        "json_length": len(raw_json),
        "snapshot_length": -1 if snapshot is None else len(snapshot),
    }
    return _PREFIX.pack(len(raw_json)) + raw_json + (snapshot or b""), headers


def _integer_header(headers: Mapping[str, Any], name: str) -> int:
    value = headers.get(name)
    if type(value) is not int:
        raise ValueError(f"Missing or invalid {name} header")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def decode(body: bytes, headers: Mapping[str, Any]) -> Msg:
    if not isinstance(body, bytes):
        raise ValueError("Message body must be bytes")
    if not isinstance(headers, Mapping):
        raise ValueError("Message headers must be a mapping")
    if _integer_header(headers, "guard_version") != _VERSION:
        raise ValueError("Unsupported transport version")

    json_length = _integer_header(headers, "json_length")
    snapshot_length = _integer_header(headers, "snapshot_length")
    if json_length <= 0 or snapshot_length < -1:
        raise ValueError("Invalid payload lengths")
    if len(body) < _PREFIX.size:
        raise ValueError("Truncated message prefix")
    if _PREFIX.unpack_from(body)[0] != json_length:
        raise ValueError("JSON length header and prefix disagree")

    json_end = _PREFIX.size + json_length
    if len(body) != json_end + max(snapshot_length, 0):
        raise ValueError("Truncated message or unexpected trailing bytes")
    document = json.loads(
        body[_PREFIX.size:json_end],
        parse_constant=_reject_constant,
    )
    if not isinstance(document, dict):
        raise ValueError("JSON payload must be an object")
    if "snapshot_jpeg" in document:
        raise ValueError("Snapshots must be carried outside JSON")

    if document.get("kind") in ("event", "event_update"):
        document["snapshot_jpeg"] = (
            None if snapshot_length == -1 else body[json_end:]
        )
    elif snapshot_length != -1:
        raise ValueError("Only event messages may carry a snapshot")

    return parse_message(document)
