"""Shared, model-free source-boundary proof and durable capture closure for S2.

Only an explicit pinned encoder-part proof can split a capture. GT times never
select the boundary. Both cached and live evaluation use these same checks.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from ml.eval import compare_s1_models as audit, run_s1

require, fingerprint = audit.require, audit.fingerprint


def load_boundaries(path: Path, sha256: str, *, assets: dict, encoded: dict,
                    segments: dict[str, list], samples: dict[str, list]) -> dict[str, dict]:
    """Accept only independently evidenced integer encoder-part cuts, never GT times."""
    require(re.fullmatch(r"[a-f0-9]{64}", sha256 or "") is not None, "Explicit boundary-map SHA256 is required")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == sha256, "Boundary-map SHA256 mismatch")
    def bad_number(value):
        raise ValueError(f"Non-finite boundary-map number: {value}")
    document = json.loads(payload, parse_constant=bad_number)
    proofs = document.get("boundaries", [document])
    require(document["schema_version"] == 1 and proofs, "Missing boundary proofs")
    result = {}
    for proof in proofs:
        video = proof["video"]
        require(proof["status"] == "verified" and proof["boundary_basis"] == "encoded_filter_frame_count"
                and video in samples and video not in result, "Unsupported/unverified/duplicate source boundary")
        prepared, boundary = encoded["outputs"][video], proof["boundary"]
        encoder_sha = hashlib.sha256(json.dumps(encoded["encoder"], sort_keys=True, separators=(",", ":"),
                                               ensure_ascii=False).encode()).hexdigest()
        require(proof["encoder"] == encoded["encoder"] and proof["encoder_canonical_json_sha256"] == encoder_sha
                and proof["ffmpeg_version"].splitlines()[0] == encoded["encoder"]["ffmpeg_version"], "Boundary encoder drift")
        for proof_key, asset_key in (("encoded_video", video), ("source_manifest", "manifest"), ("encode_report", "encode_report")):
            for key in ("sha256", "bytes"):
                require(proof[proof_key][key] == assets[asset_key][key], "Boundary source/prepared asset drift")
        source_segments = segments[video]
        require(len(source_segments) == 2 and set(proof["sources"]) == {segment["source"] for segment in source_segments},
                "Boundary needs the two actual source segments")
        for source_id, source in proof["sources"].items():
            require(all(source[key] == assets[f"source:{source_id}"][key] for key in ("sha256", "bytes")),
                    "Boundary raw source fingerprint drift")
        parts = proof["parts"]
        counts = [parts[name]["frame_count"] for name in ("part0", "part1")]
        cut = boundary["cut_before_source_frame"]
        require(all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in counts)
                and isinstance(cut, int) and not isinstance(cut, bool) and cut == counts[0]
                and sum(counts) == parts["joined"]["frame_count"] == prepared["frames"]
                and boundary["indexing"] == "zero-based" and boundary["cut_time_s"] == cut / prepared["fps"],
                "Boundary must be an exact integer encoder-part cut, not an annotation time")
        joined = {row["frame_index"]: row for row in proof["joined_boundary_frames"]}
        require(joined[cut - 1]["showinfo_checksum"] == parts["part0"]["last"]["showinfo_checksum"]
                and joined[cut]["showinfo_checksum"] == parts["part1"]["first"]["showinfo_checksum"]
                and proof["boundary_checks"] and all(value is True for value in proof["boundary_checks"].values()),
                "Boundary filter-frame continuity checks failed")
        old = [row for row in samples[video] if row["source_frame_index"] < cut]
        new = [row for row in samples[video] if row["source_frame_index"] >= cut]
        coords = lambda row: {key: row[key] for key in ("sample_index", "source_frame_index", "source_time_s")}
        require(old and new and coords(old[-1]) == boundary["last_old_selected_sample"]
                and coords(new[0]) == boundary["first_new_selected_sample"], "Boundary cache/sample coordinates differ")
        result[video] = {**boundary, "basis": proof["boundary_basis"], "evidence_sha256": fingerprint(proof)}
    return result



def effective_capture_id(source_capture_id: str, boundary: dict | None, segment_index: int) -> str:
    if boundary is None:
        return source_capture_id
    require(segment_index in (0, 1), "Only the two evidenced source segments are supported")
    return f"{source_capture_id}:source-segment:{segment_index}:cut:{boundary['cut_before_source_frame']}"


def close_capture(journal, manager, camera_id: str) -> list:
    """Read back every measured closure before allowing any temporal-state reset."""
    closed = manager.capture_closures(camera_id)
    run_s1.persist_lifecycle_updates(journal, closed, [])
    for event in closed:
        persisted = journal.get(event.id)
        require(event.ended_at is not None and math.isfinite(event.ended_at)
                and event.ended_at >= event.started_at and persisted is not None
                and persisted.ended_at == event.ended_at
                and all(persisted.details.get(key) == event.details[key] for key in
                        ("reopened_count", "active_duration_s") if key in event.details),
                "Boundary incident closure was not durably read back")
    manager.reset(camera_id)
    return closed
