"""Fail-closed source identity and atomic cache publication; no network or ffmpeg."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import pytest

from tools import fetch_data


def source(payload: bytes, **overrides) -> dict:
    return {
        "url": "https://example.invalid/source.mp4",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        **overrides,
    }


def forbidden(*args, **kwargs):
    raise AssertionError("network and ffmpeg must not run in this test")


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(fetch_data, "ffmpeg_exe", forbidden)
    monkeypatch.setattr(fetch_data.subprocess, "run", forbidden)
    monkeypatch.setattr(fetch_data.time, "sleep", lambda seconds: None)


def cached(data: Path, payload: bytes) -> Path:
    path = data / "raw" / "clip.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def test_valid_cached_source_is_reverified_without_network(tmp_path):
    body = b"verified video"
    path = cached(tmp_path, body)
    assert fetch_data.download(tmp_path, "clip", source(body)) == path
    assert path.read_bytes() == body


@pytest.mark.parametrize("body,pins,reason", [
    (b"tampered", source(b"expected"), "sha256"),
    (b"short", source(b"expected"), "размер"),
    (b"", source(b"expected"), "размер"),
])
def test_corrupt_cache_fails_without_network_or_deletion(tmp_path, body, pins, reason):
    path = cached(tmp_path, body)
    with pytest.raises(fetch_data.SourceIntegrityError, match=reason):
        fetch_data.download(tmp_path, "clip", pins)
    assert path.read_bytes() == body
    assert list(path.parent.iterdir()) == [path]


def test_new_source_is_not_published_until_verified(monkeypatch, tmp_path):
    body = b"verified video"
    final = fetch_data.raw_path(tmp_path, "clip", source(body))

    def response(req, timeout):
        assert req.full_url == "https://example.invalid/source.mp4" and timeout == 120
        assert not final.exists()
        return io.BytesIO(body)

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", response)
    assert fetch_data.download(tmp_path, "clip", source(body)) == final
    assert final.read_bytes() == body
    assert list(final.parent.iterdir()) == [final]


@pytest.mark.parametrize("valid", [True, False])
def test_ytdlp_source_is_verified_before_atomic_publication(monkeypatch, tmp_path, valid):
    expected = b"verified video"
    pins = source(expected, url="https://youtu.be/synthetic")
    final = fetch_data.raw_path(tmp_path, "clip", pins)
    calls = []

    def download(pending, spec):
        assert spec == pins and pending != final and not final.exists()
        calls.append(pending)
        pending.write_bytes(expected if valid else b"tampered video")

    monkeypatch.setattr(fetch_data, "download_ytdlp", download)
    if valid:
        assert fetch_data.download(tmp_path, "clip", pins) == final
        assert final.read_bytes() == expected
    else:
        with pytest.raises(fetch_data.SourceIntegrityError, match="sha256"):
            fetch_data.download(tmp_path, "clip", pins)
        assert not final.exists()
    assert len(calls) == 1
    assert sorted(final.parent.iterdir()) == ([final] if valid else [])


@pytest.mark.parametrize("body,reason", [(b"tampered", "sha256"), (b"short", "размер")])
def test_new_source_with_wrong_hash_or_size_is_never_published(monkeypatch, tmp_path, body, reason):
    requests = []
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: requests.append(True) or io.BytesIO(body))
    with pytest.raises(fetch_data.SourceIntegrityError, match=reason):
        fetch_data.download(tmp_path, "clip", source(b"expected"))
    assert requests == [True]  # Integrity errors do not retry a changed source.
    assert list((tmp_path / "raw").iterdir()) == []


@pytest.mark.parametrize("existing", [b"expected", b"bad cached file"])
def test_failed_download_preserves_cache_published_while_request_in_flight(monkeypatch, tmp_path, existing):
    pins = source(b"expected")
    final = fetch_data.raw_path(tmp_path, "clip", pins)

    def response(*args, **kwargs):
        # Another process publishes a cache while this request is in flight.
        final.write_bytes(existing)
        return io.BytesIO(b"tampered")

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", response)
    with pytest.raises(fetch_data.SourceIntegrityError, match="sha256"):
        fetch_data.download(tmp_path, "clip", pins)
    assert final.read_bytes() == existing
    assert list(final.parent.iterdir()) == [final]


@pytest.mark.parametrize("sha_pin,size_pin", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("use_cache", [False, True])
def test_legacy_sources_accept_missing_hash_or_size_and_check_available_pins(monkeypatch, tmp_path, sha_pin, size_pin, use_cache):
    body = b"legacy video"
    pins = source(body)
    if not sha_pin:
        pins.pop("sha256")
    if not size_pin:
        pins.pop("bytes")
    if use_cache:
        path = cached(tmp_path, body)
    else:
        monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
        path = fetch_data.raw_path(tmp_path, "clip", pins)
    assert fetch_data.download(tmp_path, "clip", pins) == path
    assert path.read_bytes() == body


def test_size_only_pin_still_rejects_wrong_size(monkeypatch, tmp_path):
    pins = {"url": "https://example.invalid/source.mp4", "bytes": 8}
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"short"))
    with pytest.raises(fetch_data.SourceIntegrityError, match="размер"):
        fetch_data.download(tmp_path, "clip", pins)
    assert list((tmp_path / "raw").iterdir()) == []


def test_empty_legacy_cache_can_be_downloaded_again(monkeypatch, tmp_path):
    path = cached(tmp_path, b"")
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"video"))
    assert fetch_data.download(tmp_path, "clip", {"url": "https://example.invalid/source.mp4"}) == path
    assert path.read_bytes() == b"video"


def test_empty_new_unpinned_source_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b""))
    with pytest.raises(fetch_data.SourceIntegrityError, match="пустой"):
        fetch_data.download(tmp_path, "clip", {"url": "https://example.invalid/source.mp4"})
    assert list((tmp_path / "raw").iterdir()) == []


def test_transport_failure_cleans_partial_files_before_retry(monkeypatch, tmp_path):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(size)

    body = b"verified video"
    responses = iter([Interrupted(b"partial"), io.BytesIO(body)])
    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", lambda *a, **k: next(responses))
    path = fetch_data.download(tmp_path, "clip", source(body), retries=2)
    assert path.read_bytes() == body
    assert list(path.parent.iterdir()) == [path]


def test_exhausted_transport_failure_leaves_no_published_or_partial_file(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise OSError("connection interrupted")

    monkeypatch.setattr(fetch_data.urllib.request, "urlopen", fail)
    with pytest.raises(OSError, match="connection interrupted"):
        fetch_data.download(tmp_path, "clip", source(b"expected"), retries=2)
    assert list((tmp_path / "raw").iterdir()) == []


def encode_fixture(data: Path) -> tuple[dict, dict, Path]:
    body = b"verified video"
    cached(data, body)
    manifest = {
        "sources": {"clip": source(body)},
        "encode": {"width": 1280, "height": 720, "fps": 25, "preset": "medium", "crf": 28},
    }
    video = {"name": "s2_01.mp4", "segments": [{"source": "clip", "ss": 0, "to": 60}]}
    output = data / "videos" / video["name"]
    output.parent.mkdir()
    output.write_bytes(b"existing encoded output")
    return manifest, video, output


def test_encode_skip_keeps_existing_cli_semantics_without_raw_inputs(tmp_path):
    manifest, video, output = encode_fixture(tmp_path)
    fetch_data.raw_path(tmp_path, "clip", manifest["sources"]["clip"]).unlink()
    assert fetch_data.encode(tmp_path, video, manifest, None, False) == output
    assert output.read_bytes() == b"existing encoded output"


def test_encode_only_force_rechecks_raw_pins_before_ffmpeg_and_preserves_output(tmp_path):
    manifest, video, output = encode_fixture(tmp_path)
    raw = fetch_data.raw_path(tmp_path, "clip", manifest["sources"]["clip"])
    raw.write_bytes(b"corrupted data")  # Same length: SHA must be checked.
    manifest["videos"] = [video]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(fetch_data.SourceIntegrityError, match="sha256"):
        fetch_data.main(["--manifest", str(path), "--data-dir", str(tmp_path), "--only", "s2_01", "--steps", "encode", "--force"])
    assert raw.read_bytes() == b"corrupted data"
    assert output.read_bytes() == b"existing encoded output"


def test_forced_encoding_process_failure_does_not_replace_existing_output(monkeypatch, tmp_path):
    manifest, video, output = encode_fixture(tmp_path)
    monkeypatch.setattr(fetch_data, "ffmpeg_exe", lambda: "ffmpeg-test-only")

    def fail(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"incomplete encoded output")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(fetch_data.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        fetch_data.encode(tmp_path, video, manifest, None, True)
    assert output.read_bytes() == b"existing encoded output"
    assert fetch_data.raw_path(tmp_path, "clip", manifest["sources"]["clip"]).read_bytes() == b"verified video"


def test_all_concat_sources_are_verified_before_ffmpeg(tmp_path):
    manifest, video, output = encode_fixture(tmp_path)
    manifest["sources"]["second"] = source(b"expected")
    second = fetch_data.raw_path(tmp_path, "second", manifest["sources"]["second"])
    second.write_bytes(b"tampered")
    video["segments"].append({"source": "second"})
    with pytest.raises(fetch_data.SourceIntegrityError, match="second: sha256"):
        fetch_data.encode(tmp_path, video, manifest, None, True)
    assert output.read_bytes() == b"existing encoded output"


def test_s2_and_clean_recipes_reference_exactly_five_pinned_sources():
    manifest = json.loads(fetch_data.MANIFEST.read_text())
    expected = {
        "intel_safety_gear": ("a0ba636766524dd0bdfa52a2a62108aafb585acd138cb0e08226d12ae35b64c5", 27534166),
        "mixkit_23237": ("e72baa23fb899ff9cfb5feeef01af167e68f3696af7b89b6fea2a28dabadcff2", 6721285),
        "mixkit_24049": ("36172448072224fef3db87b8ecf668e75a7613c2dcae13ec3298e97b47dbbb71", 5261599),
        "mixkit_23107": ("09bcf24f9dcc4156e6232a66db0262ba30c44ad32b13f35f07f3ce02aea449ac", 10382381),
        "mixkit_23224": ("d95a5150cd66389978c3f9353ad53f07f2f1f1b8bc6177f9ae863cf3d7eefef3", 10761089),
    }
    videos = [v for v in manifest["videos"] if v["name"] in {"s2_01.mp4", "s2_02.mp4", "clean_01.mp4", "clean_02.mp4"}]
    assert len(videos) == 4
    assert {seg["source"] for v in videos for seg in v["segments"]} == set(expected)
    assert {sid: (manifest["sources"][sid]["sha256"], manifest["sources"][sid]["bytes"]) for sid in expected} == expected
    events = [event for video in videos for event in video.get("events", [])]
    assert sum(not event.get("uncertain", False) for event in events) == 6
    assert sum(event.get("uncertain", False) for event in events) == 2
