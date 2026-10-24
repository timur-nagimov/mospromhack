import base64
import io

import pytest
from PIL import Image

from guard.gateway.frames import RemoteCameraView


def payload():
    output = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(output, format="JPEG")
    return {
        "camera_id": "cam", "width": 64, "height": 48,
        "capture_id": "capture-1", "frame_seq": 7, "capture_ts": 100.0, "source_pos_s": 2.5,
        "jpeg_b64": base64.b64encode(output.getvalue()).decode(),
        "people": [{"track_id": 3, "bbox": [0.1, 0.1, 0.8, 0.9]}],
        "objects": [], "highlight": [3],
        "zones": [{"polygon": [[0, 0], [1, 0], [1, 1]]}],
    }


def test_raw_and_overlay_share_frame_identity_then_expire():
    now = [10.0]
    camera = RemoteCameraView("cam", stale_after_s=2, clock=lambda: now[0], wall_clock=lambda: 100)
    data = payload()
    assert camera.update(data, sequence=42, run_id="run", producer_id="producer")
    data["capture_id"] = "mutated"  # accepted observation is detached from caller
    assert camera.frame_jpeg() == base64.b64decode(data["jpeg_b64"])
    identity, overlay = camera.overlay_jpeg()
    assert identity == ("producer", "run", 42, "analyzed")
    assert overlay != camera.frame_jpeg()
    with Image.open(io.BytesIO(overlay)) as image:
        assert image.size == (64, 48)
    assert camera.status()["frame_identity"]["capture_id"] == "capture-1"
    assert camera.status()["frame_identity"]["source_pos_s"] == 2.5
    now[0] = 12.1
    assert camera.frame_jpeg() is None
    assert camera.overlay_jpeg() is None
    assert camera.status()["running"] is False
    assert camera.status()["stale"] is True


def test_old_frame_cannot_restore_previous_run():
    camera = RemoteCameraView("cam", wall_clock=lambda: 100)
    assert camera.update(payload(), sequence=20, run_id="new", producer_id="producer")
    assert not camera.update(payload(), sequence=10, run_id="old", producer_id="producer")
    assert camera.status()["frame_identity"]["run_id"] == "new"
    camera.invalidate()
    assert not camera.running


@pytest.mark.parametrize("change", [
    {"jpeg_b64": "not-valid-base64!"}, {"width": 32}, {"height": 99999},
    {"camera_id": "other"}, {"people": [{"bbox": [0, 0, 2, 1]}]},
    {"people": [{"bbox": [0.9, 0, 0.2, 1]}]},
])
def test_bad_preview_never_replaces_good_frame(change):
    camera = RemoteCameraView("cam", wall_clock=lambda: 100)
    camera.update(payload(), sequence=1, run_id="run")
    good = camera.frame_jpeg()
    with pytest.raises(ValueError):
        camera.update({**payload(), **change}, sequence=2, run_id="run")
    assert camera.frame_jpeg() == good


def test_reconnect_cannot_make_old_capture_live_by_sending_it_again():
    camera = RemoteCameraView("cam", stale_after_s=10, wall_clock=lambda: 200)
    assert not camera.update(payload(), sequence=20, run_id="reconnected", producer_id="producer")
    assert camera.frame_jpeg() is None
    assert not camera.running


def test_delayed_delivery_only_receives_remaining_capture_ttl():
    now = [20.0]
    camera = RemoteCameraView("cam", stale_after_s=10, clock=lambda: now[0], wall_clock=lambda: 108)
    assert camera.update(payload(), sequence=20, run_id="run")  # captured 8 seconds ago
    assert camera.status()["preview_age_s"] == 8
    now[0] = 21.9
    assert camera.running
    now[0] = 22.0
    assert camera.frame_jpeg() is None
    assert camera.status()["stale"]


def test_capture_from_far_future_is_not_displayed_as_live():
    camera = RemoteCameraView("cam", stale_after_s=10, wall_clock=lambda: 50)
    assert not camera.update(payload(), sequence=20, run_id="run")
    assert not camera.running


def test_frame_expiring_during_overlay_decode_is_not_revived(monkeypatch):
    now = [20.0]
    camera = RemoteCameraView("cam", stale_after_s=10, clock=lambda: now[0], wall_clock=lambda: 108)
    draw = camera._draw

    def delayed_draw(image, data):
        result = draw(image, data)
        now[0] += 3
        return result

    monkeypatch.setattr(camera, "_draw", delayed_draw)
    assert not camera.update(payload(), sequence=20, run_id="run")
    assert not camera.running
