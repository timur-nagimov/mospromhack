import cv2
import numpy as np

from guard.contracts import Severity, Track, Violation, Zone
from guard.vision.draw import draw_event_snapshot, draw_overlay


def _track(**overrides):
    values = dict(
        track_id=7,
        cls="person",
        bbox=(10, 10, 40, 60),
        conf=0.9,
        first_seen=0,
        last_seen=1,
        attributes={"helmet": True, "vest": False},
    )
    values.update(overrides)
    return Track(**values)


def test_highlighted_track_is_red_and_source_frame_is_unchanged():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    result = draw_overlay(frame, [_track()], [], [], events_highlight={7})
    assert result.shape == frame.shape
    assert result.dtype == frame.dtype
    assert not np.shares_memory(result, frame)
    np.testing.assert_array_equal(frame, 0)
    np.testing.assert_array_equal(result[40, 10], [0, 0, 255])


def test_zone_uses_normalized_coordinates_and_translucent_fill():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    zone = Zone(
        zone_id="z1",
        name="Danger",
        polygon=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
    )
    result = draw_overlay(frame, [], [], [zone])
    np.testing.assert_array_equal(result[50, 50], [42, 24, 8])
    np.testing.assert_array_equal(result[90, 90], [0, 0, 0])
    np.testing.assert_array_equal(frame, 0)


def test_spatial_violation_without_track_draws_event_box():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    violation = Violation(
        rule_id="r1",
        type="smoking_fire",
        severity=Severity.high,
        track_id=None,
        zone_id=None,
        bbox=(10, 10, 40, 60),
    )
    result = draw_overlay(frame, [], [], [], [violation])
    np.testing.assert_array_equal(result[40, 10], [0, 0, 255])


def test_pose_draws_visible_keypoints_and_edges():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = [30, 40, 0.9]
    keypoints[7] = [50, 60, 0.9]
    track = _track(bbox=(10, 10, 90, 90), keypoints=keypoints)
    result = draw_overlay(frame, [track], [], [])
    assert result[50, 40].any()
    assert result[40, 30].any()


def test_labels_include_attributes_and_are_ascii(monkeypatch):
    labels = []
    original_put_text = cv2.putText

    def recording_put_text(image, text, *args, **kwargs):
        assert text.isascii()
        labels.append(text)
        return original_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", recording_put_text)
    frame = np.zeros((80, 200, 3), dtype=np.uint8)
    draw_overlay(frame, [_track()], [], [])
    assert any("#7" in label and "helmet:yes" in label and "vest:no" in label for label in labels)

    snapshot = draw_event_snapshot(frame, (10, 10, 40, 60), "Нарушение")
    np.testing.assert_array_equal(snapshot[40, 10], [0, 0, 255])
    np.testing.assert_array_equal(frame, 0)
