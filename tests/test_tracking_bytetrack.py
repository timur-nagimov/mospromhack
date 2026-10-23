import math

import pytest

from guard.contracts import Detection
from guard.tracking import Tracker


def detection(x=0.0, y=0.0, width=80.0, height=100.0, conf=0.9, cls="person"):
    return Detection(
        model="det",
        cls=cls,
        cls_id=0,
        conf=conf,
        bbox=(x, y, x + width, y + height),
    )


def test_uniform_motion_preserves_id_for_30_frames():
    tracker = Tracker({"person"})
    identities = []
    for frame in range(30):
        ts = frame * 0.2
        tracks = tracker.update([detection(x=30 * ts)], ts)
        if frame == 0:
            assert tracks == []
        else:
            assert len(tracks) == 1
            identities.append(tracks[0].track_id)

    assert len(set(identities)) == 1
    track = tracks[0]
    assert track.hits == 30
    assert track.first_seen == 0.0
    assert track.last_seen == pytest.approx(5.8)
    assert len(track.history) == 30
    assert track.history[-1] == pytest.approx((5.8, 214.0, 100.0, 100.0))
    assert track.velocity == pytest.approx((30.0, 0.0), abs=1e-6)


def test_two_nearby_crossing_objects_keep_their_ids():
    tracker = Tracker({"person"})
    identities = None
    for frame in range(40):
        ts = frame * 0.2
        first = detection(x=10 + 20 * ts, y=0, width=60)
        second = detection(x=200 - 20 * ts, y=5, width=60)
        detections = [first, second] if frame % 2 else [second, first]
        tracks = tracker.update(detections, ts)
        if frame == 0:
            continue
        by_y = {track.bbox[1]: track for track in tracks}
        assert set(by_y) == {0.0, 5.0}
        current = (by_y[0.0].track_id, by_y[5.0].track_id)
        if identities is None:
            identities = current
        assert current == identities
        assert by_y[0.0].bbox[0] == pytest.approx(first.bbox[0])
        assert by_y[5.0].bbox[0] == pytest.approx(second.bbox[0])


def test_one_second_missing_detections_preserves_id():
    tracker = Tracker({"person"}, track_buffer_s=3.0)
    for frame in range(10):
        ts = frame * 0.2
        tracks = tracker.update([detection(x=20 * ts)], ts)
    old_id = tracks[0].track_id
    old_hits = tracks[0].hits
    last_ts = ts
    for offset in (0.2, 0.4, 0.6, 0.8, 1.0):
        assert tracker.update([], last_ts + offset) == []

    ts = last_ts + 1.2
    recovered = tracker.update([detection(x=20 * ts)], ts)
    assert len(recovered) == 1
    assert recovered[0].track_id == old_id
    assert recovered[0].hits == old_hits + 1
    assert recovered[0].velocity == pytest.approx((20.0, 0.0), abs=1e-6)
    assert len(recovered[0].history) == old_hits + 1


@pytest.mark.parametrize("submit_empty_frame", [False, True])
def test_four_second_gap_expires_track_before_matching(submit_empty_frame):
    tracker = Tracker({"person"}, min_hits=1, track_buffer_s=3.0)
    original = tracker.update([detection()], 0.0)[0]
    if submit_empty_frame:
        assert tracker.update([], 4.0) == []
    replacement = tracker.update([detection()], 4.1)[0]
    assert replacement.track_id != original.track_id
    assert replacement.first_seen == 4.1
    assert replacement.hits == 1


def test_buffer_boundary_is_inclusive():
    tracker = Tracker({"person"}, min_hits=1, track_buffer_s=3.0)
    original_id = tracker.update([detection()], 0.0)[0].track_id
    assert tracker.update([detection()], 3.0)[0].track_id == original_id


@pytest.mark.parametrize("speed, expected", [(200.0, 2.0), (50.0, 0.5)])
def test_speed_is_in_body_heights_per_second(speed, expected):
    tracker = Tracker({"person"})
    for frame in range(30):
        ts = frame * 0.2
        tracks = tracker.update(
            [detection(x=speed * ts, width=100, height=100)], ts
        )
    track = tracks[0]
    assert track.speed_bh == pytest.approx(expected, abs=1e-6)
    assert track.velocity == pytest.approx((speed, 0.0), abs=1e-6)
    if speed == 200.0:
        assert track.speed_bh > 1.5
    else:
        assert track.speed_bh < 1.0


def test_nonuniform_epoch_timestamps_use_real_dt():
    tracker = Tracker({"person"})
    epoch = 1_700_000_000.0
    elapsed = 0.0
    identities = set()
    for step in [0.0] + [0.1, 0.4, 0.2, 0.3] * 8:
        elapsed += step
        tracks = tracker.update(
            [detection(x=40 * elapsed, y=10 * elapsed)],
            epoch + elapsed,
        )
        if tracks:
            identities.add(tracks[0].track_id)
    assert len(identities) == 1
    assert tracks[0].velocity == pytest.approx((40.0, 10.0), abs=1e-4)
    assert tracks[0].speed_bh == pytest.approx(math.hypot(40, 10) / 100, abs=1e-5)


def test_low_confidence_continues_confirmed_track():
    tracker = Tracker({"person"})
    assert tracker.update([detection(x=0)], 0.0) == []
    track = tracker.update([detection(x=2)], 0.2)[0]
    track.attributes["custom"] = "preserved"
    result = tracker.update([detection(x=4, conf=0.2)], 0.4)
    assert len(result) == 1
    assert result[0].track_id == track.track_id
    assert result[0].hits == 3
    assert result[0].attributes["custom"] == "preserved"
    assert tracker.update([detection(x=6, conf=0.05)], 0.6) == []


def test_low_confidence_does_not_spawn_confirm_or_revive():
    tracker = Tracker({"person"})
    assert tracker.update([detection(conf=0.2)], 0.0) == []
    assert tracker.update([detection(conf=0.55)], 0.2) == []
    assert tracker.update([detection()], 0.4) == []
    assert tracker.update([detection(conf=0.2)], 0.6) == []
    assert tracker.update([detection()], 0.8) == []
    track_id = tracker.update([detection()], 1.0)[0].track_id
    assert tracker.update([], 1.2) == []
    assert tracker.update([detection(conf=0.2)], 1.4) == []
    assert tracker.update([detection()], 1.6)[0].track_id == track_id


def test_class_filtering_and_class_aware_matching():
    tracker = Tracker({"person", "truck"}, min_hits=1)
    first = tracker.update(
        [detection(cls="person"), detection(cls="truck"), detection(cls="dog")],
        0.0,
    )
    ids = {track.cls: track.track_id for track in first}
    assert set(ids) == {"person", "truck"}
    second = tracker.update(
        [detection(cls="truck"), detection(cls="person")], 0.2
    )
    assert {track.cls: track.track_id for track in second} == ids


def test_repeated_timestamp_does_not_inflate_hits_or_history():
    tracker = Tracker({"person"}, min_hits=1)
    original = tracker.update([detection()], 0.0)[0]
    repeated = tracker.update([detection(x=1)], 0.0)[0]
    assert repeated.track_id == original.track_id
    assert repeated.hits == 1
    assert len(repeated.history) == 1
    assert repeated.history[-1][1] == 41.0


def test_invalid_timestamps_fail_without_advancing_state():
    tracker = Tracker({"person"}, min_hits=1)
    original_id = tracker.update([detection()], 1.0)[0].track_id
    for ts in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            tracker.update([], ts)
    assert tracker.update([detection()], 1.2)[0].track_id == original_id


def test_invalid_detection_is_ignored():
    tracker = Tracker({"person"}, min_hits=1)
    assert tracker.update(
        [
            detection(width=0),
            detection(height=-1),
            detection(x=float("nan")),
            detection(conf=float("nan")),
        ],
        0.0,
    ) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"low_thresh": 0.7, "high_thresh": 0.5},
        {"match_iou": 1.1},
        {"track_buffer_s": -1},
        {"min_hits": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Tracker({"person"}, **kwargs)
