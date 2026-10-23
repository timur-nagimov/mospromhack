import numpy as np
import pytest

from guard.contracts import Detection, Track
from guard.tracking import assign_attributes


def person(track_id=1, bbox=(0.0, 0.0, 100.0, 100.0)):
    return Track(
        track_id=track_id,
        cls="person",
        bbox=bbox,
        conf=0.9,
        first_seen=0.0,
        last_seen=0.0,
        hits=2,
    )


def detection(cls, bbox, model="ppe", conf=0.9, keypoints=None):
    return Detection(
        model=model,
        cls=cls,
        cls_id=0,
        conf=conf,
        bbox=bbox,
        keypoints=keypoints,
    )


def standing_points():
    points = np.zeros((17, 3), dtype=float)
    for index, x, y in (
        (0, 50, 10),
        (5, 35, 30),
        (6, 65, 30),
        (9, 80, 60),
        (10, 20, 60),
        (11, 40, 65),
        (12, 60, 65),
    ):
        points[index] = (x, y, 0.9)
    return points


def pose(track, points=None):
    return detection("person", track.bbox, model="pose", keypoints=points)


@pytest.mark.parametrize(
    "cls, attribute, expected, bbox",
    [
        ("helmet", "helmet", True, (40, 2, 60, 18)),
        ("no_helmet", "helmet", False, (40, 2, 60, 18)),
        ("vest", "vest", True, (35, 35, 65, 65)),
        ("no_vest", "vest", False, (35, 35, 65, 65)),
        ("respirator", "respirator", True, (40, 15, 60, 30)),
        ("no_respirator", "respirator", False, (40, 15, 60, 30)),
    ],
)
def test_ppe_values(cls, attribute, expected, bbox):
    track = person()
    assign_attributes([track], {"ppe": [detection(cls, bbox)]}, 0.0)
    assert track.attributes[attribute] is expected


def test_absent_attributes_are_unknown():
    track = person()
    assign_attributes([track], {}, 0.0)
    for name in ("helmet", "vest", "respirator"):
        assert track.attributes[name] is None
    assert track.attributes["pose_state"] == "unknown"
    assert track.attributes["lying_since"] is None
    assert track.attributes["phone"] is False
    assert track.keypoints is None


def test_ppe_survives_brief_missing_data_and_expires_after_two_seconds():
    track = person()
    helmet = detection("helmet", (40, 2, 60, 18))
    assign_attributes([track], {"ppe": [helmet]}, 0.0)
    assign_attributes([track], {}, 0.8)
    assert track.attributes["helmet"] is True
    assign_attributes([track], {}, 2.01)
    assert track.attributes["helmet"] is None


def test_ppe_temporal_majority_and_tie():
    track = person()
    yes = detection("helmet", (40, 2, 60, 18))
    no = detection("no_helmet", (40, 2, 60, 18))
    assign_attributes([track], {"ppe": [yes]}, 0.0)
    assign_attributes([track], {"ppe": [no]}, 0.2)
    assert track.attributes["helmet"] is None
    assign_attributes([track], {"ppe": [no]}, 0.4)
    assert track.attributes["helmet"] is False
    assign_attributes([track], {}, 2.5)
    assert track.attributes["helmet"] is None


def test_ppe_has_only_one_vote_per_frame_and_attribute():
    track = person()
    yes = detection("helmet", (40, 2, 60, 18), conf=0.9)
    no = detection("no_helmet", (40, 2, 60, 18), conf=0.6)
    assign_attributes([track], {"ppe": [no, no, yes]}, 0.0)
    assert track.attributes["helmet"] is True
    assign_attributes([track], {"ppe": [no]}, 0.0)
    assert track.attributes["helmet"] is False


@pytest.mark.parametrize(
    "cls,bbox",
    [
        ("helmet", (40, 70, 60, 90)),
        ("vest", (40, 0, 60, 10)),
        ("helmet", (-30, 5, -20, 15)),
    ],
)
def test_ppe_outside_required_body_region_is_not_assigned(cls, bbox):
    track = person()
    assign_attributes([track], {"ppe": [detection(cls, bbox)]}, 0.0)
    assert track.attributes[cls] is None


def test_ppe_width_tolerance():
    track = person()
    assign_attributes(
        [track], {"ppe": [detection("helmet", (-10, 5, 0, 15))]}, 0.0
    )
    assert track.attributes["helmet"] is True


def test_ppe_detection_is_assigned_to_only_one_nearby_person():
    first = person(1)
    second = person(2, (80, 0, 180, 100))
    assign_attributes(
        [first, second],
        {"ppe": [detection("helmet", (105, 5, 115, 15))]},
        0.0,
    )
    assert first.attributes["helmet"] is None
    assert second.attributes["helmet"] is True


def test_attribute_memory_is_per_track_object_not_id():
    first = person(1)
    second = person(1)
    assign_attributes(
        [first], {"ppe": [detection("helmet", (40, 2, 60, 18))]}, 0.0
    )
    assign_attributes([second], {}, 0.0)
    assert first.attributes["helmet"] is True
    assert second.attributes["helmet"] is None


def test_lying_geometry_and_continuous_start_timestamp():
    track = person(bbox=(0, 0, 140, 100))
    assign_attributes([track], {"pose": [pose(track)]}, 2.0)
    assert track.attributes["pose_state"] == "lying"
    assert track.attributes["lying_since"] == 2.0
    assign_attributes([track], {"pose": [pose(track)]}, 2.5)
    assert track.attributes["lying_since"] == 2.0

    track.bbox = (0, 0, 100, 100)
    assign_attributes(
        [track], {"pose": [pose(track, standing_points())]}, 3.0
    )
    assert track.attributes["pose_state"] == "standing"
    assert track.attributes["lying_since"] is None

    track.bbox = (0, 0, 140, 100)
    assign_attributes([track], {"pose": [pose(track)]}, 3.1)
    assert track.attributes["lying_since"] == 3.1


def test_lying_from_torso_angle():
    track = person()
    points = standing_points()
    points[5] = (20, 40, 0.9)
    points[6] = (30, 40, 0.9)
    points[11] = (70, 45, 0.9)
    points[12] = (80, 45, 0.9)
    assign_attributes([track], {"pose": [pose(track, points)]}, 0.0)
    assert track.attributes["pose_state"] == "lying"
    assert track.attributes["lying_since"] == 0.0


def test_insufficient_pose_points_produce_unknown():
    track = person()
    points = standing_points()
    points[11:13, 2] = 0.1
    assign_attributes([track], {"pose": [pose(track, points)]}, 0.0)
    assert track.attributes["pose_state"] == "unknown"


def test_missing_pose_clears_keypoints_and_breaks_lying_continuity():
    track = person(bbox=(0, 0, 140, 100))
    assign_attributes(
        [track], {"pose": [pose(track, standing_points())]}, 0.0
    )
    assign_attributes([track], {}, 0.2)
    assert track.keypoints is None
    assert track.attributes["pose_state"] == "unknown"
    assert track.attributes["lying_since"] is None
    assign_attributes([track], {"pose": [pose(track)]}, 0.4)
    assert track.attributes["lying_since"] == 0.4


def test_pose_assignment_is_one_to_one_and_keypoints_are_copied():
    first = person(1)
    second = person(2, (20, 0, 120, 100))
    points = standing_points()
    original = points.copy()
    assign_attributes(
        [first, second], {"pose": [pose(first, points)]}, 0.0
    )
    assert first.attributes["pose_state"] == "standing"
    assert second.attributes["pose_state"] == "unknown"
    assert second.keypoints is None
    points[0, 0] = 999
    np.testing.assert_array_equal(first.keypoints, original)


def test_pose_iou_below_threshold_is_rejected():
    track = person()
    unrelated_pose = detection(
        "person", (60, 0, 160, 100), model="pose", keypoints=standing_points()
    )
    assign_attributes([track], {"pose": [unrelated_pose]}, 0.0)
    assert track.keypoints is None
    assert track.attributes["pose_state"] == "unknown"


@pytest.mark.parametrize("bbox", [(76, 56, 84, 64), (46, 7, 54, 15)])
def test_phone_near_wrist_or_head(bbox):
    track = person()
    assign_attributes(
        [track],
        {
            "pose": [pose(track, standing_points())],
            "det": [detection("cell phone", bbox, model="det")],
        },
        0.0,
    )
    assert track.attributes["phone"] is True


def test_phone_intersection_without_keypoints():
    track = person()
    assign_attributes(
        [track],
        {"det": [detection("cell phone", (90, 90, 110, 110), model="det")]},
        0.0,
    )
    assert track.attributes["phone"] is True


@pytest.mark.parametrize("bbox", [(45, 85, 55, 95), (150, 5, 160, 15)])
def test_phone_far_from_keypoints_or_outside_person_is_rejected(bbox):
    track = person()
    assign_attributes(
        [track],
        {
            "pose": [pose(track, standing_points())],
            "det": [detection("cell phone", bbox, model="det")],
        },
        0.0,
    )
    assert track.attributes["phone"] is False


def test_phone_does_not_use_unconfident_keypoints():
    track = person()
    assign_attributes(
        [track],
        {
            "pose": [pose(track, np.zeros((17, 3)))],
            "det": [detection("cell phone", (45, 5, 55, 15), model="det")],
        },
        0.0,
    )
    assert track.attributes["phone"] is False


def test_phone_uses_current_keypoints_supplied_by_tracker():
    track = person()
    track.keypoints = standing_points()
    assign_attributes(
        [track],
        {"det": [detection("cell phone", (45, 85, 55, 95), model="det")]},
        0.0,
    )
    assert track.attributes["phone"] is False
    assert track.attributes["pose_state"] == "standing"


def test_attribute_time_validation():
    track = person()
    assign_attributes([track], {}, 1.0)
    for ts in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            assign_attributes([track], {}, ts)


def test_empty_people():
    assign_attributes([], {}, 0.0)
