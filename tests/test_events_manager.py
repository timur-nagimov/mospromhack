from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest

from guard.contracts import (
    TIMING_KEYS,
    CameraConfig,
    RuleConfig,
    Severity,
    SourceConfig,
    Violation,
)
from guard.events.manager import EventManager


def _camera(camera_id="cam1", **rule_options):
    return CameraConfig(
        camera_id=camera_id,
        name=camera_id,
        site="workshop",
        source=SourceConfig(type="file", uri="demo.mp4"),
        rules=[
            RuleConfig(
                rule_id="r1",
                type="zone_intrusion",
                **rule_options,
            )
        ],
    )


def _violation(
    track_id=1,
    rule_id="r1",
    bbox=(8.0, 8.0, 40.0, 80.0),
    zone_id="z1",
    severity=Severity.high,
    details=None,
    kind="zone_intrusion",
):
    return Violation(
        rule_id=rule_id,
        type=kind,
        severity=severity,
        track_id=track_id,
        zone_id=zone_id,
        bbox=bbox,
        details={} if details is None else details,
    )


def _feed(manager, camera, indices, violations_at):
    created, closed = [], []
    for index in indices:
        new, ended = manager.process(camera, index / 5, violations_at(index))
        created.extend(new)
        closed.extend(ended)
    return created, closed


def test_single_violating_frame_does_not_confirm():
    manager = EventManager()
    created, closed = _feed(
        manager,
        _camera(),
        range(31),
        lambda i: [_violation()] if i == 0 else [],
    )
    assert created == []
    assert closed == []
    assert manager.open_events() == []


def test_three_seconds_confirm_once_with_original_start():
    manager = EventManager()
    created, closed = _feed(
        manager, _camera(min_duration_s=2), range(16), lambda _: [_violation()]
    )
    assert len(created) == 1
    event = created[0]
    assert event.started_at == 0.0
    assert event.confirmed_at - event.started_at == pytest.approx(2, abs=0.25)
    assert event.latency_ms == {}
    assert closed == []
    assert len(manager.open_events()) == 1


@pytest.mark.parametrize("fraction, expected", [(0.8, 1), (0.5, 0)])
def test_flickering_uses_fraction_of_all_camera_frames(fraction, expected):
    manager = EventManager()

    def violations_at(index):
        present = index % 5 != 4 if fraction == 0.8 else index % 2 == 0
        return [_violation()] if present else []

    created, _ = _feed(
        manager,
        _camera(min_duration_s=2, min_ratio=0.7),
        range(101),
        violations_at,
    )
    assert len(created) == expected


def test_sixty_seconds_still_produce_one_event():
    manager = EventManager()
    created, closed = _feed(
        manager, _camera(), range(301), lambda _: [_violation()]
    )
    assert len(created) == 1
    assert closed == []
    assert len(manager.open_events()) == 1


def test_track_change_remembers_alias_even_after_person_moves():
    manager = EventManager()
    camera = _camera()
    moved = (200.0, 8.0, 232.0, 80.0)

    def violations_at(index):
        if index < 15:
            return [_violation(track_id=1)]
        bbox = (12.0, 8.0, 44.0, 80.0) if index < 20 else moved
        track_id = 2 if index < 25 else 1
        return [_violation(track_id=track_id, bbox=bbox)]

    created, closed = _feed(manager, camera, range(31), violations_at)
    assert len(created) == 1
    assert closed == []
    opened = manager.open_events()[0]
    assert opened.id == created[0].id
    assert opened.bbox == moved
    assert created[0].bbox == (8.0, 8.0, 40.0, 80.0)


@pytest.mark.parametrize("next_track_id", [1, 2])
def test_after_closure_and_cooldown_a_new_event_can_confirm(next_track_id):
    manager = EventManager()
    camera = _camera(min_duration_s=0.4, close_after_s=0.4, cooldown_s=1)
    first, _ = _feed(manager, camera, range(3), lambda _: [_violation()])
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])
    assert len(first) == len(closed) == 1
    assert closed[0].ended_at == pytest.approx(0.4)

    # политика CX-017: для другого ID ждём также истечения reid_gap_s=5.
    second, _ = _feed(
        manager,
        camera,
        range(31, 34),
        lambda _: [_violation(track_id=next_track_id)],
    )
    assert len(second) == 1
    assert second[0].id != first[0].id
    # политика CX-017: новый интервал начинается после обоих окон.
    assert second[0].started_at == pytest.approx(6.2)
    assert second[0].confirmed_at == pytest.approx(6.6)


def test_recently_closed_event_reopens_without_new_notification():
    manager = EventManager()
    camera = _camera(min_duration_s=0.4, close_after_s=0.4, cooldown_s=2)
    first, _ = _feed(manager, camera, range(3), lambda _: [_violation()])
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])

    # политика CX-017: другой ID возвращается через 2.4 с от ended_at,
    # внутри reid_gap_s=5; cooldown другого трека не ограничивает.
    assert manager.process(camera, 2.8, [_violation(track_id=2)]) == ([], [])
    reopened = manager.open_events()[0]
    assert reopened.id == first[0].id
    assert reopened.started_at == 0.0
    assert reopened.confirmed_at == pytest.approx(0.4)
    assert reopened.ended_at is None
    assert closed[0].ended_at == pytest.approx(0.4)

    moved = (200.0, 8.0, 232.0, 80.0)
    assert manager.process(
        camera, 3.0, [_violation(track_id=2, bbox=moved)]
    ) == ([], [])
    new, closed_again = manager.process(camera, 3.6, [])
    assert new == []
    assert len(closed_again) == 1
    assert closed_again[0].id == first[0].id
    assert closed_again[0].ended_at == 3.0
    assert closed_again[0].bbox == moved


@pytest.mark.parametrize(
    "second_bbox",
    [
        (200.0, 8.0, 232.0, 80.0),
        (12.0, 8.0, 44.0, 80.0),
    ],
)
def test_two_simultaneous_people_have_separate_events(second_bbox):
    manager = EventManager()
    created, _ = _feed(
        manager,
        _camera(),
        range(16),
        lambda _: [
            _violation(track_id=1),
            _violation(track_id=2, bbox=second_bbox),
        ],
    )
    assert len(created) == 2
    assert len({event.id for event in created}) == 2
    assert {event.track_id for event in created} == {1, 2}


def test_new_overlapping_person_cannot_steal_present_tracks_event():
    manager = EventManager()
    camera = _camera()
    first, _ = _feed(manager, camera, range(12), lambda _: [_violation()])
    second, _ = _feed(
        manager,
        camera,
        range(12, 24),
        # The unknown track deliberately comes before the known track.
        lambda _: [_violation(track_id=2), _violation(track_id=1)],
    )
    assert len(first) == len(second) == 1
    assert first[0].id != second[0].id
    assert len(manager.open_events()) == 2


def test_same_person_violating_two_rules_has_two_events():
    manager = EventManager()
    camera = _camera()
    camera.rules.append(RuleConfig(rule_id="r2", type="ppe_missing"))
    created, _ = _feed(
        manager,
        camera,
        range(16),
        lambda _: [
            _violation(),
            _violation(rule_id="r2", kind="ppe_missing"),
        ],
    )
    assert len(created) == 2
    assert {event.rule_id for event in created} == {"r1", "r2"}
    assert {event.type for event in created} == {"zone_intrusion", "ppe_missing"}


def test_closure_is_strict_and_reported_once():
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=1)
    created, _ = manager.process(camera, 0, [_violation()])
    assert len(created) == 1
    assert manager.process(camera, 1, []) == ([], [])

    new, closed = manager.process(camera, 1.001, [])
    assert new == []
    assert len(closed) == 1
    assert closed[0].id == created[0].id
    assert closed[0].ended_at == 0
    assert manager.open_events() == []
    assert manager.process(camera, 2, []) == ([], [])


def test_trackless_violation_confirms_and_survives_cell_change():
    manager = EventManager()

    def violations_at(index):
        if index == 0:
            bbox = (8.0, 8.0, 24.0, 40.0)
        elif index < 15:
            bbox = (9.0, 8.0, 25.0, 40.0)
        else:
            # New 8-pixel cell; IoU with the preceding box is 1/3.
            bbox = (17.0, 8.0, 33.0, 40.0)
        return [_violation(track_id=None, bbox=bbox)]

    created, closed = _feed(manager, _camera(), range(31), violations_at)
    assert len(created) == 1
    assert created[0].track_id is None
    assert created[0].zone_id == "z1"
    assert closed == []
    assert manager.open_events()[0].bbox == (17.0, 8.0, 33.0, 40.0)


def test_trackless_key_includes_zone():
    manager = EventManager()
    created, _ = _feed(
        manager,
        _camera(),
        range(16),
        lambda _: [
            _violation(track_id=None, zone_id="z1"),
            _violation(track_id=None, zone_id="z2"),
        ],
    )
    assert len(created) == 2
    assert {event.zone_id for event in created} == {"z1", "z2"}


def test_duplicate_violations_in_one_frame_count_only_once():
    manager = EventManager()
    created, _ = _feed(
        manager,
        _camera(),
        range(51),
        lambda i: [_violation(), _violation()] if i % 2 == 0 else [],
    )
    assert created == []


def test_confirmation_window_includes_its_left_boundary():
    manager = EventManager()
    camera = _camera(min_duration_s=1, min_ratio=1)
    assert manager.process(camera, 0, [_violation()]) == ([], [])
    assert manager.process(camera, 0.5, []) == ([], [])
    assert manager.process(camera, 1, [_violation()]) == ([], [])
    assert manager.process(camera, 1.5, [_violation()]) == ([], [])

    created, _ = manager.process(camera, 2, [_violation()])
    assert len(created) == 1
    assert created[0].started_at == 0
    assert created[0].confirmed_at == 2


def test_confirmation_can_happen_on_empty_frame_when_ratio_is_sufficient():
    manager = EventManager()
    camera = _camera(min_duration_s=1, min_ratio=0.8)
    created, _ = _feed(manager, camera, range(5), lambda _: [_violation()])
    assert created == []

    created, closed = manager.process(camera, 1, [])
    assert len(created) == 1
    assert created[0].confirmed_at == 1
    assert closed == []

    _, closed = manager.process(camera, 4, [])
    assert len(closed) == 1
    assert closed[0].ended_at == pytest.approx(0.8)


def test_series_restarts_after_long_absence():
    manager = EventManager()
    camera = _camera()
    assert manager.process(camera, 0, [_violation()]) == ([], [])
    created, _ = _feed(manager, camera, range(1, 13), lambda _: [])
    assert created == []
    created, _ = _feed(manager, camera, range(13, 24), lambda _: [_violation()])
    assert len(created) == 1
    assert created[0].started_at == pytest.approx(2.6)
    assert created[0].confirmed_at == pytest.approx(4.6)


def test_event_metadata_and_returned_snapshots_are_independent():
    manager = EventManager()
    camera = _camera(min_duration_s=0, severity=Severity.low)
    details = {"missing": ["helmet"]}
    timings = {key: float(index) for index, key in enumerate(TIMING_KEYS)}
    violation = _violation(severity=Severity.critical, details=details)
    created, _ = manager.process(camera, 10, [violation], timings)
    event = created[0]

    assert len(event.id) == 32
    assert UUID(event.id).version == 4
    assert UUID(event.id).hex == event.id
    assert event.camera_id == camera.camera_id
    assert event.site == camera.site
    assert event.zone_id == violation.zone_id
    assert event.rule_id == violation.rule_id
    assert event.type == violation.type
    assert event.severity == Severity.critical
    assert event.started_at == event.confirmed_at == 10
    assert event.latency_ms == timings
    assert event.details == details

    timings["total"] = 999
    details["missing"].append("vest")
    assert event.latency_ms["total"] != 999
    assert event.details == {"missing": ["helmet"]}

    snapshot = manager.open_events()[0]
    snapshot.details["missing"].append("respirator")
    snapshot.bbox = (0, 0, 1, 1)
    current = manager.open_events()[0]
    assert current.details == {"missing": ["helmet"]}
    assert current.bbox == violation.bbox


def test_missing_rule_uses_rule_config_defaults():
    manager = EventManager()
    camera = _camera()
    camera.rules = []
    created, _ = _feed(
        manager,
        camera,
        range(16),
        lambda _: [_violation(severity=Severity.critical)],
    )
    assert len(created) == 1
    defaults = RuleConfig(rule_id="r1", type="zone_intrusion")
    assert created[0].confirmed_at == pytest.approx(defaults.min_duration_s)
    assert created[0].severity == Severity.critical


def test_cameras_are_isolated_and_reset_only_removes_requested_camera():
    manager = EventManager()
    first = _camera("first", min_duration_s=0)
    second = _camera("second", min_duration_s=0)
    first_events, _ = manager.process(first, 10, [_violation()])
    second_events, _ = manager.process(second, 0, [_violation()])

    assert len(manager.open_events()) == 2
    assert [e.id for e in manager.open_events("second")] == [second_events[0].id]
    assert manager.open_events("unknown") == []

    manager.reset("first")
    manager.reset("unknown")
    assert manager.open_events("first") == []
    assert [e.id for e in manager.open_events()] == [second_events[0].id]

    restarted, _ = manager.process(first, 0, [_violation()])
    assert len(restarted) == 1
    assert restarted[0].id != first_events[0].id


def test_concurrent_calls_do_not_duplicate_confirmation():
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: manager.process(camera, 0, [_violation()])[0],
                range(32),
            )
        )
    assert sum(len(events) for events in results) == 1
    assert len(manager.open_events()) == 1


def test_invalid_timestamps_do_not_corrupt_existing_state():
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    created, _ = manager.process(camera, 1, [_violation()])
    for ts in (0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            manager.process(camera, ts, [])
    assert [e.id for e in manager.open_events()] == [created[0].id]
    assert manager.process(camera, 2, [_violation()]) == ([], [])


def test_trackless_fire_confirms_with_twenty_pixel_jitter():
    from random import Random

    manager = EventManager()
    camera = _camera(min_duration_s=2, min_ratio=0.7)
    camera.rules[0].type = "smoking_fire"
    rng = Random(1)

    def violations_at(index):
        dx = rng.uniform(-20, 20)
        dy = rng.uniform(-20, 20)
        return [
            _violation(
                track_id=None,
                bbox=(100.0 + dx, 100.0 + dy, 160.0 + dx, 180.0 + dy),
                kind="smoking_fire",
                severity=Severity.critical,
            )
        ]

    created, closed = _feed(manager, camera, range(16), violations_at)
    assert len(created) == 1
    event = created[0]
    assert event.track_id is None
    assert event.type == "smoking_fire"
    assert event.severity == Severity.critical
    assert event.started_at == 0.0
    assert event.confirmed_at - event.started_at == pytest.approx(2, abs=0.25)
    assert closed == []
    assert [event.id for event in manager.open_events()] == [created[0].id]


def test_two_distant_trackless_boxes_confirm_separately():
    manager = EventManager()
    boxes = [
        (100.0, 100.0, 160.0, 180.0),
        (500.0, 100.0, 560.0, 180.0),
    ]

    def violations_at(index):
        ordered = boxes if index % 2 == 0 else list(reversed(boxes))
        return [_violation(track_id=None, bbox=bbox) for bbox in ordered]

    created, closed = _feed(
        manager,
        _camera(min_duration_s=2),
        range(16),
        violations_at,
    )
    assert len(created) == 2
    assert len({event.id for event in created}) == 2
    assert {event.bbox for event in created} == set(boxes)
    assert all(event.track_id is None for event in created)
    assert all(event.zone_id == "z1" for event in created)
    assert all(event.started_at == 0.0 for event in created)
    assert all(event.confirmed_at == pytest.approx(2) for event in created)
    assert closed == []
    assert len(manager.open_events()) == 2


def test_trackless_box_moving_five_pixels_per_frame_keeps_one_event():
    manager = EventManager()

    def violations_at(index):
        dx = 5.0 * index
        return [
            _violation(
                track_id=None,
                bbox=(100.0 + dx, 100.0, 160.0 + dx, 180.0),
            )
        ]

    created, closed = _feed(
        manager,
        _camera(min_duration_s=2),
        range(31),
        violations_at,
    )
    assert len(created) == 1
    assert created[0].started_at == 0.0
    assert created[0].confirmed_at == pytest.approx(2)
    assert closed == []
    opened = manager.open_events()
    assert len(opened) == 1
    assert opened[0].id == created[0].id
    assert opened[0].bbox == (250.0, 100.0, 310.0, 180.0)


def test_trackless_event_confirms_again_after_closure_and_cooldown():
    manager = EventManager()
    camera = _camera(min_duration_s=0.4, close_after_s=0.4, cooldown_s=1)
    bbox = (100.0, 100.0, 160.0, 180.0)
    first, _ = _feed(
        manager,
        camera,
        range(3),
        lambda _: [_violation(track_id=None, bbox=bbox)],
    )
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])
    assert len(first) == len(closed) == 1
    assert closed[0].id == first[0].id
    assert closed[0].ended_at == pytest.approx(0.4)
    assert manager.open_events() == []

    # политика CX-017: без трека новое событие ждёт истечения reid_gap_s=5.
    second, closed_again = _feed(
        manager,
        camera,
        range(31, 34),
        lambda _: [_violation(track_id=None, bbox=bbox)],
    )
    assert len(second) == 1
    assert second[0].id != first[0].id
    # политика CX-017: ожидания соответствуют интервалу после reid_gap_s.
    assert second[0].started_at == pytest.approx(6.2)
    assert second[0].confirmed_at == pytest.approx(6.6)
    assert closed_again == []
    assert [event.id for event in manager.open_events()] == [second[0].id]


@pytest.mark.parametrize("min_duration", [0, 0.4])
def test_trackless_matches_are_globally_greedy_and_one_to_one(min_duration):
    manager = EventManager()
    camera = _camera(min_duration_s=min_duration, min_ratio=1)
    left = (0.0, 0.0, 60.0, 80.0)
    right = (50.0, 0.0, 110.0, 80.0)
    ambiguous = (20.0, 0.0, 80.0, 80.0)

    def violations_at(index):
        if index == 0:
            boxes = [left, right]
        else:
            # The first box prefers left, but the second has an exact match.
            # Reserve that exact match first, then match ambiguous to right.
            boxes = [ambiguous, left]
        return [_violation(track_id=None, bbox=bbox) for bbox in boxes]

    created, closed = _feed(manager, camera, range(3), violations_at)
    assert len(created) == 2
    assert len({event.id for event in created}) == 2
    assert all(event.started_at == 0.0 for event in created)
    assert all(
        event.confirmed_at == pytest.approx(min_duration)
        for event in created
    )
    assert closed == []
    opened = manager.open_events()
    assert len(opened) == 2
    assert {event.bbox for event in opened} == {left, ambiguous}


@pytest.mark.parametrize("min_duration", [0, 0.4])
def test_trackless_distance_breaks_equal_iou_ties(min_duration):
    manager = EventManager()
    camera = _camera(min_duration_s=min_duration, min_ratio=1)
    left = (0.0, 0.0, 4.0, 80.0)
    right = (30.0, 0.0, 34.0, 80.0)
    moved = (20.0, 0.0, 24.0, 80.0)

    first, closed = manager.process(
        camera,
        0,
        [
            _violation(track_id=None, bbox=left),
            _violation(track_id=None, bbox=right),
        ],
    )
    assert closed == []
    # Both targets have zero IoU and pass the distance threshold.
    # Right is nearer, despite being inserted after left.
    later, closed = _feed(
        manager,
        camera,
        range(1, 3),
        lambda _: [_violation(track_id=None, bbox=moved)],
    )
    assert closed == []
    if min_duration == 0:
        assert len(first) == 2
        assert later == []
        right_id = next(event.id for event in first if event.bbox == right)
        opened = {event.id: event for event in manager.open_events()}
        assert len(opened) == 2
        assert opened[right_id].bbox == moved
        assert {event.bbox for event in opened.values()} == {left, moved}
    else:
        assert first == []
        assert len(later) == 1
        assert later[0].started_at == 0.0
        assert later[0].confirmed_at == pytest.approx(0.4)
        assert later[0].bbox == moved


@pytest.mark.parametrize("changed_field", ["zone_id", "rule_id"])
def test_trackless_event_matching_respects_rule_and_zone(changed_field):
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    camera.rules.append(
        RuleConfig(rule_id="r2", type="zone_intrusion", min_duration_s=0)
    )
    first, _ = manager.process(camera, 0, [_violation(track_id=None)])
    options = {changed_field: "z2" if changed_field == "zone_id" else "r2"}
    second, closed = manager.process(
        camera,
        0.2,
        [_violation(track_id=None, **options)],
    )
    assert len(first) == len(second) == 1
    assert second[0].id != first[0].id
    assert getattr(second[0], changed_field) == options[changed_field]
    assert closed == []
    assert len(manager.open_events()) == 2


def test_trackless_event_reopens_during_cooldown_after_box_moves():
    manager = EventManager()
    camera = _camera(min_duration_s=0.4, close_after_s=0.4, cooldown_s=2)
    first, _ = _feed(
        manager,
        camera,
        range(3),
        lambda _: [_violation(track_id=None)],
    )
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])
    assert len(first) == len(closed) == 1
    assert manager.open_events() == []

    moved = (18.0, 8.0, 50.0, 80.0)
    assert manager.process(
        camera,
        2.8,
        [_violation(track_id=None, bbox=moved)],
    ) == ([], [])
    opened = manager.open_events()
    assert len(opened) == 1
    assert opened[0].id == first[0].id
    assert opened[0].started_at == 0.0
    assert opened[0].confirmed_at == pytest.approx(0.4)
    assert opened[0].ended_at is None
    assert opened[0].bbox == moved
    assert closed[0].ended_at == pytest.approx(0.4)


def test_s1_01_three_visits_at_same_place_produce_three_events():
    manager = EventManager()
    camera = _camera(
        min_duration_s=0.4,
        min_ratio=1,
        close_after_s=0.4,
        cooldown_s=30,
    )
    intervals = [(77, 133, 1), (181, 198, 2), (251, 259, 3)]

    def violations_at(index):
        for start, end, track_id in intervals:
            if start <= index <= end:
                return [_violation(track_id=track_id)]
        return []

    created, closed = _feed(manager, camera, range(266), violations_at)
    assert len(created) == len(closed) == 3
    assert len({event.id for event in created}) == 3
    assert [event.track_id for event in created] == [1, 2, 3]
    assert [event.started_at for event in created] == pytest.approx(
        [15.4, 36.2, 50.2]
    )
    assert [event.ended_at for event in closed] == pytest.approx(
        [26.6, 39.6, 51.8]
    )
    assert [event.ended_at - event.started_at for event in closed] == (
        pytest.approx([11.2, 3.4, 1.6])
    )
    assert all(event.details.get("reopened_count", 0) == 0 for event in closed)
    assert manager.open_events() == []


def test_same_track_returns_after_eight_seconds_without_counting_the_pause():
    manager = EventManager()
    camera = _camera(min_duration_s=0.4, close_after_s=0.4, cooldown_s=30)
    first, _ = _feed(manager, camera, range(3), lambda _: [_violation()])
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])
    assert len(first) == len(closed) == 1

    moved = (200.0, 8.0, 232.0, 80.0)
    assert manager.process(camera, 8.4, [_violation(bbox=moved)]) == ([], [])
    reopened = manager.open_events()[0]
    assert reopened.id == first[0].id
    assert reopened.started_at == first[0].started_at
    assert reopened.confirmed_at == first[0].confirmed_at
    assert reopened.details["reopened_count"] == 1
    assert reopened.details["active_duration_s"] == pytest.approx(0.4)
    assert reopened.ended_at is None

    assert manager.process(camera, 8.6, [_violation(bbox=moved)]) == ([], [])
    new, closed_again = manager.process(camera, 9.2, [])
    assert new == []
    assert len(closed_again) == 1
    assert closed_again[0].id == first[0].id
    assert closed_again[0].ended_at == pytest.approx(8.6)
    assert closed_again[0].details["reopened_count"] == 1
    assert closed_again[0].details["active_duration_s"] == pytest.approx(0.6)
    assert "reopened_count" not in first[0].details
    assert "reopened_count" not in closed[0].details
    assert reopened.details["active_duration_s"] == pytest.approx(0.4)

    assert manager.process(camera, 16.6, [_violation()]) == ([], [])
    twice = manager.open_events()[0]
    assert twice.id == first[0].id
    assert twice.details["reopened_count"] == 2
    assert twice.details["active_duration_s"] == pytest.approx(0.6)


@pytest.mark.parametrize("gap, expected_events", [(1.0, 1), (7.0, 2)])
def test_changed_track_uses_reid_gap_and_normal_confirmation(gap, expected_events):
    manager = EventManager()
    camera = _camera(
        min_duration_s=0.4,
        min_ratio=1,
        close_after_s=0.4,
        cooldown_s=30,
    )
    first, _ = _feed(manager, camera, range(3), lambda _: [_violation()])
    _, closed = _feed(manager, camera, range(3, 6), lambda _: [])
    assert len(first) == len(closed) == 1

    # Intersection width 24, union width 40: IoU is exactly 0.6.
    changed = _violation(track_id=2, bbox=(16.0, 8.0, 48.0, 80.0))
    start = 0.4 + gap
    assert manager.process(camera, start, [changed]) == ([], [])
    if expected_events == 1:
        opened = manager.open_events()[0]
        assert opened.id == first[0].id
        assert opened.details["reopened_count"] == 1
        assert opened.details["active_duration_s"] == pytest.approx(0.4)
    else:
        # A rejected reassociation starts a candidate, not an immediate alert.
        assert manager.open_events() == []

    assert manager.process(camera, start + 0.2, [changed]) == ([], [])
    second, ended = manager.process(camera, start + 0.4, [changed])
    assert ended == []
    assert len(first) + len(second) == expected_events
    if expected_events == 2:
        assert second[0].id != first[0].id
        assert second[0].started_at == pytest.approx(start)
        assert second[0].confirmed_at == pytest.approx(start + 0.4)


@pytest.mark.parametrize("gap, reopens", [(1.999, True), (2.0, False), (2.001, False)])
def test_same_track_cooldown_is_strict_and_measured_from_ended_at(gap, reopens):
    manager = EventManager(reid_gap_s=5)
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=2)
    first, _ = manager.process(camera, 0, [_violation()])
    _, closed = manager.process(camera, 0.6, [])
    assert closed[0].ended_at == 0

    second, ended = manager.process(camera, gap, [_violation()])
    assert ended == []
    assert len(second) == (0 if reopens else 1)
    opened = manager.open_events()[0]
    assert (opened.id == first[0].id) is reopens
    if reopens:
        assert opened.details["reopened_count"] == 1


@pytest.mark.parametrize("track_id", [2, None])
@pytest.mark.parametrize("gap, reopens", [(5.0, True), (5.001, False)])
def test_spatial_reid_boundary_is_inclusive_from_ended_at(track_id, gap, reopens):
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=30)
    initial_track = None if track_id is None else 1
    first, _ = manager.process(camera, 0, [_violation(track_id=initial_track)])
    _, closed = manager.process(camera, 0.6, [])
    assert closed[0].ended_at == 0

    second, ended = manager.process(camera, gap, [_violation(track_id=track_id)])
    assert ended == []
    assert len(second) == (0 if reopens else 1)
    opened = manager.open_events()[0]
    assert (opened.id == first[0].id) is reopens


@pytest.mark.parametrize("track_id", [2, None])
@pytest.mark.parametrize(
    "manager_gap, rule_gap, reopens",
    [
        (2.0, None, False),
        (9.0, None, True),
        (9.0, 1.0, False),
        (2.0, 8.0, True),
    ],
)
def test_rule_reid_gap_overrides_manager_default(
    track_id, manager_gap, rule_gap, reopens
):
    manager = EventManager(reid_gap_s=manager_gap)
    params = {} if rule_gap is None else {"reid_gap_s": rule_gap}
    camera = _camera(
        min_duration_s=0,
        close_after_s=0.4,
        cooldown_s=30,
        params=params,
    )
    initial_track = None if track_id is None else 1
    first, _ = manager.process(camera, 0, [_violation(track_id=initial_track)])
    manager.process(camera, 0.6, [])

    second, closed = manager.process(camera, 7, [_violation(track_id=track_id)])
    assert closed == []
    assert len(second) == (0 if reopens else 1)
    assert (manager.open_events()[0].id == first[0].id) is reopens


@pytest.mark.parametrize("track_id", [2, None])
def test_reid_can_outlive_a_shorter_cooldown(track_id):
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=0.5)
    initial_track = None if track_id is None else 1
    first, _ = manager.process(camera, 0, [_violation(track_id=initial_track)])
    manager.process(camera, 0.6, [])

    assert manager.process(camera, 2, [_violation(track_id=track_id)]) == ([], [])
    opened = manager.open_events()[0]
    assert opened.id == first[0].id
    assert opened.details["reopened_count"] == 1


def test_reidentified_track_keeps_same_track_cooldown_after_closure():
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=30)
    first, _ = manager.process(camera, 0, [_violation()])
    assert manager.process(camera, 0.2, [_violation(track_id=2)]) == ([], [])
    manager.process(camera, 0.8, [])

    moved = (200.0, 8.0, 232.0, 80.0)
    assert manager.process(
        camera, 8.2, [_violation(track_id=2, bbox=moved)]
    ) == ([], [])
    opened = manager.open_events()[0]
    assert opened.id == first[0].id
    assert opened.track_id == 1
    assert opened.bbox == moved
    assert opened.details["reopened_count"] == 1
    assert opened.details["active_duration_s"] == pytest.approx(0.2)


@pytest.mark.parametrize("gap, reopens", [(1.0, True), (7.0, False)])
def test_trackless_center_matching_reopens_only_within_reid_gap(gap, reopens):
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=30)
    camera.rules[0].type = "smoking_fire"
    first, _ = manager.process(
        camera,
        0,
        [_violation(
            track_id=None,
            bbox=(0.0, 0.0, 4.0, 80.0),
            kind="smoking_fire",
        )],
    )
    manager.process(camera, 0.6, [])

    # Zero IoU; center distance 20 is below half the box height.
    second, closed = manager.process(
        camera,
        gap,
        [_violation(
            track_id=None,
            bbox=(20.0, 0.0, 24.0, 80.0),
            kind="smoking_fire",
        )],
    )
    assert closed == []
    assert len(second) == (0 if reopens else 1)
    assert (manager.open_events()[0].id == first[0].id) is reopens


@pytest.mark.parametrize("next_track_id, reopens", [(1, True), (2, False), (None, False)])
def test_sparse_frames_close_before_matching(next_track_id, reopens):
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=30)
    first, _ = manager.process(camera, 0, [_violation()])

    second, closed = manager.process(
        camera, 8, [_violation(track_id=next_track_id)]
    )
    assert len(closed) == 1
    assert closed[0].id == first[0].id
    assert closed[0].ended_at == 0
    assert "reopened_count" not in closed[0].details
    assert len(second) == (0 if reopens else 1)
    opened = manager.open_events()[0]
    assert (opened.id == first[0].id) is reopens
    if reopens:
        assert opened.details["reopened_count"] == 1
        assert opened.details["active_duration_s"] == 0


@pytest.mark.parametrize("next_track_id", [2, None])
def test_open_event_can_survive_id_change_beyond_reid_gap(next_track_id):
    manager = EventManager(reid_gap_s=1)
    camera = _camera(min_duration_s=0, close_after_s=3, cooldown_s=30)
    first, _ = manager.process(camera, 0, [_violation()])
    assert manager.process(
        camera, 2, [_violation(track_id=next_track_id)]
    ) == ([], [])
    opened = manager.open_events()[0]
    assert opened.id == first[0].id
    assert "reopened_count" not in opened.details


def test_temporarily_missing_track_id_then_new_id_keeps_one_event():
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    first, _ = manager.process(camera, 0, [_violation()])
    assert manager.process(camera, 0.2, [_violation(track_id=None)]) == ([], [])
    assert manager.process(camera, 0.4, [_violation(track_id=2)]) == ([], [])
    assert [event.id for event in manager.open_events()] == [first[0].id]


def test_trackless_observation_cannot_steal_present_tracks_event():
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    first, _ = manager.process(camera, 0, [_violation()])
    second, closed = manager.process(
        camera, 0.2, [_violation(track_id=None), _violation()]
    )
    assert closed == []
    assert len(second) == 1
    assert second[0].id != first[0].id
    assert second[0].track_id is None
    assert len(manager.open_events()) == 2


def test_two_trackless_observations_cannot_share_a_tracked_event():
    manager = EventManager()
    camera = _camera(min_duration_s=0)
    first, _ = manager.process(camera, 0, [_violation()])
    second, closed = manager.process(
        camera,
        0.2,
        [
            _violation(track_id=None),
            _violation(track_id=None, bbox=(12.0, 8.0, 44.0, 80.0)),
        ],
    )
    assert closed == []
    assert len(second) == 1
    assert second[0].id != first[0].id
    assert len(manager.open_events()) == 2


@pytest.mark.parametrize("track_id", [2, None])
def test_reid_of_tracked_event_requires_iou_not_only_center(track_id):
    manager = EventManager()
    camera = _camera(min_duration_s=0, close_after_s=0.4, cooldown_s=30)
    first, _ = manager.process(
        camera, 0, [_violation(bbox=(0.0, 0.0, 4.0, 80.0))]
    )
    manager.process(camera, 0.6, [])
    second, closed = manager.process(
        camera,
        1,
        [_violation(track_id=track_id, bbox=(20.0, 0.0, 24.0, 80.0))],
    )
    assert closed == []
    assert len(second) == 1
    assert second[0].id != first[0].id


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None, "bad"])
def test_invalid_manager_reid_gap_is_rejected(value):
    with pytest.raises(ValueError, match="reid_gap_s"):
        EventManager(reid_gap_s=value)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), None, "bad"])
def test_invalid_rule_reid_gap_does_not_advance_camera_time(value):
    manager = EventManager()
    camera = _camera(min_duration_s=0, params={"reid_gap_s": value})
    with pytest.raises(ValueError, match="reid_gap_s"):
        manager.process(camera, 10, [_violation()])

    camera.rules[0].params = {}
    created, closed = manager.process(camera, 0, [_violation()])
    assert len(created) == 1
    assert closed == []
