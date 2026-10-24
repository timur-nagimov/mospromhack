"""Сквозные проверки реальных Tracker, attributes, plugins и EventManager.

Ни скорости, ни атрибуты, ни нарушения вручную не подставляются.
Созданные оповещения считаются списком, без удаления дублей по ID.
Задержка подтверждения измеряется от первого Violation правила;
для падения предварительно действует собственный min_lying_s.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from guard.contracts import Event, FrameContext, Severity, Track, Violation
from guard.events.manager import EventManager
from guard.rules.base import available_rules, build_rules, load_plugins
from guard.sim import FPS, FRAME_SIZE, START_TS, Scenario, make_scenario
from guard.tracking import Tracker, assign_attributes

EPS = 1e-6
SEEDS = (0, 17, 41)

# Намеренное присутствие акторов до внесения потерь детектора.
ACTIVE_INTERVALS = {
    "zone_intrusion": ((0.0, 16.0),),
    "ppe_missing": ((0.0, 8.0),),
    "fall_immobile": ((0.0, 12.5),),
    "vehicle_proximity": ((0.0, 14.0),),
    "running": ((0.0, 5.0),),
    "noisy_frame": ((0.0, 10.0),),
    "track_return": ((0.0, 6.0), (8.0, 14.0)),
}


@dataclass(slots=True)
class _Step:
    ts: float
    people: list[Track]
    objects: list[Track]
    violations: list[Violation]
    created: list[Event]
    closed: list[Event]


@dataclass(slots=True)
class _Run:
    steps: list[_Step]
    open_events: list[Event]

    @property
    def created(self) -> list[Event]:
        return [event for step in self.steps for event in step.created]

    @property
    def closed(self) -> list[Event]:
        return [event for step in self.steps for event in step.closed]


def _run(scenario: Scenario) -> _Run:
    # Явная загрузка исключает зависимость от частично заполненного
    # реестра после тестов отдельных плагинов.
    load_plugins()
    rules = build_rules(scenario.camera)
    assert len(rules) == 1
    assert rules[0].type == scenario.camera.rules[0].type

    people_tracker = Tracker(
        classes={"person"},
        track_buffer_s=scenario.track_buffer_s,
    )
    object_tracker = Tracker(
        classes={"truck"},
        track_buffer_s=scenario.track_buffer_s,
    )
    manager = EventManager()
    width, height = FRAME_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    steps = []

    for frame_idx, (ts, detections) in enumerate(scenario.frames):
        people = people_tracker.update(detections["det"], ts)
        objects = object_tracker.update(detections["det"], ts)
        assign_attributes(people, detections, ts)
        ctx = FrameContext(
            camera=scenario.camera,
            ts=ts,
            frame_idx=frame_idx,
            frame=frame,
            detections=detections,
            people=people,
            objects=objects,
        )
        violations = [
            violation
            for rule in rules
            for violation in rule.evaluate(ctx)
        ]
        # Пустые кадры также обязаны попадать в окно подтверждения.
        created, closed = manager.process(scenario.camera, ts, violations)
        steps.append(_Step(
            ts=ts,
            # Tracker переиспользует изменяемые Track: сохраняем снимки.
            people=deepcopy(people),
            objects=deepcopy(objects),
            violations=deepcopy(violations),
            created=created,
            closed=closed,
        ))

    return _Run(
        steps=steps,
        open_events=manager.open_events(scenario.camera.camera_id),
    )


def _assert_one_event(
    scenario: Scenario,
    result: _Run,
    expected_type: str,
    expected_severity: Severity,
) -> Event:
    created = result.created
    assert len(created) == 1, [
        (event.type, event.track_id, event.started_at, event.confirmed_at)
        for event in created
    ]
    event = created[0]
    cfg = scenario.camera.rules[0]

    assert event.type == expected_type
    assert event.severity == expected_severity
    assert event.rule_id == cfg.rule_id
    assert event.camera_id == scenario.camera.camera_id
    assert event.site == scenario.camera.site
    assert event.zone_id == (cfg.zones[0] if cfg.zones else None)

    evidence = [
        (step.ts, violation)
        for step in result.steps
        for violation in step.violations
    ]
    assert evidence
    assert all(v.type == expected_type for _, v in evidence)
    assert all(v.severity == expected_severity for _, v in evidence)
    assert event.track_id == evidence[0][1].track_id
    assert event.started_at == pytest.approx(
        evidence[0][0], rel=0.0, abs=EPS
    )

    delay = event.confirmed_at - evidence[0][0]
    assert delay >= cfg.min_duration_s - EPS
    assert delay <= cfg.min_duration_s + 0.5 + EPS
    assert event.confirmed_at - event.started_at == pytest.approx(
        delay, rel=0.0, abs=EPS
    )

    assert result.closed
    assert all(closed.id == event.id for closed in result.closed)
    assert result.closed[-1].ended_at == pytest.approx(
        evidence[-1][0], rel=0.0, abs=EPS
    )
    assert result.open_events == []
    return event


def test_plugins_are_registered() -> None:
    load_plugins()
    registered = {item["type"] for item in available_rules()}
    assert {
        "zone_intrusion",
        "ppe_missing",
        "fall_immobile",
        "vehicle_proximity",
        "running",
        "phone_usage",
        "smoking_fire",
        "fence_climb",
    } <= registered


@pytest.mark.parametrize("name", tuple(ACTIVE_INTERVALS))
def test_streams_include_dropouts_and_missing_keypoints(name: str) -> None:
    scenario = make_scenario(name)
    intervals = ACTIVE_INTERVALS[name]
    active = []
    poses = []

    for ts, detections in scenario.frames:
        elapsed = round(ts - START_TS, 6)
        present = any(start <= elapsed < end for start, end in intervals)
        if present:
            active.append(detections)
        else:
            assert all(not values for values in detections.values())

        for model_id, values in detections.items():
            for detection in values:
                assert detection.model == model_id
                x1, y1, x2, y2 = detection.bbox
                assert 0 <= x1 < x2 <= FRAME_SIZE[0]
                assert 0 <= y1 < y2 <= FRAME_SIZE[1]
        poses.extend(detections["pose"])

    missing_ratio = sum(not frame["det"] for frame in active) / len(active)
    assert 0.05 <= missing_ratio <= 0.10
    assert any(pose.keypoints is None for pose in poses)
    assert any(
        pose.keypoints is not None
        and np.any(pose.keypoints[:, 2] == 0.0)
        for pose in poses
    )
    assert any(
        pose.keypoints is not None
        and np.all(pose.keypoints[:, 2] >= 0.5)
        for pose in poses
    )
    np.testing.assert_allclose(
        np.diff([ts for ts, _ in scenario.frames]),
        1.0 / FPS,
        rtol=0.0,
        atol=EPS,
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("name", "expected_type", "severity"),
    [
        ("zone_intrusion", "zone_intrusion", Severity.high),
        ("ppe_missing", "ppe_missing", Severity.high),
        ("fall_immobile", "fall_immobile", Severity.critical),
        ("vehicle_proximity", "vehicle_proximity", Severity.critical),
        ("running", "running", Severity.medium),
    ],
)
def test_positive_scenario(
    name: str,
    expected_type: str,
    severity: Severity,
    seed: int,
) -> None:
    scenario = make_scenario(name, seed=seed)
    result = _run(scenario)
    event = _assert_one_event(scenario, result, expected_type, severity)
    assert len(result.closed) == 1

    people = [person for step in result.steps for person in step.people]
    ids = {person.track_id for person in people}
    assert len(ids) == (2 if name == "ppe_missing" else 1)

    if name == "ppe_missing":
        offenders = [
            person for person in people
            if (person.bbox[0] + person.bbox[2]) / 2.0 < 660.0
        ]
        protected = [
            person for person in people
            if (person.bbox[0] + person.bbox[2]) / 2.0 > 660.0
        ]
        assert offenders and protected
        assert all(p.attributes["helmet"] is False for p in offenders)
        assert all(p.attributes["helmet"] is True for p in protected)
        assert {p.track_id for p in offenders} == {event.track_id}
        protected_ids = {p.track_id for p in protected}
        assert all(
            v.track_id not in protected_ids
            for step in result.steps
            for v in step.violations
        )
        assert event.details["missing"] == ["helmet"]

    elif name == "fall_immobile":
        lying = [
            person
            for step in result.steps
            if 2.6 - EPS <= step.ts - START_TS < 12.5
            for person in step.people
        ]
        assert lying
        assert all(p.attributes["pose_state"] == "lying" for p in lying)
        assert all(
            (p.bbox[2] - p.bbox[0]) / (p.bbox[3] - p.bbox[1]) > 1.3
            for p in lying
        )
        complete = [
            p.keypoints
            for p in lying
            if p.keypoints is not None
            and np.all(p.keypoints[[5, 6, 11, 12], 2] >= 0.5)
        ]
        assert complete
        for points in complete:
            torso = (
                points[[11, 12], :2].mean(axis=0)
                - points[[5, 6], :2].mean(axis=0)
            )
            assert abs(torso[0]) > abs(torso[1])
        assert event.details["lying_s"] >= 2.0 - EPS

    elif name == "vehicle_proximity":
        vehicles = [
            obj for step in result.steps for obj in step.objects
        ]
        assert vehicles
        assert {obj.cls for obj in vehicles} == {"truck"}
        assert len({obj.track_id for obj in vehicles}) == 1
        assert event.details["vehicle_track_id"] in {
            obj.track_id for obj in vehicles
        }
        assert 0.0 < event.details["distance_px"] < 90.0

    elif name == "running":
        assert event.details["speed_bh"] >= 1.8


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("close_after_s", (1.0, 3.0))
def test_new_track_id_within_cooldown_does_not_notify_twice(
    seed: int,
    close_after_s: float,
) -> None:
    scenario = make_scenario("track_return", seed=seed)
    cfg = scenario.camera.rules[0]
    cfg.close_after_s = close_after_s
    result = _run(scenario)

    before = [
        step for step in result.steps
        if step.ts < START_TS + 6.0 and step.people
    ]
    after = [
        step for step in result.steps
        if START_TS + 8.0 <= step.ts < START_TS + 14.0 and step.people
    ]
    gap = [
        step for step in result.steps
        if START_TS + 6.0 <= step.ts < START_TS + 8.0
    ]
    assert before and after
    assert len(gap) == 2 * FPS
    assert all(not step.people for step in gap)

    before_ids = {p.track_id for step in before for p in step.people}
    after_ids = {p.track_id for step in after for p in step.people}
    assert len(before_ids) == len(after_ids) == 1
    assert before_ids.isdisjoint(after_ids)
    assert after[0].ts - before[-1].ts < cfg.cooldown_s

    event = _assert_one_event(
        scenario, result, "zone_intrusion", Severity.high
    )
    assert event.track_id in before_ids
    assert any(
        v.track_id in after_ids
        for step in after
        for v in step.violations
    )

    closures_before_return = [
        (step.ts, closed)
        for step in result.steps
        if step.ts < after[0].ts
        for closed in step.closed
    ]
    if close_after_s == 1.0:
        assert len(closures_before_return) == 1
        closed_ts, closed = closures_before_return[0]
        assert closed.id == event.id
        assert after[0].ts - closed_ts < cfg.cooldown_s
        # Закрытие до возврата и повторное закрытие после ухода —
        # обновления одного события, а не новые оповещения.
        assert len(result.closed) == 2
    else:
        assert closures_before_return == []
        assert len(result.closed) == 1


class _SingleFramePpeAlarm(AssertionError):
    """Известное ложное подтверждение после одного отрицательного голоса."""


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.xfail(
    strict=True,
    raises=_SingleFramePpeAlarm,
    reason=(
        "assign_attributes сохраняет единственный голос no_helmet в окне "
        "[ts-2, ts]; без других PPE-голосов helmet=False держится до t+2 "
        "включительно. ppe_missing повторяет Violation, и EventManager "
        "при min_duration_s=2, min_ratio=0.7 создаёт ложное оповещение."
    ),
)
def test_single_false_no_helmet_frame_produces_no_event(seed: int) -> None:
    scenario = make_scenario("noisy_frame", seed=seed)
    false_observations = [
        ts
        for ts, detections in scenario.frames
        for detection in detections["ppe"]
        if detection.cls == "no_helmet"
    ]
    assert false_observations == [START_TS + 4.0]
    assert sum(
        len(detections["ppe"]) for _, detections in scenario.frames
    ) == 1

    result = _run(scenario)
    false_frame = next(
        step for step in result.steps if step.ts == false_observations[0]
    )
    assert len(false_frame.people) == 1

    if result.created:
        # Только известная ошибка подтверждения получает xfail.
        # Ошибки сборки конвейера, геометрии, типа или задержки остаются
        # обычными падениями теста благодаря raises выше.
        event = _assert_one_event(
            scenario, result, "ppe_missing", Severity.high
        )
        assert event.details["missing"] == ["helmet"]
        assert event.started_at == pytest.approx(
            START_TS + 4.0, rel=0.0, abs=EPS
        )
        raise _SingleFramePpeAlarm(
            "Один исходный no_helmet создал оповещение: "
            f"started_at={event.started_at}, "
            f"confirmed_at={event.confirmed_at}"
        )

    assert result.created == []
    assert result.closed == []
    assert result.open_events == []
