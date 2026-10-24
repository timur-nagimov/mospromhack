from datetime import datetime, timezone

import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, EventStatus, FrameContext, RuleConfig, SourceConfig, Track, Zone
from guard.events.manager import EventManager
from guard.pipeline import compat
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.rules.plugins.zone_overcapacity import ZoneOvercapacityRule
from guard.tracking import Tracker
from ml.eval.run_s1 import persist_lifecycle_updates


def setup(**params):
    config = RuleConfig(rule_id="capacity", type="zone_overcapacity", zones=["left"],
                        params={"capacity": 1, **params}, min_duration_s=1, min_ratio=1,
                        close_after_s=.5, cooldown_s=0)
    camera = CameraConfig(camera_id="a", name="a", site="synthetic", source=SourceConfig(type="file", uri="unused"),
        zones=[Zone(zone_id="left", name="Left", polygon=[(0, .1), (.65, .1), (.65, 1), (0, 1)]),
               Zone(zone_id="right", name="Right", polygon=[(.7, .1), (1, .1), (1, 1), (.7, 1)])], rules=[config])
    return camera, ZoneOvercapacityRule(config, camera)


def person(index, ts=10., x=None, **kwargs):
    return Track(track_id=index, cls="person", bbox=(x or 10 + index * 15, 10, (x or 10 + index * 15) + 12, 90),
                 conf=.95, first_seen=0, last_seen=ts, hits=3, **kwargs)


def context(camera, people=(), ts=10., status="observed"):
    return FrameContext(camera=camera, ts=ts, frame_idx=int(ts * 10) if np.isfinite(ts) else 0,
                        frame=np.zeros((100, 100, 3), np.uint8), people=list(people),
                        model_observations={} if status is None else {"det": status})


def test_threshold_and_stable_group_bbox_and_registry():
    camera, rule = setup(capacity=2)
    for ts, count in ((1., 1), (2., 2), (3., 3)):
        rows = rule.evaluate(context(camera, [person(i, ts) for i in range(1, count + 1)], ts))
        assert len(rows) == (count > 2)
    event = rows[0]
    assert event.track_id is None and event.zone_id == "left"
    assert event.bbox == (0, 10, 65, 100)
    assert event.details["observed_count"] == 3 and event.details["participant_track_ids"] == [1, 2, 3]
    load_plugins()
    assert get_rule("zone_overcapacity") is ZoneOvercapacityRule
    metadata = next(row for row in available_rules() if row["type"] == rule.type)
    assert metadata["required_models"] == ["det"] and metadata["params_schema"]["required"] == ["capacity"]


@pytest.mark.parametrize("status", [None, "error", "unsupported", "not_observable", "invalid"])
def test_model_unknown_or_error_does_not_become_observed_empty(status):
    camera, rule = setup()
    ctx = context(camera, [person(1), person(2)], status=status)
    assert rule.evaluate(ctx) == []
    assert ctx.rule_observations["capacity"] == ("not_observable" if status is None else "error" if status == "invalid" else status)
    empty = context(camera, [], ts=11)
    assert rule.evaluate(empty) == [] and empty.rule_observations["capacity"] == "observed"


@pytest.mark.parametrize("status", ["error", "unsupported"])
def test_clock_anomaly_does_not_hide_model_failure(status):
    camera, rule = setup()
    rule.evaluate(context(camera))
    ctx = context(camera, ts=9, status=status)
    assert rule.evaluate(ctx) == [] and ctx.rule_observations["capacity"] == status


def test_stale_low_conf_small_outside_and_nonfinite_tracks_are_not_counted():
    camera, rule = setup()
    low, small, invalid = person(2), person(3), person(4)
    low.conf = .1
    small.bbox = (30, 80, 40, 90)
    invalid.bbox = (float("nan"), 10, 50, 90)
    ctx = context(camera, [person(1), person(2, 9), low, small, invalid, person(5, x=80)])
    assert rule.evaluate(ctx) == []


def test_duplicate_identity_and_nearly_identical_boxes_count_once():
    camera, rule = setup()
    a, duplicate = person(1, x=10), person(2, x=10)
    assert rule.evaluate(context(camera, [a, a, duplicate])) == []
    a.last_seen = duplicate.last_seen = 11
    rows = rule.evaluate(context(camera, [a, a, duplicate, person(3, 11, x=40)], ts=11))
    assert rows[0].details["observed_count"] == 2
    assert rows[0].details["suppressed_near_duplicate_track_ids"] == [2]


def test_inactive_zone_foreign_camera_and_clock_replay_are_not_evidence():
    camera, rule = setup()
    camera.zones[0].active_hours = ["11:00-12:00"]
    early = datetime(2026, 1, 1, 8, tzinfo=timezone.utc).timestamp()
    later = early + 7200
    assert rule.evaluate(context(camera, [person(1, early), person(2, early)], early))
    assert not rule.evaluate(context(camera, [person(1, later), person(2, later)], later))
    ctx = context(camera, [person(1, later), person(2, later)], later)
    assert not rule.evaluate(ctx) and ctx.rule_observations["capacity"] == "not_observable"
    assert not rule.evaluate(ctx)  # repeated timestamps never become eligible again
    other = camera.model_copy(update={"camera_id": "b"})
    assert not rule.evaluate(context(other, [person(1, later + 1), person(2, later + 1)], later + 1))


@pytest.mark.parametrize("params", [{}, {"capacity": 0}, {"capacity": True}, {"capacity": 1.5},
    {"capacity": 1, "min_conf": float("nan")}, {"capacity": 1, "min_conf": 2},
    {"capacity": 1, "min_track_hits": False}, {"capacity": 1, "min_box_h_px": -1}, {"capacity": 1, "extra": 1}])
def test_invalid_parameters_are_rejected(params):
    camera, _ = setup()
    cfg = camera.rules[0].model_copy(update={"params": params})
    with pytest.raises(ValueError):
        ZoneOvercapacityRule(cfg, camera)


@pytest.mark.parametrize("zones", [None, [], ["left", "right"], ["missing"]])
def test_one_explicit_existing_zone_is_required(zones):
    camera, _ = setup()
    with pytest.raises(ValueError, match="zone"):
        ZoneOvercapacityRule(camera.rules[0].model_copy(update={"zones": zones}), camera)


def test_real_tracker_rule_manager_journal_and_group_membership_change(tmp_path):
    camera, rule = setup(min_track_hits=1)
    tracker, manager = Tracker(classes={"person"}, min_hits=1), EventManager()
    created = []
    with compat.Journal(tmp_path / "events.db") as journal:
        for ts in (0., .5, 1.):
            people = tracker.update([Detection(model="det", cls="person", cls_id=0, conf=.95, bbox=(10, 10, 22, 90)),
                                     Detection(model="det", cls="person", cls_id=0, conf=.95, bbox=(40, 10, 52, 90))], ts)
            new, closed, reopened = manager.process_with_updates(camera, ts, rule.evaluate(context(camera, people, ts)))
            for event in new:
                journal.add(event)
                created.append(event)
            persist_lifecycle_updates(journal, closed, reopened)
        assert len(created) == 1 and created[0].track_id is None
        identifier = created[0].id
        journal.update(identifier, status=EventStatus.confirmed, operator_note="operator evidence")
        swapped = [person(100, 1.2, x=15), person(101, 1.2, x=45)]
        new, closed, reopened = manager.process_with_updates(camera, 1.2, rule.evaluate(context(camera, swapped, 1.2)))
        assert not new and not closed and not reopened
        _, closed, reopened = manager.process_with_updates(camera, 2., rule.evaluate(context(camera, [], 2.)))
        persist_lifecycle_updates(journal, closed, reopened)
        saved = journal.get(identifier)
        assert saved.ended_at == 1.2 and saved.status == EventStatus.confirmed
        assert saved.operator_note == "operator evidence"
        assert journal.query(limit=10)[1] == 1
