"""Explicit current-frame coverage, with synthetic detections and real state."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, FrameContext, RuleConfig, Track
from guard.metrics import MetricsRegistry
from guard.pipeline import compat
from guard.pipeline.testing import make_settings
from guard.pipeline.worker import CameraWorker
from guard.rules.base import RulePlugin
from guard.sources import FramePacket


def camera():
    return CameraConfig(camera_id="coverage", name="unit", site="unit", source={"type": "file", "uri": "never-opened"})


def context(**kwargs):
    return FrameContext(camera=camera(), ts=0, frame_idx=0, frame=np.zeros((24, 32, 3), np.uint8), **kwargs)


class CoverageRule(RulePlugin):
    required_models = frozenset({"det", "ppe"})

    def evaluate(self, ctx):
        status = self.observation_status(ctx)
        ctx.rule_observations[self.cfg.rule_id] = status
        return []


@pytest.mark.parametrize("models,expected", [
    ({}, "not_observable"), ({"det": "observed"}, "not_observable"),
    ({"det": "observed", "ppe": "observed"}, "observed"),
    ({"det": "observed", "ppe": "not_observable"}, "not_observable"),
    ({"det": "not_observable", "ppe": "unsupported"}, "unsupported"),
    ({"det": "error", "ppe": "unsupported"}, "error"),
    ({"det": "observed", "ppe": "unknown-value"}, "error"),
    ({"det": "observed", "ppe": None}, "error"),
    ({"det": "observed", "ppe": "observed", "irrelevant": "error"}, "observed"),
])
def test_required_model_status_combines_priority_without_mutating_context(models, expected):
    ctx = context(model_observations=models)
    rule = CoverageRule(RuleConfig(rule_id="coverage", type="unit"), ctx.camera)
    before = deepcopy(ctx.model_observations)
    assert rule.observation_status(ctx) == expected
    assert ctx.model_observations == before and ctx.rule_observations == {}


def test_context_default_maps_are_independent_and_no_required_models_is_observed():
    first, second = context(), context()
    first.model_observations["det"] = "error"
    first.rule_observations["one"] = "error"
    assert second.model_observations == second.rule_observations == {}
    rule = CoverageRule(RuleConfig(rule_id="coverage", type="unit"), second.camera)
    rule.required_models = frozenset()
    assert rule.observation_status(second) == "observed"


def test_new_plugin_can_explicitly_report_without_automatic_helper_side_effects():
    ctx = context(model_observations={"det": "observed", "ppe": "error"})
    rule = CoverageRule(RuleConfig(rule_id="coverage", type="unit"), ctx.camera)
    assert rule.evaluate(ctx) == [] and ctx.rule_observations == {"coverage": "error"}


class ScriptedDetector:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0
        self.before_call = lambda: None

    def detect(self, frame, timings=None, **kwargs):
        self.before_call()
        self.calls += 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        if timings is not None:
            timings["infer"] = timings.get("infer", 0) + 1
        return value


def worker_fixture(tmp_path, det, ppe, mode="full_frame"):
    from guard.rules.base import load_plugins
    load_plugins()
    cfg = CameraConfig(camera_id="coverage", name="unit", site="unit", source={"type": "file", "uri": "never-opened"},
        rules=[RuleConfig(rule_id="ppe", type="ppe_missing", min_duration_s=100)])
    detectors = {"det": ScriptedDetector(det), "ppe": ScriptedDetector(ppe)}
    engine = SimpleNamespace(settings=make_settings(tmp_path, tracker_min_hits=1, ppe_input_mode=mode),
        metrics=MetricsRegistry(), get_detector=detectors.__getitem__, events=compat.EventManager(), run_id="unit",
        on_analysis=lambda *args: None, handle_events=lambda *args, **kwargs: pytest.fail("No alarm expected"),
        close_capture=lambda *args: None)
    worker = CameraWorker(cfg, engine)
    worker.rules = [CoverageRule(RuleConfig(rule_id="coverage", type="unit"), cfg)]
    return worker, detectors


def process(worker, ts, capture="one"):
    return worker.process(FramePacket(np.zeros((240, 140, 3), np.uint8), ts, round(ts * 5) + 1, 0, capture_id=capture))


def test_worker_valid_empty_is_observed_error_is_distinct_and_next_frame_is_fresh(tmp_path):
    worker, _ = worker_fixture(tmp_path, [[], RuntimeError("async backend failed"), []], [[], [], []])
    assert "model_observations" not in worker.status()
    first = process(worker, 0)
    assert first.detections == {"det": [], "ppe": []}
    assert first.model_observations == {"det": "observed", "ppe": "observed"}
    assert first.rule_observations == {"coverage": "observed"}
    failed = process(worker, .2)
    assert failed.detections == {"det": [], "ppe": []}
    assert failed.model_observations == {"det": "error", "ppe": "observed"}
    assert failed.rule_observations == {"coverage": "error"}
    assert "infer:det" in worker.errors and worker.metrics.errors == 1
    current = process(worker, .4)
    assert current.model_observations == first.model_observations
    assert current.rule_observations == {"coverage": "observed"} and not worker.errors
    assert failed.model_observations["det"] == "error"  # old snapshots are independent
    status = worker.status()
    status["model_observations"]["det"] = "unsupported"
    status["rule_observations"]["coverage"] = "error"
    assert worker.last_context is current and current.model_observations["det"] == "observed"
    assert worker.status()["rule_observations"] == {"coverage": "observed"}


def test_crop_no_people_or_stale_is_not_observable_failure_masks_old_false_and_successful_empty_runs(tmp_path):
    person = [Detection("det", "person", 0, .9, (0, 0, 100, 200))]
    negative = [Detection("ppe", "no_helmet", 0, .9, (40, 10, 60, 30))]
    worker, detectors = worker_fixture(tmp_path, [person, person, person, [], person, person],
        [negative, negative, negative, RuntimeError("ROI failed"), []], "person_crop")
    for ts in (0, .2, .4):
        current = process(worker, ts)
        assert current.model_observations == {"det": "observed", "ppe": "observed"}
    assert current.people[0].attributes["helmet"] is False
    # The production tracker currently returns observed tracks only. Exercise
    # the compatible retained-track input explicitly without changing its policy.
    update = worker.person_tracker.update
    retained = current.people[0]
    worker.person_tracker.update = lambda detections, ts: [retained]
    stale = process(worker, .6)
    worker.person_tracker.update = update
    assert stale.people and stale.people[0].last_seen == .4 and stale.people[0].attributes["helmet"] is None
    assert stale.model_observations == {"det": "observed", "ppe": "not_observable"}
    assert stale.rule_observations == {"coverage": "not_observable"} and detectors["ppe"].calls == 3
    failed = process(worker, .8)
    assert failed.model_observations["ppe"] == "error" and failed.people[0].attributes["helmet"] is None
    assert failed.rule_observations == {"coverage": "error"} and "infer:ppe" in worker.errors
    recovered = process(worker, 1)
    assert recovered.model_observations["ppe"] == "observed" and recovered.detections["ppe"] == []
    assert recovered.people[0].attributes["_tracking_attribute_state"]["ppe"]["helmet"][-1] == (1, None)
    assert detectors["ppe"].calls == 5 and not worker.errors and worker.metrics.errors == 1


def test_crop_valid_zero_eligible_never_calls_ppe_and_invalid_roi_is_error(tmp_path):
    worker, detectors = worker_fixture(tmp_path, [[], []], [], "person_crop")
    empty = process(worker, 0)
    assert empty.people == [] and empty.model_observations["ppe"] == "not_observable"
    assert detectors["ppe"].calls == 0
    worker.person_tracker.update = lambda dets, ts: [Track(1, "person", (400, 0, 500, 200), .9, ts, ts)]
    invalid = process(worker, .2)
    assert invalid.model_observations["ppe"] == "error" and invalid.people[0].attributes["helmet"] is None
    assert detectors["ppe"].calls == 0 and worker.metrics.errors == 1


def test_new_capture_clears_last_context_before_detection_and_recreates_rules(tmp_path):
    worker, detectors = worker_fixture(tmp_path, [RuntimeError("old failure"), []], [[], []])
    old_rule = worker.rules[0]
    old = process(worker, 0, "old")
    assert old.model_observations["det"] == "error" and worker.last_context is old

    def before_new_detection():
        assert worker.last_context is None
        assert "model_observations" not in worker.status()
        assert worker.rules[0] is not old_rule

    detectors["det"].before_call = before_new_detection
    current = process(worker, .2, "new")
    assert current.model_observations == {"det": "observed", "ppe": "observed"}
    assert current.rule_observations == {"coverage": "observed"} and worker.last_context is current
    assert old.rule_observations == {"coverage": "error"}


def test_rule_exception_does_not_claim_observed_coverage(tmp_path):
    worker, _ = worker_fixture(tmp_path, [[]], [[]])

    def fail(ctx):
        ctx.rule_observations["coverage"] = "observed"
        raise RuntimeError("plugin failed")

    worker.rules[0].evaluate = fail
    current = process(worker, 0)
    assert current.model_observations == {"det": "observed", "ppe": "observed"}
    assert current.rule_observations == {"coverage": "error"} and "rule:coverage" in worker.status()["errors"]


@pytest.mark.parametrize("mode,ppe_status,calls", [("full_frame", "observed", 1), ("person_crop", "not_observable", 0)])
def test_s2_real_component_context_propagates_explicit_status_to_opt_in_rule_and_trace(mode, ppe_status, calls):
    from guard.pipeline.ppe import PpePolicy
    from ml.eval import run_s2

    cfg = camera()
    rule = CoverageRule(RuleConfig(rule_id="coverage", type="unit"), cfg)
    detectors = {"det": ScriptedDetector([[]]), "ppe": ScriptedDetector([[]])}
    result = run_s2.process_frame(cfg, np.zeros((240, 140, 3), np.uint8), {"source_time_s": 0, "source_frame_index": 0},
        detectors, compat.Tracker(classes={"person"}), compat.EventManager(), [rule], ppe_policy=PpePolicy(mode))
    assert result["trace"]["model_observations"] == {"det": "observed", "ppe": ppe_status}
    assert result["trace"]["rule_observations"] == {"coverage": ppe_status}
    assert detectors["ppe"].calls == calls


def test_legacy_plugins_do_not_implicitly_call_observation_helper(monkeypatch):
    from guard.rules.base import get_rule
    legacy = ("zone_intrusion", "ppe_missing", "fall_immobile", "vehicle_proximity", "running", "phone_usage",
              "smoking_fire", "fence_climb", "height_no_harness", "unsafe_carry")

    def unexpected(*args):
        pytest.fail("Legacy behavior must not be automatically gated by new metadata")

    monkeypatch.setattr(RulePlugin, "observation_status", unexpected)
    ctx = context(model_observations={"det": "error", "ppe": "not_observable"})
    for name in legacy:
        rule = get_rule(name)(RuleConfig(rule_id=name, type=name), ctx.camera)
        assert rule.evaluate(ctx) == []
    assert ctx.rule_observations == {}
