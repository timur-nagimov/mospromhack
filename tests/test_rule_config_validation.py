"""Constructor validation before durable camera CRUD, never model execution."""
from copy import deepcopy
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, RuleConfig
from guard.pipeline.engine import CameraError, PipelineEngine
from guard.pipeline.testing import make_settings
from guard.rules import base


@pytest.fixture
def strict_rule(monkeypatch):
    base.load_plugins()

    class StrictRule(base.RulePlugin):
        type = "unit_strict_config"
        title = "Strict configuration unit plugin"
        required_models = frozenset({"det"})
        params_schema = {"type": "object", "required": ["limit"], "additionalProperties": False,
                         "properties": {"limit": {"type": "integer", "minimum": 1}}}
        calls = []

        def __init__(self, cfg, camera):
            super().__init__(cfg, camera)
            self.calls.append((cfg, camera))
            if "limit" not in cfg.params:
                raise ValueError("limit is required")
            if set(cfg.params) != {"limit"}:
                raise ValueError("unknown parameter")
            value = cfg.params["limit"]
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError("limit must be numeric")
            numeric = float(value)  # genuine OverflowError for oversized integers
            if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
                raise ValueError("limit must be a positive integer")
            self.limit = int(numeric)

        def evaluate(self, context):
            pytest.fail("Configuration validation must not evaluate frames")

    monkeypatch.setitem(base._REGISTRY, StrictRule.type, StrictRule)
    return StrictRule


@pytest.fixture
def engine(tmp_path, monkeypatch, strict_rule):
    class InertWorker:
        created = []

        def __init__(self, cfg, owner):
            self.cfg, self.errors, self.running, self.stops = cfg, {}, False, 0
            self.created.append(self)

        def start(self):
            self.running = True

        def stop(self):
            self.stops += 1
            self.running = False

        def status(self):
            return {"running": self.running, "errors": dict(self.errors)}

    monkeypatch.setattr("guard.pipeline.worker.CameraWorker", InertWorker)
    backend = SimpleNamespace(name="unit-no-models", load=Mock(side_effect=AssertionError("No model loading")),
                              infer=Mock(side_effect=AssertionError("No inference")))
    settings = make_settings(tmp_path, autostart=False)
    owner = PipelineEngine(settings, backend=backend, journal=object())
    owner._persist = Mock(wraps=owner._persist)
    owner.events.reset = Mock(wraps=owner.events.reset)
    owner.test_worker = InertWorker
    # Omit app lifespan: real HTTP routing/validation, no source/media threads.
    app = create_app(settings, engine=owner)
    app.state.engine = owner
    client = TestClient(app)
    yield owner, client
    owner.stop()
    client.close()


def camera(params, *, camera_enabled=False, rule_enabled=True, camera_id="cam"):
    return CameraConfig(camera_id=camera_id, name="configuration unit", site="unit", enabled=camera_enabled,
        source={"type": "file", "uri": "never-opened.mp4"},
        rules=[RuleConfig(rule_id="strict_rule", type="unit_strict_config", enabled=rule_enabled, params=params)])


@pytest.mark.parametrize("camera_enabled,rule_enabled", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("params,exception", [({}, ValueError), ({"limit": None}, TypeError), ({"limit": 10 ** 500}, OverflowError)])
def test_constructor_errors_are_named_camera_errors_for_enabled_and_disabled_configuration(engine, strict_rule, camera_enabled, rule_enabled, params, exception):
    owner, _ = engine
    cfg = camera(params, camera_enabled=camera_enabled, rule_enabled=rule_enabled)
    with pytest.raises(CameraError, match=r"rule strict_rule \(unit_strict_config\): invalid configuration") as error:
        owner.add_camera(cfg)
    assert isinstance(error.value.__cause__, exception)
    assert len(strict_rule.calls) == 1
    owner._persist.assert_not_called()
    assert not owner.settings.cameras_config.exists() and owner.cameras == {} and owner.workers == {}
    assert owner.test_worker.created == []
    owner.backend.load.assert_not_called()
    owner.backend.infer.assert_not_called()


@pytest.mark.parametrize("camera_enabled,rule_enabled", [(False, False), (False, True), (True, False), (True, True)])
def test_valid_constructor_parameters_are_accepted_without_mutating_configuration(engine, strict_rule, camera_enabled, rule_enabled):
    owner, _ = engine
    cfg = camera({"limit": 3}, camera_enabled=camera_enabled, rule_enabled=rule_enabled)
    before = cfg.model_dump(mode="json")
    owner.validate_camera(cfg)
    assert cfg.model_dump(mode="json") == before
    assert strict_rule.calls == [(cfg.rules[0], cfg)]
    assert owner.test_worker.created == []
    owner._persist.assert_not_called()
    owner.backend.load.assert_not_called()


@pytest.mark.parametrize("params", [{}, {"limit": None}, {"limit": 0}, {"limit": 2, "typo": True}])
def test_disabled_camera_with_disabled_strict_rule_returns_422_before_any_persist_or_worker(engine, params):
    owner, client = engine
    response = client.post("/api/cameras", json=camera(params, rule_enabled=False).model_dump(mode="json"))
    assert response.status_code == 422 and "strict_rule" in response.json()["detail"] and "unit_strict_config" in response.json()["detail"]
    assert not owner.settings.cameras_config.exists()
    assert owner.cameras == {} and owner.workers == {} and not owner.test_worker.created
    owner._persist.assert_not_called()
    owner.events.reset.assert_not_called()
    owner.backend.load.assert_not_called()


@pytest.mark.parametrize("old_enabled", [False, True])
def test_invalid_put_is_atomic_and_unsupported_patch_cannot_change_old_config(engine, strict_rule, old_enabled):
    owner, client = engine
    valid = camera({"limit": 2}, camera_enabled=old_enabled)
    created = client.post("/api/cameras", json=valid.model_dump(mode="json"))
    assert created.status_code == 201
    path = owner.settings.cameras_config
    # Preserve comments/format as well as parsed values on validation failure.
    path.write_bytes(b"# existing user formatting\n" + path.read_bytes())
    original = path.read_bytes()
    cameras, workers = owner.cameras, dict(owner.workers)
    active = list(owner.test_worker.created)
    saved = client.get("/api/cameras/cam").json()
    owner._persist.reset_mock()
    owner.events.reset.reset_mock()
    bad = deepcopy(saved)
    bad.update(name="must not persist", enabled=not old_enabled)
    bad["rules"][0]["enabled"] = False
    bad["rules"][0]["params"] = {"limit": None}
    rejected = client.put("/api/cameras/cam", json=bad)
    assert rejected.status_code == 422 and "unit_strict_config" in rejected.json()["detail"]
    assert client.patch("/api/cameras/cam", json={"rules": bad["rules"]}).status_code == 405
    assert path.read_bytes() == original
    assert owner.cameras is cameras and owner.workers == workers
    assert client.get("/api/cameras/cam").json() == saved
    assert owner.test_worker.created == active and all(worker.stops == 0 for worker in active)
    owner._persist.assert_not_called()
    owner.events.reset.assert_not_called()
    owner.backend.load.assert_not_called()
    assert not list(path.parent.glob(".cameras.*.yaml"))


def test_valid_disabled_rule_configuration_is_persisted_and_round_trips(engine, strict_rule):
    from guard.pipeline.config import load_cameras
    owner, client = engine
    cfg = camera({"limit": 4}, rule_enabled=False)
    response = client.post("/api/cameras", json=cfg.model_dump(mode="json"))
    assert response.status_code == 201 and response.json()["rules"][0]["params"] == {"limit": 4}
    assert load_cameras(owner.settings.cameras_config) == owner.list_cameras()
    assert len(strict_rule.calls) == 1 and not owner.test_worker.created
    owner._persist.assert_called_once()
    owner.backend.load.assert_not_called()


def test_existing_zone_window_unknown_type_and_required_model_checks_keep_their_order(engine, strict_rule):
    owner, _ = engine
    cfg = camera({"limit": 1})
    for changed, message in ((cfg.rules[0].model_copy(update={"type": "unit_unknown"}), "unknown type"),
                             (cfg.rules[0].model_copy(update={"zones": ["missing"]}), "unknown zones"),
                             (cfg.rules[0].model_copy(update={"min_ratio": 2}), "min_duration_s")):
        with pytest.raises(CameraError, match=message):
            owner.validate_camera(cfg.model_copy(update={"rules": [changed]}))
    assert not strict_rule.calls
    owner.models = {"pose": object()}
    with pytest.raises(CameraError, match="required models.*det"):
        owner.validate_camera(cfg)
    assert len(strict_rule.calls) == 1
    disabled = cfg.model_copy(update={"rules": [cfg.rules[0].model_copy(update={"enabled": False})]})
    owner.validate_camera(disabled)  # existing missing-model check applies only to enabled rules
    assert len(strict_rule.calls) == 2


def test_all_existing_constructors_validate_without_io_threads_models_or_frame_evaluation(engine, monkeypatch):
    import builtins
    from pathlib import Path
    import socket
    import threading

    owner, _ = engine
    legacy = ("zone_intrusion", "ppe_missing", "fall_immobile", "vehicle_proximity", "running", "phone_usage",
              "smoking_fire", "fence_climb", "height_no_harness", "unsafe_carry")
    cfg = camera({"limit": 1}).model_copy(update={"rules": [RuleConfig(rule_id=name, type=name, enabled=False) for name in legacy]})
    before = cfg.model_dump(mode="json")

    def forbidden(*args, **kwargs):
        pytest.fail("Constructor validation must be model-free and have no I/O/evaluate side effects")

    with monkeypatch.context() as blocked:
        blocked.setattr(builtins, "open", forbidden)
        blocked.setattr(Path, "open", forbidden)
        blocked.setattr(socket, "socket", forbidden)
        blocked.setattr(threading.Thread, "start", forbidden)
        blocked.setattr(owner, "get_detector", forbidden)
        for name in legacy:
            blocked.setattr(base.get_rule(name), "evaluate", forbidden)
        owner.validate_camera(cfg)
    assert cfg.model_dump(mode="json") == before
    assert owner.workers == {} and not owner.test_worker.created
    owner.backend.load.assert_not_called()


@pytest.mark.parametrize("has_forklift", [False, True])
def test_camera_crud_validates_dynamic_vehicle_models_before_persistence(engine, has_forklift):
    owner, client = engine
    owner.models = {"det": object(), **({"forklift": object()} if has_forklift else {})}
    cfg = CameraConfig(
        camera_id="warehouse", name="Warehouse", site="Test site", enabled=False,
        source={"type": "file", "uri": "never-opened.mp4"},
        rules=[RuleConfig(rule_id="vehicle", type="vehicle_proximity",
                          params={"models": ["det", "forklift"]})],
    )
    response = client.post("/api/cameras", json=cfg.model_dump(mode="json"))
    if has_forklift:
        assert response.status_code == 201
        assert response.json()["rules"][0]["params"]["models"] == ["det", "forklift"]
        owner._persist.assert_called_once()
    else:
        assert response.status_code == 422 and "forklift" in response.json()["detail"]
        owner._persist.assert_not_called()
        assert not owner.cameras and not owner.settings.cameras_config.exists()
    # Per-camera dependencies must not mutate the shared plugin metadata.
    assert base.get_rule("vehicle_proximity").required_models == frozenset({"det"})
    owner.backend.load.assert_not_called()
    owner.backend.infer.assert_not_called()
    assert not owner.test_worker.created
