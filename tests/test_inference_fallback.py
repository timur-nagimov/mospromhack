"""Офлайн-проверки переключения, восстановления, интерфейса и health."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import ModelSpec
from guard.inference.base import InferenceBackend
from guard.inference.fallback import BackendUnavailable, FallbackBackend
from guard.inference import fallback as fb
from guard.pipeline.engine import PipelineEngine


class FakeBackend(InferenceBackend):
    name = "fake"

    def __init__(self, name="linq-tpu", value=1.0, batch_size=4, fixed=False):
        self.name = name
        self.value = value
        self.batch_size = batch_size
        self.fixed = fixed
        self.window = 2
        self.specs = {}
        self.load_calls = []
        self.calls = []
        self.failure = None
        self.load_failure = None
        self.closed = 0
        self.stats_data = {}

    def load(self, spec):
        self.load_calls.append(spec)
        if self.load_failure is not None:
            raise self.load_failure
        self.specs[spec.model_id] = spec

    def infer(self, model_id, batch):
        return self.infer_timed(model_id, batch)[0]

    def infer_timed(self, model_id, batch):
        self.calls.append((model_id, batch.copy()))
        if self.failure is not None:
            raise self.failure
        assert model_id in self.specs
        if self.fixed:
            assert batch.shape[0] == self.batch_size
        return [batch + self.value], {"infer": 1.0, "transfer": 0.0}

    def max_batch_size(self, model_id):
        return self.batch_size

    def class_names(self, model_id):
        return ["person"] if model_id in self.specs else None

    def loaded_models(self):
        return sorted(self.specs)

    def stats(self):
        return {"backend": self.name, "models": {}, **self.stats_data}

    def close(self):
        self.closed += 1


@pytest.fixture
def spec():
    return ModelSpec(
        model_id="m",
        onnx_path="m.onnx",
        tpu_program="file:///fake/m.tpu",
        input_size=(8, 8),
        output_format="raw",
    )


@pytest.fixture
def batch():
    return np.arange(3 * 3 * 8 * 8, dtype=np.float32).reshape(3, 3, 8, 8)


@pytest.fixture
def clock(monkeypatch):
    current = [100.0]
    monkeypatch.setattr(fb, "time", SimpleNamespace(monotonic=lambda: current[0]))
    return current


@pytest.mark.parametrize("failure", [BackendUnavailable("VPN down"), RuntimeError("worker failed")])
def test_switch_and_return_with_same_spec(spec, batch, clock, failure):
    primary = FakeBackend()
    secondary = FakeBackend("onnx-cpu", value=2.0, batch_size=1)
    backend = FallbackBackend(primary, secondary)
    try:
        backend.load(spec)
        assert backend.name == "linq-tpu"
        assert secondary.load_calls == []
        np.testing.assert_array_equal(backend.infer("m", batch)[0], batch + 1)
        primary.failure = failure
        outputs, timings = backend.infer_timed("m", batch)
        np.testing.assert_array_equal(outputs[0], batch + 2)
        assert timings == {"infer": 3.0, "transfer": 0.0}
        assert secondary.load_calls == [spec]
        assert secondary.load_calls[0] is spec
        assert secondary.specs["m"].onnx_path == "m.onnx"
        assert backend.loaded_models() == ["m"]
        assert backend.class_names("m") == ["person"]
        assert backend.max_batch_size("m") == 1
        state = backend.stats()
        assert state["active"] == "onnx-cpu"
        assert state["active_backend"] == "onnx-cpu" and state["requested_backend"] == "linq-tpu"
        assert state["fallback_enabled"] is True
        assert state["switches"] == 1
        assert state["available"] is True
        assert state["last_error"]
        primary_calls = len(primary.calls)

        clock[0] += 14.9
        backend.infer("m", batch)
        assert len(primary.calls) == primary_calls
        assert backend.stats()["fallback_seconds"] == pytest.approx(14.9)

        primary.failure = None
        clock[0] = 115.0
        np.testing.assert_array_equal(backend.infer("m", batch)[0], batch + 1)
        state = backend.stats()
        assert state["active"] == "linq-tpu"
        assert state["fallback"] is False
        assert state["switches"] == 2
        assert state["fallback_seconds"] == pytest.approx(15.0)
        assert state["last_error"] is None
        clock[0] = 200.0
        assert backend.stats()["fallback_seconds"] == pytest.approx(15.0)
    finally:
        backend.close()


def test_failed_probe_does_not_hit_primary_on_every_frame(spec, batch, clock):
    primary = FakeBackend()
    primary.failure = OSError("offline")
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    backend = FallbackBackend(primary, secondary)
    try:
        backend.load(spec)
        backend.infer("m", batch)
        assert len(primary.calls) == 1
        clock[0] = 115.0
        backend.infer("m", batch)
        assert len(primary.calls) == 2
        clock[0] = 129.9
        backend.infer("m", batch)
        assert len(primary.calls) == 2
        assert backend.stats()["switches"] == 1
    finally:
        backend.close()


def test_load_failure_uses_cpu_and_primary_is_reloaded_on_probe(spec, batch, clock):
    primary = FakeBackend()
    primary.load_failure = ValueError("missing TPU program")
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    backend = FallbackBackend(primary, secondary)
    try:
        backend.load(spec)
        assert backend.stats()["fallback"]
        assert secondary.specs["m"] is spec
        backend.infer("m", batch)
        primary.load_failure = None
        clock[0] = 115.0
        backend.infer("m", batch)
        assert backend.stats()["active"] == "linq-tpu"
        assert len(primary.load_calls) == 2
    finally:
        backend.close()


def test_static_secondary_batch_and_async_interface(spec, batch):
    primary = FakeBackend()
    primary.failure = RuntimeError("TPU unavailable")
    secondary = FakeBackend("onnx-cpu", value=7.0, batch_size=2, fixed=True)
    backend = FallbackBackend(primary, secondary)
    backend.load(spec)
    try:
        outputs, timings = backend.submit("m", batch).result(timeout=3)
        np.testing.assert_array_equal(outputs[0], batch + 7)
        assert [piece.shape[0] for _, piece in secondary.calls] == [2, 2]
        assert timings["infer"] == pytest.approx(2.0)
        pool = backend._pool
    finally:
        backend.close()
    assert pool is not None
    assert not any(thread.is_alive() for thread in pool._threads)
    assert primary.closed == secondary.closed == 1
    backend.close()
    assert primary.closed == secondary.closed == 1
    assert backend.stats()["available"] is False
    with pytest.raises(BackendUnavailable, match="закрыт"):
        backend.submit("m", batch).result(timeout=1)


def test_both_backends_unavailable_then_cpu_recovers(spec, batch, clock):
    primary = FakeBackend()
    primary.failure = OSError("VPN down")
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    secondary.failure = RuntimeError("CPU failed")
    backend = FallbackBackend(primary, secondary)
    try:
        backend.load(spec)
        with pytest.raises(BackendUnavailable, match="CPU failed"):
            backend.infer("m", batch)
        assert backend.stats()["available"] is False
        secondary.failure = None
        backend.infer("m", batch)
        assert backend.stats()["available"] is True
        assert backend.stats()["active"] == "onnx-cpu"
        assert len(primary.calls) == 1
    finally:
        backend.close()


def test_all_model_loads_failed_are_reported(spec):
    primary = FakeBackend()
    secondary = FakeBackend("onnx-cpu")
    primary.load_failure = ValueError("no TPU")
    secondary.load_failure = FileNotFoundError("no ONNX")
    backend = FallbackBackend(primary, secondary)
    try:
        with pytest.raises(BackendUnavailable, match="no ONNX"):
            backend.load(spec)
        assert not backend.stats()["available"]
        assert backend.loaded_models() == []
    finally:
        backend.close()


def test_one_failed_model_does_not_hide_working_model(spec, batch):
    primary = FakeBackend()
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    backend = FallbackBackend(primary, secondary)
    try:
        backend.load(spec)
        other = spec.model_copy(update={"model_id": "other", "onnx_path": "other.onnx"})
        backend.load(other)
        primary.failure = OSError("offline")
        secondary.failure = RuntimeError("failed")
        with pytest.raises(BackendUnavailable):
            backend.infer("m", batch)
        assert backend.stats()["available"] is True
        with pytest.raises(BackendUnavailable):
            backend.infer("other", batch)
        assert backend.stats()["available"] is False
    finally:
        backend.close()


def test_only_one_recovery_probe_runs(spec, batch, clock):
    primary = FakeBackend()
    primary.failure = OSError("offline")
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    backend = FallbackBackend(primary, secondary)
    entered = threading.Event()
    release = threading.Event()
    try:
        backend.load(spec)
        backend.infer("m", batch)
        original = primary.infer_timed
        primary.failure = None

        def slow_probe(model_id, value):
            entered.set()
            assert release.wait(timeout=3)
            return original(model_id, value)

        primary.infer_timed = slow_probe
        clock[0] = 115.0
        future = backend.submit("m", batch)
        assert entered.wait(timeout=2)
        # Пока проба ждёт, другой вызов обслуживается на CPU.
        outputs = backend.infer("m", batch)
        np.testing.assert_array_equal(outputs[0], batch + secondary.value)
        assert len(primary.calls) == 1
        release.set()
        future.result(timeout=3)
        assert len(primary.calls) == 2
        assert backend.stats()["active"] == "linq-tpu"
    finally:
        release.set()
        backend.close()


def test_get_backend_fail_closed_default_explicit_opt_in_and_kwargs(monkeypatch, spec, batch):
    import guard.inference as inference

    created = []

    class Primary(FakeBackend):
        name = "linq-tpu"

        def __init__(self, host=None, models_dir=None):
            super().__init__("linq-tpu")
            self.host = host
            self.models_dir = models_dir
            created.append(self)

    class Secondary(FakeBackend):
        name = "onnx-cpu"

        def __init__(self, models_dir=None, intra_op_threads=None, max_batch=8):
            super().__init__("onnx-cpu", batch_size=max_batch)
            self.models_dir = models_dir
            self.intra_op_threads = intra_op_threads
            created.append(self)

    modules = {
        "guard.inference.linq_tpu": SimpleNamespace(LinqTpuBackend=Primary),
        "guard.inference.onnx_cpu": SimpleNamespace(OnnxCpuBackend=Secondary),
    }
    imports = []

    def import_backend(name):
        imports.append(name)
        return modules[name]

    monkeypatch.setattr(inference.importlib, "import_module", import_backend)
    monkeypatch.delenv("GUARD_BACKEND_FALLBACK", raising=False)
    strict = inference.get_backend("linq-tpu", host="fake")
    assert isinstance(strict, Primary) and len(created) == 1
    assert imports == ["guard.inference.linq_tpu"]
    strict.load(spec)
    strict.failure = BackendUnavailable("TPU unavailable")
    with pytest.raises(BackendUnavailable, match="TPU unavailable"):
        strict.infer("m", batch)
    assert len(created) == 1  # CPU was neither imported/constructed nor called.
    strict.close()
    backend = inference.get_backend(
        "linq-tpu", fallback="onnx-cpu", host="fake", models_dir="/fake/models", intra_op_threads=2, max_batch=3,
    )
    try:
        assert isinstance(backend, FallbackBackend)
        assert backend.primary.host == "fake"
        assert backend.secondary.models_dir == "/fake/models"
        assert backend.secondary.intra_op_threads == 2
        assert backend.secondary.batch_size == 3
        marker = lambda: "transport"
        backend._client = marker
        assert backend.primary._client is marker
    finally:
        backend.close()

    monkeypatch.setenv("GUARD_BACKEND_FALLBACK", "none")
    direct = inference.get_backend("tpu", host="fake")
    assert isinstance(direct, Primary)
    direct.close()

    monkeypatch.setenv("GUARD_BACKEND", "linq-tpu")
    monkeypatch.setenv("GUARD_BACKEND_FALLBACK", "onnx-cpu")
    from_env = inference.get_backend()
    assert isinstance(from_env, FallbackBackend)
    from_env.close()
    forced_strict = inference.get_backend("tpu", fallback="none")
    assert isinstance(forced_strict, Primary)
    forced_strict.close()

    cpu = inference.get_backend("cpu")
    assert isinstance(cpu, Secondary)
    cpu.close()
    monkeypatch.setenv("GUARD_BACKEND_FALLBACK", "invalid")
    with pytest.raises(ValueError, match="GUARD_BACKEND_FALLBACK"):
        inference.get_backend("linq-tpu")


def make_engine(backend, errors=(), *, age=0.0):
    engine = PipelineEngine.__new__(PipelineEngine)
    # _health_base читает settings.mode (режимы local/edge/gateway, A3).
    engine.settings = SimpleNamespace(mode="local")
    engine.backend = backend
    engine._stopped = False
    engine._lock = threading.RLock()
    engine.started_at = time.time()
    engine.workers = {
        str(index): SimpleNamespace(
            running=True,
            errors=dict(values),
            last=SimpleNamespace(done_mono=time.monotonic() - age),
            detectors={"m": object()},
        )
        for index, values in enumerate(errors)
    }
    engine.cameras = dict(engine.workers)
    return engine


@pytest.mark.parametrize(
    ("stats", "expected"),
    [
        ({}, "ok"),
        ({"active": "onnx-cpu", "fallback": True, "available": True}, "degraded"),
        ({"active": "onnx-cpu", "fallback": True, "available": False}, "error"),
        ({"circuit_open": True}, "error"),
        ({"closed": True}, "error"),
        ({"active": "onnx-cpu", "fallback": True, "circuit_open": True, "available": True}, "degraded"),
    ],
)
def test_health_three_states(stats, expected):
    backend = FakeBackend()
    backend.stats_data = stats
    engine = make_engine(backend)
    result = engine.health()
    assert result["status"] == expected
    assert result["inference"] == {
        "active": stats.get("active", backend.name),
        "circuit_open": stats.get("circuit_open", False),
        "last_error": None,
    }
    assert result["cameras_total"] == 0
    assert result["backend"] == "linq-tpu"


def test_health_current_camera_errors_and_recovery():
    backend = FakeBackend()
    engine = make_engine(backend, [{"infer:m": "failed"}, {}])
    assert engine.health()["status"] == "ok"
    engine.workers["1"].errors["infer:m"] = "failed"
    result = engine.health()
    assert result["status"] == "degraded"
    assert result["inference"]["last_error"] == "failed"
    engine.workers["1"].errors.clear()
    assert engine.health()["status"] == "ok"
    engine.workers["0"].running = False
    assert engine.health()["status"] == "ok"


def test_health_ignores_stale_camera_errors_and_cumulative_counters():
    backend = FakeBackend()
    backend.stats_data = {"models": {"m": {"errors": 100, "last_error": "historical"}}}
    engine = make_engine(backend, [{"infer:m": "old"}], age=31.0)
    assert engine.health()["status"] == "ok"
    engine.workers["0"].last = None
    assert engine.health()["status"] == "degraded"
    engine.workers["0"].errors.clear()
    assert engine.health()["status"] == "ok"


def test_health_missing_models_and_stats_failure():
    backend = FakeBackend()
    engine = make_engine(backend, [{"model:m": "missing ONNX"}])
    engine.workers["0"].detectors = {}
    assert engine.health()["status"] == "error"
    engine.workers.clear()

    def broken_stats():
        raise RuntimeError("stats failed")

    backend.stats = broken_stats
    result = engine.health()
    assert result["status"] == "degraded"
    assert "stats failed" in result["inference"]["last_error"]


def test_health_tracks_real_fallback_state(spec, batch, clock):
    primary = FakeBackend()
    secondary = FakeBackend("onnx-cpu", batch_size=1)
    backend = FallbackBackend(primary, secondary)
    engine = make_engine(backend)
    try:
        backend.load(spec)
        assert engine.health()["status"] == "ok"
        primary.failure = OSError("VPN down")
        backend.infer("m", batch)
        assert engine.health()["status"] == "degraded"
        secondary.failure = RuntimeError("CPU unavailable")
        with pytest.raises(BackendUnavailable):
            backend.infer("m", batch)
        assert engine.health()["status"] == "error"
        primary.failure = None
        clock[0] = 115.0
        backend.infer("m", batch)
        assert engine.health()["status"] == "ok"
    finally:
        backend.close()


def test_engine_stop_closes_scheduler_and_media_before_backend_then_journal():
    calls = []
    backend = FakeBackend()
    backend.close = lambda: calls.append("backend")
    engine = make_engine(backend)
    engine._schedulers = {"m": SimpleNamespace(stop=lambda: calls.append("scheduler"))}
    engine.media = SimpleNamespace(stop=lambda: calls.append("media"))
    engine.journal = SimpleNamespace(close=lambda: calls.append("journal"))
    engine.stop()
    assert calls == ["scheduler", "media", "backend", "journal"]
