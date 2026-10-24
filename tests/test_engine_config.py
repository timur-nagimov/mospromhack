from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from guard.contracts import CameraConfig, SourceConfig
from guard.pipeline import config
from guard.pipeline.engine import CameraError, CameraExists, PipelineEngine
from guard.pipeline.settings import Settings


def camera(camera_id="cam", *, enabled=True, name="Камера"):
    return CameraConfig(
        camera_id=camera_id, name=name, site="Цех", enabled=enabled,
        source=SourceConfig(type="rtsp", uri="rtsp://user:secret@example.invalid/live"),
    )


@pytest.fixture
def fake_worker(monkeypatch):
    class FakeWorker:
        created = []
        fail_phase = None

        def __init__(self, cfg, engine):
            # Даже конструктор настоящего worker сбрасывает метрики камеры.
            assert cfg in config.load_cameras(engine.settings.cameras_config)
            self.cfg = cfg
            self.engine = engine
            self.running = False
            self.errors = {}
            self.stops = 0
            self.created.append(self)
            engine.metrics.reset(cfg.camera_id)
            if self.fail_phase == "init":
                raise OSError(cfg.source.uri)

        def start(self):
            self.running = True
            if self.fail_phase == "start":
                raise OSError(self.cfg.source.uri)

        def stop(self):
            self.stops += 1
            self.running = False

        def status(self):
            return {"running": self.running, "errors": dict(self.errors)}

    monkeypatch.setattr("guard.pipeline.worker.CameraWorker", FakeWorker)
    return FakeWorker


@pytest.fixture
def engine(tmp_path, fake_worker):
    settings = Settings(
        data_dir=tmp_path / "data", models_dir=tmp_path / "models",
        cameras_config=tmp_path / "configs" / "cameras.yaml",
        models_config=tmp_path / "configs" / "models.yaml", db_path=tmp_path / "guard.db",
        media_dir=tmp_path / "media", frontend_dist=tmp_path / "frontend", autostart=False,
    )
    events = SimpleNamespace(open_events=lambda camera_id: [], reset=Mock())
    eng = PipelineEngine(settings, backend=SimpleNamespace(name="fake"), journal=object(), event_manager=events)
    eng.media = SimpleNamespace(remove_camera=Mock())
    eng.add_camera(camera())
    eng.add_camera(camera("other"))
    events.reset.reset_mock()
    return eng


def fail_saving(monkeypatch, phase):
    if phase == "write":
        fdopen = config.os.fdopen

        @contextmanager
        def partial_write(fd, *args, **kwargs):
            with fdopen(fd, *args, **kwargs) as fh:
                def write(text):
                    fh.write(text[:20])
                    raise OSError("disk full")

                yield SimpleNamespace(write=write)

        monkeypatch.setattr(config.os, "fdopen", partial_write)
    else:
        def fail(*args):
            raise OSError(f"{phase} refused")

        monkeypatch.setattr(config.os, phase, fail)


@pytest.mark.parametrize("phase", ["write", "fsync", "replace"])
@pytest.mark.parametrize("operation", ["add", "update", "delete"])
@pytest.mark.parametrize("enabled", [True, False])
def test_save_failure_keeps_disk_and_all_runtime_state(engine, fake_worker, monkeypatch, phase, operation, enabled):
    if not enabled:
        engine.update_camera("cam", camera(enabled=False))
        engine.events.reset.reset_mock()
    path = engine.settings.cameras_config
    original = path.read_bytes()
    cameras = engine.cameras
    workers = dict(engine.workers)
    metrics = engine.metrics.snapshot()
    created = list(fake_worker.created)
    fail_saving(monkeypatch, phase)

    with pytest.raises(OSError):
        if operation == "add":
            engine.add_camera(camera("new", enabled=enabled))
        elif operation == "update":
            engine.update_camera("cam", camera(enabled=enabled, name="Изменённая камера"))
        else:
            engine.delete_camera("cam")

    assert path.read_bytes() == original
    assert not list(path.parent.glob(".cameras.*.yaml"))
    assert engine.cameras is cameras
    assert engine.workers == workers
    assert all(worker.running and worker.stops == 0 for worker in workers.values())
    assert fake_worker.created == created
    assert engine.metrics.snapshot() == metrics
    engine.events.reset.assert_not_called()
    engine.media.remove_camera.assert_not_called()


def test_save_serialization_failure_keeps_original_yaml(tmp_path, monkeypatch):
    path = tmp_path / "cameras.yaml"
    path.write_text("# Preserve even formatting and comments on failure\ncameras: []\n", encoding="utf-8")
    original = path.read_bytes()

    def fail(*args, **kwargs):
        raise ValueError("cannot serialize")

    monkeypatch.setattr(config.yaml, "safe_dump", fail)
    with pytest.raises(ValueError, match="cannot serialize"):
        config.save_cameras(path, [camera()])

    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_save_roundtrip_and_replace_only_after_sync(tmp_path, monkeypatch):
    path = tmp_path / "new" / "cameras.yaml"
    synced = []
    fsync = config.os.fsync
    replace = config.os.replace

    def record_sync(fd):
        fsync(fd)
        synced.append(True)

    def checked_replace(source, destination):
        assert synced == [True]
        assert Path(source).parent == path.parent
        assert not path.exists()
        assert config.load_cameras(source) == [camera(), camera("disabled", enabled=False)]
        replace(source, destination)

    monkeypatch.setattr(config.os, "fsync", record_sync)
    monkeypatch.setattr(config.os, "replace", checked_replace)
    config.save_cameras(path, (cam for cam in [camera(), camera("disabled", enabled=False)]))

    assert config.load_cameras(path) == [camera(), camera("disabled", enabled=False)]
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("enabled", [True, False])
def test_add_starts_only_enabled_new_camera(engine, fake_worker, enabled):
    old_workers = dict(engine.workers)
    created = len(fake_worker.created)
    cam = camera("new", enabled=enabled)

    assert engine.add_camera(cam) == cam

    assert engine.get_camera("new") == cam
    assert config.load_cameras(engine.settings.cameras_config) == engine.list_cameras()
    assert len(fake_worker.created) == created + int(enabled)
    assert ("new" in engine.workers) == enabled
    assert engine.camera_status("new")["running"] == enabled
    assert all(engine.workers[cid] is worker and worker.stops == 0 for cid, worker in old_workers.items())


@pytest.mark.parametrize("was_enabled", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_update_restarts_only_target_camera(engine, was_enabled, enabled):
    if not was_enabled:
        engine.update_camera("cam", camera(enabled=False))
    old_worker = engine.workers.get("cam")
    other = engine.workers["other"]
    engine.events.reset.reset_mock()
    changed = camera(enabled=enabled, name="Новая камера")

    assert engine.update_camera("cam", changed) == changed

    assert config.load_cameras(engine.settings.cameras_config) == engine.list_cameras()
    assert engine.get_camera("cam") == changed
    assert engine.camera_status("cam")["running"] == enabled
    if old_worker:
        assert old_worker.stops == 1
        assert not old_worker.running
        assert engine.workers.get("cam") is not old_worker
    assert engine.workers["other"] is other and other.running and other.stops == 0
    engine.events.reset.assert_called_once_with("cam")
    engine.media.remove_camera.assert_not_called()


@pytest.mark.parametrize("enabled", [True, False])
def test_delete_removes_only_target_runtime_and_config(engine, enabled):
    if not enabled:
        engine.update_camera("cam", camera(enabled=False))
    old_worker = engine.workers.get("cam")
    other = engine.workers["other"]
    engine.events.reset.reset_mock()

    engine.delete_camera("cam")

    assert engine.get_camera("cam") is None
    assert config.load_cameras(engine.settings.cameras_config) == [camera("other")]
    assert engine.workers == {"other": other}
    if old_worker:
        assert old_worker.stops == 1 and not old_worker.running
    assert other.running and other.stops == 0
    assert "cam" not in engine.metrics.snapshot()
    engine.events.reset.assert_called_once_with("cam")
    engine.media.remove_camera.assert_called_once_with("cam")


@pytest.mark.parametrize("operation", ["add", "update"])
@pytest.mark.parametrize("phase", ["init", "start"])
def test_start_failure_keeps_committed_config_and_exposes_error(engine, fake_worker, monkeypatch, operation, phase):
    # После os.replace rollback не обещается: сохранённый конфиг виден,
    # запуск явно завершается ошибкой, GET cameras объясняет отсутствие worker.
    old_worker = engine.workers.get("cam")
    other = engine.workers["other"]
    cam = camera("new" if operation == "add" else "cam", name="Не удалось запустить")
    fake_worker.fail_phase = phase

    with pytest.raises(RuntimeError, match="is configured but worker failed to start") as error:
        if operation == "add":
            engine.add_camera(cam)
        else:
            engine.update_camera(cam.camera_id, cam)

    assert cam.source.uri not in str(error.value)
    assert "secret" not in str(error.value)
    assert engine.get_camera(cam.camera_id) == cam
    assert cam in config.load_cameras(engine.settings.cameras_config)
    assert cam.camera_id not in engine.workers
    status = engine.camera_status(cam.camera_id)
    assert not status["running"]
    assert "OSError" in status["errors"]["start"]
    assert "secret" not in status["errors"]["start"]
    failed_worker = fake_worker.created[-1]
    if phase == "start":
        assert failed_worker.stops == 1 and not failed_worker.running
    if operation == "update":
        assert old_worker.stops == 1 and not old_worker.running
    assert engine.workers["other"] is other and other.running and other.stops == 0

    # Используем существующий HTTP-контракт без запуска lifespan/источников.
    # Настройки задаём явно, чтобы импорт приложения не загружал реальный .env.
    monkeypatch.setattr(Settings, "from_env", lambda: engine.settings)
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from guard.api.app import router

    app = FastAPI()
    app.state.engine = engine
    app.include_router(router)
    response = TestClient(app).get("/api/cameras")
    assert response.status_code == 200
    saved = next(item for item in response.json() if item["camera_id"] == cam.camera_id)
    assert saved["status"] == status

    fake_worker.fail_phase = None
    engine.update_camera(cam.camera_id, cam)
    assert engine.camera_status(cam.camera_id) == {"running": True, "errors": {}}
    assert engine.workers["other"] is other and other.stops == 0


@pytest.mark.parametrize("operation", ["disable", "delete"])
def test_disabled_or_deleted_camera_clears_previous_start_error(engine, fake_worker, operation):
    fake_worker.fail_phase = "start"
    with pytest.raises(RuntimeError):
        engine.update_camera("cam", camera())
    fake_worker.fail_phase = None

    if operation == "disable":
        engine.update_camera("cam", camera(enabled=False))
    else:
        engine.delete_camera("cam")

    assert not engine.camera_status("cam")["running"]
    assert engine.camera_status("cam")["errors"] == {}
    assert "cam" not in engine.workers


@pytest.mark.parametrize("operation", ["invalid_add", "invalid_update", "duplicate", "missing_update", "missing_delete"])
def test_rejected_mutation_does_not_write_or_change_workers(engine, monkeypatch, operation):
    persist = Mock(side_effect=AssertionError("must validate before persisting"))
    monkeypatch.setattr(engine, "_persist", persist)
    workers = dict(engine.workers)

    with pytest.raises((CameraError, CameraExists, KeyError)):
        if operation == "invalid_add":
            engine.add_camera(camera("new").model_copy(update={"sample_fps": 31}))
        elif operation == "invalid_update":
            engine.update_camera("cam", camera("wrong"))
        elif operation == "duplicate":
            engine.add_camera(camera())
        elif operation == "missing_update":
            engine.update_camera("missing", camera("missing"))
        else:
            engine.delete_camera("missing")

    persist.assert_not_called()
    assert engine.workers == workers
    assert all(worker.running and worker.stops == 0 for worker in workers.values())
