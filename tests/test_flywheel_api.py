"""API маховика через TestClient: сводка, сэмплы и фильтры, кадр, проверка человеком, захват, датасеты, модели, задачи."""

import json
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import CameraConfig, RuleConfig, SourceConfig
from guard.flywheel import cards as fw_cards
from guard.flywheel.collector import SampleJob
from guard.flywheel.jobs import JobManager
from guard.pipeline.config import save_cameras
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings, write_images


def _jpeg(w=320, h=240):
    return cv2.imencode(".jpg", np.full((h, w, 3), 90, np.uint8))[1].tobytes()


def _add(store, sid, **kw):
    sample = {"id": sid, "camera_id": "cam1", "source": "uncertain", "width": 320, "height": 240,
              "created_at": kw.pop("created_at", time.time()),
              "detections": [{"model_id": "det", "cls": "person", "conf": 0.45, "bbox": [10, 10, 60, 160]}]}
    sample.update(kw)
    store.write_frame(f"frames/{sample['camera_id']}/{sid}.jpg", _jpeg())
    return store.add(sample)


@pytest.fixture()
def env(tmp_path):
    settings = make_settings(tmp_path)
    engine = PipelineEngine(settings, backend=FakeBackend())
    app = create_app(settings, engine=engine)
    with TestClient(app) as client:
        yield client, engine, tmp_path


def test_summary_fields(env):
    client, engine, _ = env
    _add(engine.flywheel_store, "u1")
    _add(engine.flywheel_store, "e1", source="event", event_id="ev1", holdout=1)
    data = client.get("/api/flywheel/summary").json()
    for key in ("total", "today", "by_source", "by_status", "by_class", "reviewed", "holdout", "dropped",
                "last_dataset", "models", "collecting"):
        assert key in data
    assert data["total"] == 2 and data["today"] == 2 and data["holdout"] == 1 and data["reviewed"] == 0
    assert data["by_source"]["event"] == 1 and data["by_source"]["uncertain"] == 1
    assert data["dropped"] == 0 and data["collecting"] is True
    assert data["models"] == {"deployed": [], "candidate": []} or isinstance(data["models"]["deployed"], list)


def test_samples_list_filters_get_and_frame(env):
    client, engine, _ = env
    store = engine.flywheel_store
    _add(store, "u1", created_at=100.0)
    _add(store, "e1", source="event", event_id="ev1", created_at=200.0)
    _add(store, "h1", holdout=1, created_at=300.0)
    body = client.get("/api/flywheel/samples").json()
    assert body["total"] == 3 and [s["id"] for s in body["items"]] == ["h1", "e1", "u1"]
    assert body["items"][0]["frame_url"] == "/api/flywheel/samples/h1/frame.jpg"
    assert [s["id"] for s in client.get("/api/flywheel/samples", params={"event_id": "ev1"}).json()["items"]] == ["e1"]
    assert client.get("/api/flywheel/samples", params={"holdout": "0"}).json()["total"] == 2
    assert client.get("/api/flywheel/samples", params={"source": "event,uncertain", "order": "created_at"}).json()[
        "items"][0]["id"] == "u1"
    assert client.get("/api/flywheel/samples", params={"cls": "person", "limit": 1}).json()["total"] == 3
    assert client.get("/api/flywheel/samples", params={"order": "bad"}).status_code == 422
    assert client.get("/api/flywheel/samples", params={"holdout": "maybe"}).status_code == 422
    one = client.get("/api/flywheel/samples/e1").json()
    assert one["event_id"] == "ev1" and one["detections"][0]["cls"] == "person"
    assert client.get("/api/flywheel/samples/nope").status_code == 404
    frame = client.get("/api/flywheel/samples/e1/frame.jpg")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg" and frame.content[:2] == b"\xff\xd8"


def test_patch_review_status_and_labels(env):
    client, engine, _ = env
    store = engine.flywheel_store
    _add(store, "s1")
    store.update("s1", labels=[{"cls": "person", "bbox": [10, 10, 60, 160], "origin": "teacher", "conf": 0.9}],
                 status="auto")
    r = client.patch("/api/flywheel/samples/s1", json={
        "status": "accepted",
        "labels": [{"cls": "person", "bbox": [10, 10, 60, 160]}, {"cls": "helmet", "bbox": [20, 10, 40, 30]}],
    })
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "accepted" and out["reviewed_at"] is not None
    assert [(lb["cls"], lb["origin"]) for lb in out["labels"]] == [("person", "teacher"), ("helmet", "operator")]
    assert client.patch("/api/flywheel/samples/s1", json={"status": "auto"}).status_code == 422
    assert client.patch("/api/flywheel/samples/s1", json={"labels": [{"cls": "x", "bbox": [5, 5, 1, 1]}]}).status_code == 422
    assert client.patch("/api/flywheel/samples/none", json={"status": "rejected"}).status_code == 404
    rejected = client.patch("/api/flywheel/samples/s1", json={"status": "rejected"}).json()
    assert rejected["status"] == "rejected" and len(rejected["labels"]) == 2  # метки без изменений


def test_capture_current_frame_of_running_camera(tmp_path):
    settings = make_settings(tmp_path)
    folder = write_images(tmp_path / "imgs", n=3)
    cam = CameraConfig(camera_id="cam1", name="Цех", site="Цех 1", source=SourceConfig(type="images", uri=str(folder)),
                       sample_fps=5, rules=[RuleConfig(rule_id="z", type="zone_intrusion")])
    save_cameras(settings.cameras_config, [cam])
    engine = PipelineEngine(settings, backend=FakeBackend())
    app = create_app(settings, engine=engine)
    with TestClient(app) as client:
        deadline = time.time() + 5
        while time.time() < deadline and engine.metrics.snapshot().get("cam1", {}).get("frames_processed", 0) < 1:
            time.sleep(0.05)
        r = client.post("/api/flywheel/capture/cam1")
        assert r.status_code == 201, r.text
        sample = r.json()
        assert sample["source"] == "operator" and sample["camera_id"] == "cam1" and (sample["width"], sample["height"]) == (320, 240)
        assert client.get(sample["frame_url"]).status_code == 200
        assert client.post("/api/flywheel/capture/nope").status_code == 404


def test_datasets_create_and_list(env):
    client, engine, _ = env
    store = engine.flywheel_store
    assert client.post("/api/flywheel/datasets", json={}).status_code == 422  # принятых нет
    _add(store, "a")
    store.review("a", status="accepted", labels=[{"cls": "person", "bbox": [10, 10, 60, 160]}])
    r = client.post("/api/flywheel/datasets", json={"include_holdout": False, "classes": ["person"]})
    assert r.status_code == 201, r.text
    ds = r.json()
    assert ds["classes"] == ["person"] and ds["n_train"] + ds["n_val"] == 1
    listed = client.get("/api/flywheel/datasets").json()
    assert [d["version"] for d in listed] == [ds["version"]]
    assert client.get("/api/flywheel/summary").json()["last_dataset"]["version"] == ds["version"]


def test_models_from_cards(env, tmp_path, monkeypatch):
    client, *_ = env
    cards = tmp_path / "cards"
    cards.mkdir()
    (cards / "forklift-v1.json").write_text(json.dumps({"model_id": "forklift", "version": "forklift-v1",
                                                         "created_at": "2026-10-24T15:30:00+03:00", "status": "candidate",
                                                         "gate": {"promote": True}}), encoding="utf-8")
    (cards / "det-v2.json").write_text(json.dumps({"model_id": "det", "version": "det-v2",
                                                    "created_at": "2026-10-24T18:00:00+03:00", "status": "deployed"}),
                                       encoding="utf-8")
    (cards / "broken.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(fw_cards, "CARDS_DIR", cards)
    models = client.get("/api/flywheel/models").json()
    assert [m["version"] for m in models] == ["det-v2", "forklift-v1"]  # новые сверху, битая пропущена
    summary = client.get("/api/flywheel/summary").json()["models"]
    assert [m["version"] for m in summary["deployed"]] == ["det-v2"]
    assert summary["candidate"][0]["promote"] is True


def test_jobs_permissions_missing_module_and_run(env, tmp_path):
    client, engine, _ = env
    r = client.post("/api/flywheel/jobs", json={"kind": "train", "args": {}})
    assert r.status_code == 403 and "GUARD_FLYWHEEL_ALLOW_TRAIN" in r.json()["detail"]
    assert client.post("/api/flywheel/jobs", json={"kind": "hack", "args": {}}).status_code == 422

    code = tmp_path / "code"
    (code / "ml" / "flywheel").mkdir(parents=True)
    (code / "ml" / "flywheel" / "__init__.py").write_text("", encoding="utf-8")
    (code / "ml" / "flywheel" / "export.py").write_text(
        "import os, sys\nprint('args', sys.argv[1:], os.environ['GUARD_FLYWHEEL_DIR'])\n", encoding="utf-8")
    manager = JobManager(engine.flywheel_store.root, allow_train=True, python=sys.executable, code_root=code)
    client.app.state.flywheel_jobs = manager

    missing = client.post("/api/flywheel/jobs", json={"kind": "gate", "args": {}}).json()
    assert missing["status"] == "failed" and "ml/flywheel/gate.py ещё нет" in missing["error"]
    job = client.post("/api/flywheel/jobs", json={"kind": "export", "args": {"include_holdout": True, "classes": ["a", "b"],
                                                                             "limit": 5}}).json()
    assert job["status"] == "running" and job["log_url"] == f"/api/flywheel/jobs/{job['id']}/log"
    deadline = time.time() + 10
    while time.time() < deadline and client.get(f"/api/flywheel/jobs/{job['id']}").json()["status"] == "running":
        time.sleep(0.05)
    done = client.get(f"/api/flywheel/jobs/{job['id']}").json()
    assert done["status"] == "done" and done["returncode"] == 0
    assert "--include-holdout" in done["log_tail"] and "a,b" in done["log_tail"]
    assert str(engine.flywheel_store.root) in client.get(done["log_url"]).text
    assert {j["id"] for j in client.get("/api/flywheel/jobs").json()} == {job["id"], missing["id"]}
    assert client.get("/api/flywheel/jobs/zzz").status_code == 404
    # описание задачи переживает перезапуск сервиса
    again = JobManager(engine.flywheel_store.root, code_root=code)
    assert again.get(job["id"]).status == "done"


def test_ws_message_does_not_break_existing_clients(env):
    client, engine, _ = env
    with client.websocket_connect("/ws/events") as ws:
        engine.flywheel_writer.write(SampleJob(
            sample={"camera_id": "cam1", "source": "operator", "created_at": time.time()},
            frame=np.zeros((48, 64, 3), np.uint8), detections={},
        ))
        msg = ws.receive_json()
        while msg.get("kind") != "flywheel_sample":
            msg = ws.receive_json()
        assert msg["type"] == "flywheel_sample" and msg["sample"]["source"] == "operator"
        assert set(msg["sample"]) >= {"id", "camera_id", "source", "status", "frame_url", "holdout"}
