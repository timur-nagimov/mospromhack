"""HTTP/WS API по docs/CONTRACTS.md: камеры (CRUD с сохранением в yaml), события, статистика, метрики, WS."""

import time
from datetime import datetime

import pytest
import yaml
from fastapi.testclient import TestClient

from guard.api.app import create_app
from guard.contracts import Event, Severity
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings, write_images
from guard.rules.geometry import LOCAL_TZ


@pytest.fixture()
def env(tmp_path):
    settings = make_settings(tmp_path)
    engine = PipelineEngine(settings, backend=FakeBackend())
    app = create_app(settings, engine=engine)
    with TestClient(app) as client:
        yield client, engine, settings, tmp_path


def camera_body(tmp_path, camera_id="cam1", **extra):
    folder = write_images(tmp_path / f"imgs_{camera_id}", n=3)
    body = {
        "camera_id": camera_id,
        "name": "Цех",
        "site": "Цех 1",
        "source": {"type": "images", "uri": str(folder)},
        "sample_fps": 5,
        "zones": [{"zone_id": "z1", "name": "Опасная", "kind": "danger", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]]}],
        "rules": [{"rule_id": "s1", "type": "zone_intrusion", "zones": ["z1"]}],
    }
    body.update(extra)
    return body


def add_event(engine, i=0, **kw):
    base = dict(
        id=f"ev{i}", camera_id="cam1", site="Цех 1", zone_id="z1", rule_id="s1", type="zone_intrusion",
        severity=Severity.high, track_id=i, started_at=1000.0 + i * 100, confirmed_at=1002.0 + i * 100, bbox=(1, 2, 3, 4),
    )
    base.update(kw)
    ev = Event(**base)
    engine.journal.add(ev)
    return ev


def wait_for(fn, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = fn()
        if r:
            return r
        time.sleep(0.05)
    return fn()


# ---------- служебное ----------


def test_health_and_rule_types(env):
    client, *_ = env
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["backend"] == "fake" and h["cameras_running"] == 0
    types = {r["type"]: r for r in client.get("/api/rules/types").json()}
    assert "zone_intrusion" in types and types["zone_intrusion"]["params_schema"]["type"] == "object"


def test_cors_for_vite_dev_server(env):
    client, *_ = env
    r = client.options(
        "/api/cameras", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"}
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


# ---------- камеры ----------


def test_camera_crud_persists_and_hot_restarts(env):
    client, engine, settings, tmp_path = env
    assert client.get("/api/cameras").json() == []
    r = client.post("/api/cameras", json=camera_body(tmp_path))
    assert r.status_code == 201, r.text
    assert r.json()["status"]["running"] is True
    saved = yaml.safe_load(settings.cameras_config.read_text(encoding="utf-8"))["cameras"]
    assert [c["camera_id"] for c in saved] == ["cam1"]
    assert client.post("/api/cameras", json=camera_body(tmp_path)).status_code == 409

    first_worker = engine.workers["cam1"]
    body = camera_body(tmp_path, name="Цех (обновлён)")
    body["status"] = {"running": False}  # только для чтения — игнорируется
    r = client.put("/api/cameras/cam1", json=body)
    assert r.status_code == 200 and r.json()["name"] == "Цех (обновлён)"
    assert engine.workers["cam1"] is not first_worker and not first_worker.running
    assert "Цех (обновлён)" in settings.cameras_config.read_text(encoding="utf-8")
    assert client.get("/api/cameras/cam1").json()["status"]["running"] is True

    assert client.delete("/api/cameras/cam1").status_code == 204
    assert client.get("/api/cameras/cam1").status_code == 404
    assert client.delete("/api/cameras/cam1").status_code == 404
    assert yaml.safe_load(settings.cameras_config.read_text(encoding="utf-8"))["cameras"] == []


@pytest.mark.parametrize(
    "patch, detail",
    [
        ({"rules": [{"rule_id": "x", "type": "no_such_rule"}]}, "unknown type"),
        ({"rules": [{"rule_id": "x", "type": "zone_intrusion", "zones": ["nope"]}]}, "unknown zones"),
        ({"sample_fps": 0}, "sample_fps"),
        ({"zones": [{"zone_id": "z1", "name": "z", "polygon": [[0, 0], [2, 0], [0, 1]]}], "rules": []}, "normalized"),
        ({"zones": [{"zone_id": "z1", "name": "z", "polygon": [[0, 0], [1, 0], [0, 1]], "active_hours": ["8-20"]}], "rules": []}, "active_hours"),
    ],
)
def test_camera_validation(env, patch, detail):
    client, _, _, tmp_path = env
    r = client.post("/api/cameras", json=camera_body(tmp_path, **patch))
    assert r.status_code == 422 and detail in r.text


def test_put_camera_id_mismatch_and_missing(env):
    client, _, _, tmp_path = env
    assert client.put("/api/cameras/other", json=camera_body(tmp_path)).status_code == 422
    assert client.put("/api/cameras/cam1", json=camera_body(tmp_path)).status_code == 404


def test_frame_jpg_and_mjpeg_stream(env):
    client, _, _, tmp_path = env
    client.post("/api/cameras", json=camera_body(tmp_path))
    r = wait_for(lambda: (resp := client.get("/api/cameras/cam1/frame.jpg")).status_code == 200 and resp)
    assert r and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"
    assert client.get("/api/cameras/nope/frame.jpg").status_code == 404
    with client.stream("GET", "/api/cameras/cam1/stream.mjpg?fps=10") as resp:
        assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
        buf = b""
        for chunk in resp.iter_bytes():
            buf += chunk
            if b"\xff\xd9" in buf:
                break
    assert b"--frame" in buf and b"Content-Type: image/jpeg" in buf and b"\xff\xd8" in buf


# ---------- события ----------


def test_events_list_filters_and_pagination(env):
    client, engine, *_ = env
    add_event(engine, 0)
    add_event(engine, 1, camera_id="cam2", severity=Severity.critical, type="fall_immobile")
    add_event(engine, 2, zone_id="z2")
    data = client.get("/api/events").json()
    assert data["total"] == 3 and [e["id"] for e in data["items"]] == ["ev2", "ev1", "ev0"]
    assert client.get("/api/events?camera_id=cam2").json()["total"] == 1
    assert client.get("/api/events?type=zone_intrusion,fall_immobile").json()["total"] == 3
    assert client.get("/api/events?severity=critical").json()["total"] == 1
    assert client.get("/api/events?zone_id=z2").json()["total"] == 1
    assert client.get("/api/events?from=1050&to=1150").json()["total"] == 1
    page = client.get("/api/events?limit=1&offset=1").json()
    assert page["total"] == 3 and [e["id"] for e in page["items"]] == ["ev1"]
    iso = datetime.fromtimestamp(1150, LOCAL_TZ).replace(tzinfo=None).isoformat()
    assert client.get(f"/api/events?from={iso}").json()["total"] == 1
    assert client.get("/api/events?from=yesterday").status_code == 422


def test_event_get_patch_and_ws_update(env):
    client, engine, *_ = env
    add_event(engine, 0)
    assert client.get("/api/events/ev0").json()["status"] == "new"
    assert client.get("/api/events/missing").status_code == 404
    with client.websocket_connect("/ws/events") as ws:
        r = client.patch("/api/events/ev0", json={"status": "confirmed", "operator_note": "остановили работы"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "confirmed" and body["operator_note"] == "остановили работы"
        msg = ws.receive_json()
        while msg["kind"] != "event_update":
            msg = ws.receive_json()
        assert msg["event"]["id"] == "ev0" and msg["event"]["status"] == "confirmed"
    assert client.patch("/api/events/ev0", json={"status": "bogus"}).status_code == 422
    assert client.patch("/api/events/missing", json={"status": "false"}).status_code == 404


def test_ws_receives_new_event_and_metrics(env, monkeypatch):
    client, engine, *_ = env
    import guard.api.app as app_mod

    with client.websocket_connect("/ws/events") as ws:
        ev = add_event(engine, 5)
        engine.bus.publish({"kind": "event", "event": ev.model_dump(mode="json")})
        msg = ws.receive_json()
        while msg["kind"] != "event":
            msg = ws.receive_json()
        assert msg["event"]["id"] == "ev5"
        deadline = time.time() + app_mod.METRICS_PERIOD_S + 3
        kinds = set()
        while "metrics" not in kinds and time.time() < deadline:
            m = ws.receive_json()
            kinds.add(m["kind"])
            if m["kind"] == "metrics":
                assert {"backend", "cameras", "inference"} <= set(m["data"])
        assert "metrics" in kinds


def test_snapshot_and_clip_files(env):
    client, engine, settings, _ = env
    ev = add_event(engine, 0)
    assert client.get("/api/events/ev0/snapshot.jpg").status_code == 404
    assert client.get("/api/events/ev0/clip.mp4").status_code == 404
    snap_rel = engine.media.rel_path(ev, ".jpg")
    clip_rel = engine.media.rel_path(ev, ".mp4")
    (settings.data_dir / snap_rel).parent.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / snap_rel).write_bytes(b"\xff\xd8jpeg\xff\xd9")
    (settings.data_dir / clip_rel).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 200)
    engine.journal.update("ev0", snapshot_path=snap_rel, clip_path=clip_rel)
    r = client.get("/api/events/ev0/snapshot.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    r = client.get("/api/events/ev0/clip.mp4", headers={"Range": "bytes=0-9"})
    assert r.status_code == 206 and r.headers["content-type"] == "video/mp4" and len(r.content) == 10


def test_stats_groupings(env):
    client, engine, *_ = env
    add_event(engine, 0)
    add_event(engine, 1, site="Склад", camera_id="cam2")
    add_event(engine, 2, type="ppe_missing")
    r = client.get("/api/stats?group_by=type").json()
    assert r["group_by"] == "type" and {i["key"]: i["count"] for i in r["items"]} == {"ppe_missing": 1, "zone_intrusion": 2}
    sites = {i["key"]: i["count"] for i in client.get("/api/stats?group_by=site").json()["items"]}
    assert sites == {"Склад": 1, "Цех 1": 2}
    wh = client.get("/api/stats?group_by=weekday_hour").json()["items"]
    local = datetime.fromtimestamp(1002.0, LOCAL_TZ)
    assert wh[0]["key"] == f"{local.weekday()}:{local.hour}" and sum(i["count"] for i in wh) == 3
    assert client.get("/api/stats?group_by=camera&from=1050").json()["items"] == [
        {"key": "cam1", "count": 1},
        {"key": "cam2", "count": 1},
    ]
    assert client.get("/api/stats?group_by=bogus").status_code == 422


def test_metrics_shape(env):
    client, _, _, tmp_path = env
    client.post("/api/cameras", json=camera_body(tmp_path))
    data = wait_for(lambda: (m := client.get("/api/metrics").json())["cameras"].get("cam1", {}).get("fps_processed") and m)
    assert data["backend"] == "fake"
    cam = data["cameras"]["cam1"]
    assert {"fps_in", "fps_processed", "lag_s", "timings_ms"} <= set(cam)
    assert {"total", "infer", "decode", "rules", "alert"} <= set(cam["timings_ms"])
    assert set(cam["timings_ms"]["total"]) == {"p50", "p95"}
    inf = data["inference"]
    assert {"backend", "device", "utilization", "models"} <= set(inf)
    assert "queue" in inf["models"]["det"]["timings_ms"]
