"""Журнал действий: что пишется (кто, роль, когда, что, адрес) и как читается через /api/audit."""

import time

import cv2
import numpy as np
import pytest

from guard.auth import passwords
from guard.auth.testing import auth_env
from guard.contracts import Event, Severity


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        yield e


def _items(env, **params):
    admin = env.login("admin")
    r = env.client.get("/api/audit", params=params, headers=admin)
    assert r.status_code == 200, r.text
    return r.json()


def _add_event(env, event_id="ev1"):
    env.engine.journal.add(Event(
        id=event_id, camera_id="cam1", site="Цех 1", zone_id="z1", rule_id="s1", type="zone_intrusion",
        severity=Severity.high, track_id=1, started_at=1000.0, confirmed_at=1002.0, bbox=(1, 2, 3, 4),
    ))


def test_login_logout_and_failures_are_recorded(env):
    headers = env.login("operator")
    env.client.post("/api/auth/login", json={"username": "operator", "password": "nope"})
    env.client.post("/api/auth/login", json={"username": "ghost", "password": "x"})
    env.client.post("/api/auth/logout", headers=headers)
    items = {(i["action"], i["username"]) for i in _items(env, action="auth.*")["items"]}
    assert ("auth.login", "operator") in items and ("auth.logout", "operator") in items
    assert ("auth.login_failed", "operator") in items and ("auth.login_failed", "ghost") in items
    entry = _items(env, action="auth.login", username="operator")["items"][0]
    assert entry["role"] == "operator" and entry["ip"] == "testclient" and entry["ts"] == pytest.approx(env.clock())
    assert "nope" not in str(_items(env)["items"])  # пароли в журнал не попадают


def test_event_update_records_status_change(env):
    _add_event(env)
    headers = env.login("operator")
    r = env.client.patch("/api/events/ev1", json={"status": "confirmed", "operator_note": "Остановили работы"},
                         headers=headers)
    assert r.status_code == 200
    entry = _items(env, action="event.update")["items"][0]
    assert entry["username"] == "operator" and entry["target"] == "ev1"
    assert entry["details"]["status"] == {"from": "new", "to": "confirmed"}
    assert entry["details"]["operator_note"] == "Остановили работы"


def test_user_actions_are_recorded_without_passwords(env):
    admin = env.login("admin")
    env.client.post("/api/users", json={"username": "ivanov", "display_name": "Иванов", "role": "worker",
                                        "password": "pass1234"}, headers=admin)
    env.client.patch("/api/users/ivanov", json={"role": "operator", "password": "pass5678"}, headers=admin)
    created = _items(env, action="user.create")["items"][0]
    assert created["target"] == "ivanov" and created["details"]["role"] == "worker"
    updated = _items(env, action="user.update")["items"][0]
    assert updated["details"]["role"] == {"from": "worker", "to": "operator"}
    assert updated["details"]["password_changed"] is True
    text = str(_items(env)["items"])
    assert "pass1234" not in text and "pass5678" not in text


def test_flywheel_review_and_jobs_are_recorded(env, monkeypatch):
    store = env.engine.flywheel_store
    store.write_frame("frames/cam1/s1.jpg", cv2.imencode(".jpg", np.full((24, 32, 3), 90, np.uint8))[1].tobytes())
    store.add({"id": "s1", "camera_id": "cam1", "source": "uncertain", "width": 32, "height": 24,
               "created_at": time.time(), "detections": []})
    safety, admin = env.login("safety"), env.login("admin")
    assert env.client.patch("/api/flywheel/samples/s1", json={"status": "accepted"}, headers=safety).status_code == 200

    class FakeJobs:
        def start(self, kind, args):
            return {"id": "job-1", "kind": kind}

        def to_dict(self, job, tail=False):
            return dict(job)

    env.app.state.flywheel_jobs = FakeJobs()
    assert env.client.post("/api/flywheel/jobs", json={"kind": "export", "args": {"limit": 5}},
                           headers=admin).status_code == 201
    review = _items(env, action="flywheel.review")["items"][0]
    assert review["username"] == "safety" and review["target"] == "s1" and review["details"]["status"] == "accepted"
    job = _items(env, action="flywheel.job")["items"][0]
    assert job["target"] == "job-1" and job["details"]["kind"] == "export" and job["details"]["args"] == {"limit": 5}


def test_audit_filters_and_pagination(env):
    for name in ("worker", "operator", "safety"):
        env.login(name)
        env.clock.advance(60)
    data = _items(env, action="auth.login", limit=2, offset=0)
    assert data["total"] == 4  # три входа и вход admin для чтения
    assert len(data["items"]) == 2
    assert [i["username"] for i in _items(env, action="auth.login", username="worker")["items"]] == ["worker"]
    start = env.clock() - 180
    window = _items(env, action="auth.login", **{"from": start + 30, "to": start + 90})
    assert [i["username"] for i in window["items"]] == ["operator"]
    iso = _items(env, action="auth.login", **{"from": "2000-01-01T00:00:00"})
    assert iso["total"] >= 4
    admin = env.login("admin")
    assert env.client.get("/api/audit", params={"from": "nonsense"}, headers=admin).status_code == 422
    assert env.client.get("/api/audit", params={"limit": 5000}, headers=admin).status_code == 422


def test_audit_visibility(env):
    assert env.client.get("/api/audit", headers=env.login("safety")).status_code == 200
    r = env.client.get("/api/audit", headers=env.login("operator"))
    assert r.status_code == 403 and r.json()["permission"] == "audit.view"


def test_audit_failure_does_not_break_the_action(env, monkeypatch):
    _add_event(env)
    headers = env.login("operator")

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(env.auth.store, "add_audit", boom)
    r = env.client.patch("/api/events/ev1", json={"status": "false"}, headers=headers)
    assert r.status_code == 200


def test_audit_with_auth_disabled_records_admin(tmp_path):
    with auth_env(tmp_path, auth=False) as e:
        e.engine.journal.add(Event(
            id="ev1", camera_id="cam1", site="Цех 1", rule_id="s1", type="zone_intrusion",
            severity=Severity.high, started_at=1000.0, confirmed_at=1002.0, bbox=(1, 2, 3, 4),
        ))
        assert e.client.patch("/api/events/ev1", json={"status": "confirmed"}).status_code == 200
        items = e.client.get("/api/audit", params={"action": "event.update"}).json()["items"]
        assert items[0]["username"] == "admin" and items[0]["role"] == "admin"
