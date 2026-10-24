"""WS /ws/events: без входа — закрытие 4401, без права — 4403; метрики только с правом system.view."""

import time

import pytest
from starlette.websockets import WebSocketDisconnect

from guard.auth import deps, passwords, roles
from guard.auth.testing import auth_env


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        yield e


def _close_code(client, headers=None):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/events", headers=headers or {}) as ws:
            ws.receive_json()
    return exc.value.code


def _publish_until(ws, engine, message, kind, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        engine.bus.publish(message)
        got = ws.receive_json()
        if got["kind"] == kind:
            return got
    raise AssertionError(f"no {kind} message")


def test_without_session_closes_4401(env):
    assert _close_code(env.client) == 4401


def test_without_permission_closes_4403(env, monkeypatch):
    stripped = roles.Role("worker", "Сотрудник", "без журнала", frozenset({roles.ANALYTICS_VIEW}))
    monkeypatch.setitem(roles.ROLES, "worker", stripped)
    headers = env.login("worker")
    assert _close_code(env.client, headers) == 4403


def test_worker_gets_events_but_not_metrics(env):
    headers = env.login("worker")
    with env.client.websocket_connect("/ws/events", headers=headers) as ws:
        env.engine.bus.publish({"kind": "metrics", "data": {"secret": 1}})
        env.engine.bus.publish({"kind": "event", "event": {"id": "e1"}})
        assert ws.receive_json() == {"kind": "event", "event": {"id": "e1"}}


def test_operator_gets_metrics(env):
    headers = env.login("operator")
    with env.client.websocket_connect("/ws/events", headers=headers) as ws:
        got = _publish_until(ws, env.engine, {"kind": "metrics", "data": {"x": 1}}, "metrics")
        assert got["data"] == {"x": 1}


def test_logout_closes_open_socket(env, monkeypatch):
    monkeypatch.setattr(deps, "SESSION_RECHECK_S", 0.0)
    headers = env.login("operator")
    with env.client.websocket_connect("/ws/events", headers=headers) as ws:
        env.engine.bus.publish({"kind": "event", "event": {"id": "e1"}})
        assert ws.receive_json()["kind"] == "event"
        assert env.client.post("/api/auth/logout", headers=headers).status_code == 200
        env.engine.bus.publish({"kind": "event", "event": {"id": "e2"}})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 4401


def test_auth_disabled_socket_works_without_cookie(tmp_path):
    with auth_env(tmp_path, auth=False) as e:
        with e.client.websocket_connect("/ws/events") as ws:
            got = _publish_until(ws, e.engine, {"kind": "metrics", "data": {}}, "metrics")
            assert got["kind"] == "metrics"
