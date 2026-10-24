"""Матрица «роль × маршрут → 200, 401 или 403» по всем маршрутам API (docs/ROLES.md, «Права»).

Таблица ниже — контракт, записанный независимо от кода. Отдельные тесты проверяют, что она покрывает
каждый маршрут приложения и что любой /api/* без сессии отвечает 401.
"""

import pytest
from fastapi.routing import iter_route_contexts
from starlette.routing import compile_path
from starlette.websockets import WebSocketDisconnect

from guard.auth import passwords, roles
from guard.auth.deps import UNDECLARED, route_policy
from guard.auth.testing import auth_env, camera_dict
from guard.contracts import CameraConfig, Event, Severity
from guard.pipeline.testing import write_images

ROLES = ("worker", "operator", "safety", "admin")

ROUTES = [
    # метод, путь, тело, право (None — достаточно входа), коды ответа для роли с правом
    ("GET", "/api/stats", None, "analytics.view", {200}),
    ("GET", "/api/analytics", None, "analytics.view", {200}),
    ("GET", "/api/reports/shift", None, "analytics.view", {200}),
    ("GET", "/api/reports/presets", None, "analytics.view", {200}),
    ("GET", "/api/events", None, "events.view", {200}),
    ("GET", "/api/events/ev1", None, "events.view", {200}),
    ("PATCH", "/api/events/ev1", {"operator_note": "проверка прав"}, "events.handle", {200}),
    ("GET", "/api/events/ev1/snapshot.jpg", None, "video.view", {404}),
    ("GET", "/api/events/ev1/clip.mp4", None, "video.view", {404}),
    ("GET", "/api/cameras", None, "cameras.view", {200}),
    ("GET", "/api/cameras/cam1", None, "cameras.view", {200}),
    ("GET", "/api/rules/types", None, "cameras.view", {200}),
    ("GET", "/api/cameras/cam1/frame.jpg", None, "video.view", {503}),
    ("GET", "/api/cameras/cam1/stream.mjpg", None, "video.view", {503}),
    ("GET", "/api/metrics", None, "system.view", {200}),
    ("PUT", "/api/cameras/cam1", "camera", "zones.edit", {200}),
    ("POST", "/api/cameras", {}, "cameras.manage", {422}),
    ("DELETE", "/api/cameras/missing", None, "cameras.manage", {404}),
    ("POST", "/api/cameras/probe", {}, "cameras.manage", {422}),  # маршрут F1: проверяется, когда он есть
    ("GET", "/api/flywheel/summary", None, "flywheel.review", {200}),
    ("GET", "/api/flywheel/samples", None, "flywheel.review", {200}),
    ("GET", "/api/flywheel/samples/missing", None, "flywheel.review", {404}),
    ("GET", "/api/flywheel/samples/missing/frame.jpg", None, "flywheel.review", {404}),
    ("PATCH", "/api/flywheel/samples/missing", {"status": "accepted"}, "flywheel.review", {404}),
    ("POST", "/api/flywheel/capture/missing", None, "flywheel.review", {404}),
    ("GET", "/api/flywheel/datasets", None, "flywheel.review", {200}),
    ("GET", "/api/flywheel/models", None, "flywheel.review", {200}),
    ("GET", "/api/flywheel/jobs", None, "flywheel.review", {200}),
    ("GET", "/api/flywheel/jobs/missing", None, "flywheel.review", {404}),
    ("GET", "/api/flywheel/jobs/missing/log", None, "flywheel.review", {404}),
    ("POST", "/api/flywheel/datasets", {"include_holdout": "maybe"}, "flywheel.train", {422}),
    ("POST", "/api/flywheel/jobs", {}, "flywheel.train", {422}),
    ("GET", "/api/audit", None, "audit.view", {200}),
    ("GET", "/api/users", None, "users.manage", {200}),
    ("POST", "/api/users", {}, "users.manage", {422}),
    ("PATCH", "/api/users/missing", {"display_name": "x"}, "users.manage", {404}),
    ("GET", "/api/employees", None, "employees.view", {200}),
    ("POST", "/api/employees", {}, "employees.manage", {422}),
    ("GET", "/api/employees/missing/report", None, "employees.view", {200, 404}),
    ("GET", "/api/events/missing/employees", None, "employees.view", {200, 404}),
    ("PUT", "/api/events/missing/employees", {"employee_ids": []}, "events.assign", {200, 404}),
    ("GET", "/api/auth/me", None, None, {200}),
    ("GET", "/api/auth/roles", None, None, {200}),
]
PUBLIC = {("GET", "/health"), ("GET", "/api/auth/config"), ("POST", "/api/auth/login")}
LOGIN_ONLY = {("POST", "/api/auth/logout")}  # проверяется отдельно: выход закрывает сессию


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("matrix")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(passwords, "SCRYPT_N", 2**10)
        with auth_env(tmp) as e:
            folder = write_images(tmp / "imgs", n=2)
            e.camera = camera_dict(folder)
            e.engine.add_camera(CameraConfig(**e.camera))
            e.engine.journal.add(Event(
                id="ev1", camera_id="cam1", site="Цех 1", zone_id="z1", rule_id="s1", type="zone_intrusion",
                severity=Severity.high, track_id=1, started_at=1000.0, confirmed_at=1002.0, bbox=(1, 2, 3, 4),
            ))
            e.headers = {role: e.login(role) for role in ROLES}
            yield e


def _app_routes(app):
    """(вид, метод, шаблон пути, объект маршрута) для каждого маршрута API приложения."""
    out = []
    for rc in iter_route_contexts(app.routes):
        path = rc.path_format
        if not path or not path.startswith(("/api", "/ws", "/health")):
            continue
        if rc.methods is None:
            out.append(("ws", "WS", path, rc.original_route))
        else:
            for method in sorted(rc.methods - {"HEAD"}):
                out.append(("http", method, path, rc.original_route))
    return out


def _exists(app, method, path):
    for kind, m, template, _ in _app_routes(app):
        if kind == "http" and m == method and compile_path(template)[0].match(path):
            return True
    return False


def _request(env, method, path, body, headers):
    if body == "camera":
        body = env.camera
    return env.client.request(method, path, json=body, headers=headers)


@pytest.mark.parametrize("role", (*ROLES, "anonymous"))
@pytest.mark.parametrize(
    "method,path,body,perm,ok", ROUTES, ids=[f"{m} {p}" for m, p, *_ in ROUTES],
)
def test_role_route_matrix(env, role, method, path, body, perm, ok):
    if not _exists(env.app, method, path):
        pytest.skip(f"{method} {path}: маршрута нет в приложении")
    headers = env.headers.get(role, {})
    r = _request(env, method, path, body, headers)
    if role == "anonymous":
        assert r.status_code == 401, r.text
        assert r.json() == {"detail": "auth required"}
    elif perm is None or perm in roles.permissions_for(role):
        assert r.status_code in ok, f"{role}: {r.status_code} {r.text[:200]}"
    else:
        assert r.status_code == 403, f"{role}: {r.status_code} {r.text[:200]}"
        assert r.json() == {"detail": "forbidden", "permission": perm}


def test_matrix_covers_every_route(env):
    """Новый маршрут без строки в таблице ROUTES — сигнал дописать его в матрицу и в docs/ROLES.md."""
    missing = []
    for kind, method, template, _ in _app_routes(env.app):
        if kind == "ws" or (method, template) in PUBLIC | LOGIN_ONLY:
            continue
        regex = compile_path(template)[0]
        if not any(m == method and regex.match(p) for m, p, *_ in ROUTES):
            missing.append(f"{method} {template}")
    assert not missing, missing


def test_every_route_declares_access(env):
    """Каждый маршрут объявляет доступ: открыт, только вход или конкретные права."""
    undeclared = [
        f"{method} {template}"
        for _, method, template, route in _app_routes(env.app)
        if route_policy(route, method) is UNDECLARED
    ]
    assert not undeclared, undeclared


def test_every_api_route_without_session_is_401(env):
    for kind, method, template, _ in _app_routes(env.app):
        path = template.replace("{", "").replace("}", "")  # {camera_id} → camera_id
        if kind == "ws":
            with pytest.raises(WebSocketDisconnect) as exc:
                with env.client.websocket_connect(path) as ws:
                    ws.receive_json()
            assert exc.value.code == 4401, template
            continue
        r = env.client.request(method, path)
        if (method, template) in PUBLIC:
            assert r.status_code != 401, f"{method} {template}"
        else:
            assert r.status_code == 401, f"{method} {template}: {r.status_code}"


def _late_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/api/forgotten")
    def forgotten():
        return {"ok": True}

    @router.post("/api/cameras/probe")
    def probe():  # как маршрут F1, если он придёт без явного права
        return {"ok": True}

    return router


def test_new_router_without_declarations_is_closed_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)
    with auth_env(tmp_path) as e:
        e.app.include_router(_late_router())
        assert e.client.get("/api/forgotten").status_code == 401
        safety, admin = e.login("safety"), e.login("admin")
        r = e.client.get("/api/forgotten", headers=safety)
        assert r.status_code == 403 and r.json() == {"detail": "forbidden", "permission": "admin"}
        assert e.client.get("/api/forgotten", headers=admin).status_code == 200
        # проверка потока закрыта правом cameras.manage даже без объявления в модуле
        r = e.client.post("/api/cameras/probe", headers=safety)
        assert r.status_code == 403 and r.json()["permission"] == "cameras.manage"
        # настоящий маршрут F1 зарегистрирован раньше заглушки: без тела он отвечает 422, но право уже пройдено
        assert e.client.post("/api/cameras/probe", headers=admin).status_code in (200, 422)


class IdleConsumer:
    """Потребитель очереди шлюза, который ничего не получает."""

    def status(self):
        return {"connected": False}

    def run(self, receive, stop):
        stop.wait()

    def close(self):
        pass


def test_gateway_mode_uses_the_same_rights(tmp_path, monkeypatch):
    from guard.gateway.engine import GatewayEngine
    from guard.pipeline.testing import make_settings

    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)
    settings = make_settings(tmp_path, mode="gateway", auth=True, autostart=False)
    with auth_env(tmp_path, engine=GatewayEngine(settings, consumer=IdleConsumer())) as e:
        assert e.client.get("/api/stats").status_code == 401
        worker, operator, admin = e.login("worker"), e.login("operator"), e.login("admin")
        assert e.client.get("/api/stats", headers=worker).status_code == 200
        assert e.client.get("/api/metrics", headers=worker).status_code == 403
        assert e.client.get("/api/metrics", headers=operator).status_code == 200
        body = camera_dict(tmp_path, camera_id="cam9")
        assert e.client.post("/api/cameras", json=body, headers=operator).status_code == 403
        assert e.client.post("/api/cameras", json=body, headers=admin).status_code == 409  # камеры настраивает edge


def test_public_routes(env):
    assert env.client.get("/health").status_code == 200
    assert env.client.get("/api/auth/config").status_code == 200
    r = env.client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid credentials"}


@pytest.mark.parametrize("role", ROLES)
def test_logout_for_every_role(env, role):
    headers = env.login(role)
    assert env.client.post("/api/auth/logout", headers=headers).status_code == 200
    assert env.client.get("/api/auth/me", headers=headers).status_code == 401


def test_curl_scenario_from_brief(env):
    """worker: статистика 200, кадр 403, PATCH события 403; без cookie — 401."""
    worker = env.headers["worker"]
    assert env.client.get("/api/stats", headers=worker).status_code == 200
    assert env.client.get("/api/cameras/cam1/frame.jpg", headers=worker).status_code == 403
    assert env.client.patch("/api/events/ev1", json={"status": "confirmed"}, headers=worker).status_code == 403
    assert env.client.get("/api/stats").status_code == 401
