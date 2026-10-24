"""Проверка безопасности входа и прав глазами атакующего (docs/ROLES.md, «Безопасность: что проверено»).

Каждый тест воспроизводит попытку обхода через TestClient: другой метод (HEAD, OPTIONS), другой вид пути,
чужой источник запроса, повтор токена после выхода, подбор пароля, утечка хэшей и паролей, подмена роли
заголовком, опасный адрес в проверке потока.
"""

from __future__ import annotations

import json
import logging

import pytest
from starlette.websockets import WebSocketDisconnect

from guard.auth import deps, passwords
from guard.auth.testing import SESSION_COOKIE, auth_env, camera_dict
from guard.contracts import CameraConfig, Event, Severity
from guard.pipeline.testing import write_images

EVIL = "http://evil.example"
GET_ROUTES = [
    "/api/stats", "/api/analytics", "/api/reports/presets", "/api/events", "/api/events/ev1", "/api/cameras",
    "/api/cameras/cam1", "/api/rules/types", "/api/metrics", "/api/audit", "/api/users", "/api/employees",
    "/api/flywheel/summary", "/api/flywheel/jobs", "/api/auth/me", "/api/auth/roles",
]
VIDEO_ROUTES = [
    "/api/cameras/cam1/frame.jpg", "/api/cameras/cam1/stream.mjpg", "/api/events/ev1/snapshot.jpg",
    "/api/events/ev1/clip.mp4",
]


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


def _seed(e, tmp_path):
    folder = write_images(tmp_path / "imgs", n=2)
    e.engine.add_camera(CameraConfig(**camera_dict(folder)))
    e.engine.journal.add(Event(
        id="ev1", camera_id="cam1", site="Цех 1", zone_id="z1", rule_id="s1", type="zone_intrusion",
        severity=Severity.high, started_at=1000.0, confirmed_at=1001.0, bbox=(1, 2, 3, 4),
    ))
    return folder


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        e.folder = _seed(e, tmp_path)
        yield e


@pytest.fixture()
def spa_env(tmp_path):
    """Со сборкой интерфейса: неизвестные пути уходят в статику (index.html)."""
    tmp_path = tmp_path / "spa"
    dist = tmp_path / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html>SPA</html>", encoding="utf-8")
    with auth_env(tmp_path, frontend_dist=dist) as e:
        _seed(e, tmp_path)
        yield e


# ---------------------------------------------------------------- обход прав: методы и пути


@pytest.mark.parametrize("path", GET_ROUTES + VIDEO_ROUTES)
def test_head_gives_no_data(env, path):
    """HEAD у маршрутов FastAPI не объявлен: 405 без тела — и без входа, и с входом без права."""
    r = env.client.head(path)
    assert r.status_code in (401, 405), (path, r.status_code)
    assert not r.content
    worker = env.login("worker")
    r = env.client.head(path, headers=worker)
    assert r.status_code in (403, 405) and not r.content


@pytest.mark.parametrize("path", VIDEO_ROUTES)
def test_video_is_403_for_worker(env, path):
    worker = env.login("worker")
    assert env.client.get(path, headers=worker).status_code == 403


@pytest.mark.parametrize("path", ["/api/users", "/api/cameras", "/api/events/ev1"])
def test_options_gives_no_data(env, path):
    admin = env.login("admin")
    r = env.client.options(path, headers=admin)  # не preflight: маршрута OPTIONS нет
    assert r.status_code == 405 and "admin" not in r.text
    r = env.client.options(path, headers={"Origin": EVIL, "Access-Control-Request-Method": "GET"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers


@pytest.mark.parametrize("path", [
    "/api/users/", "/api%2Fusers", "/api/users%2F", "/API/users", "//api/users", "/api/./users",
    "/api/cameras/cam1/../../users", "/api/users;x", "/api/users?x=/health", "/api/users%00",
])
def test_path_variants_do_not_leak(env, spa_env, path):
    """Другая запись пути не обходит проверку: права висят на маршруте, а не на префиксе пути."""
    for e in (env, spa_env):
        for headers in ({}, e.login("worker")):
            r = e.client.get(path, headers=headers, follow_redirects=False)
            assert r.status_code in (401, 403, 404, 200, 307), (path, r.status_code)
            assert "display_name" not in r.text, (path, r.text[:200])
            if r.status_code == 200:
                assert r.text == "<html>SPA</html>", path  # только страница интерфейса


def test_trailing_slash_only_redirects(env):
    """Слэш на конце: редирект на канонический путь без данных, дальше — обычная проверка права."""
    worker = env.login("worker")
    for headers in ({}, worker):
        r = env.client.get("/api/users/", headers=headers, follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"].endswith("/api/users") and not r.content
    assert env.client.get("/api/users", headers=worker).status_code == 403


def test_spa_fallback_does_not_serve_api_paths(spa_env):
    assert spa_env.client.get("/api/unknown").status_code == 404
    assert spa_env.client.get("/ws/unknown").status_code == 404
    assert spa_env.client.get("/login").text == "<html>SPA</html>"


def test_mjpeg_stream_needs_video_view(env):
    assert env.client.get("/api/cameras/cam1/stream.mjpg?mode=live").status_code == 401
    worker = env.login("worker")
    assert env.client.get("/api/cameras/cam1/stream.mjpg?mode=live", headers=worker).status_code == 403


def test_ws_needs_session_and_rejects_garbage_cookie(env):
    for headers in ({}, {"Cookie": f"{SESSION_COOKIE}=forged"}):
        with pytest.raises(WebSocketDisconnect) as exc:
            with env.client.websocket_connect("/ws/events", headers=headers) as ws:
                ws.receive_json()
        assert exc.value.code == 4401


# ---------------------------------------------------------------- сессии


def _token(headers):
    return headers["Cookie"].split("=", 1)[1]


def test_session_fixation_attacker_token_is_ignored(env):
    planted = "attacker-chosen-token"
    r = env.client.post(
        "/api/auth/login", json={"username": "operator", "password": "operator"},
        headers={"Cookie": f"{SESSION_COOKIE}={planted}"},
    )
    assert r.status_code == 200
    assert r.cookies.get(SESSION_COOKIE) != planted
    env.client.cookies.clear()
    assert env.client.get("/api/auth/me", headers={"Cookie": f"{SESSION_COOKIE}={planted}"}).status_code == 401


def test_relogin_closes_previous_session_of_the_browser(env):
    """«Сменить роль» и повторный вход в том же браузере закрывают прежнюю сессию, а не оставляют её на 12 ч."""
    old = env.login("admin")
    r = env.client.post("/api/auth/login", json={"username": "worker", "password": "worker"}, headers=old)
    assert r.status_code == 200
    env.client.cookies.clear()
    assert env.client.get("/api/auth/me", headers=old).status_code == 401


def test_token_replay_after_logout_disable_and_password_change(env):
    admin = env.login("admin")
    # выход
    op = env.login("operator")
    assert env.client.post("/api/auth/logout", headers=op).status_code == 200
    assert env.client.get("/api/auth/me", headers=op).status_code == 401
    assert env.client.post("/api/auth/logout", headers=op).status_code == 401
    # смена пароля администратором закрывает все сессии пользователя
    op1, op2 = env.login("operator"), env.login("operator")
    r = env.client.patch("/api/users/operator", json={"password": "new-pass-1"}, headers=admin)
    assert r.status_code == 200
    for h in (op1, op2):
        assert env.client.get("/api/auth/me", headers=h).status_code == 401
    # смена своего пароля оставляет текущую сессию, остальные закрывает
    a2 = env.login("admin")
    assert env.client.patch("/api/users/admin", json={"password": "admin-2"}, headers=admin).status_code == 200
    assert env.client.get("/api/auth/me", headers=admin).status_code == 200
    assert env.client.get("/api/auth/me", headers=a2).status_code == 401
    # отключение
    s = env.login("safety")
    assert env.client.patch("/api/users/safety", json={"active": False}, headers=admin).status_code == 200
    assert env.client.get("/api/auth/me", headers=s).status_code == 401
    r = env.client.post("/api/auth/login", json={"username": "safety", "password": "safety"})
    assert r.status_code == 401


def test_session_expires_and_token_is_not_logged_or_returned(env, caplog):
    caplog.set_level(logging.DEBUG)
    r = env.client.post("/api/auth/login", json={"username": "worker", "password": "worker"})
    token = r.cookies.get(SESSION_COOKIE)
    env.client.cookies.clear()
    assert token and token not in r.text and token not in caplog.text
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    env.clock.advance(12 * 3600 + 1)
    assert env.client.get("/api/auth/me", headers={"Cookie": f"{SESSION_COOKIE}={token}"}).status_code == 401


def test_role_change_ends_long_connections(env, monkeypatch):
    """Понижение роли обрывает уже открытые MJPEG и WebSocket, как выход."""
    monkeypatch.setattr(deps, "SESSION_RECHECK_S", 0.0)
    admin, op = env.login("admin"), env.login("operator")
    user = env.auth.resolve(_token(op))
    assert env.auth.session_alive(user)
    assert env.client.patch("/api/users/operator", json={"role": "worker"}, headers=admin).status_code == 200
    assert not env.auth.session_alive(user)
    # новая проверка по той же сессии — уже с правами worker
    assert env.client.get("/api/cameras/cam1/frame.jpg", headers=op).status_code == 403


# ---------------------------------------------------------------- CORS и CSRF


def test_cors_reflects_only_listed_origins_with_credentials(tmp_path):
    with auth_env(tmp_path, cors_origins=["http://localhost:5173"]) as e:
        r = e.client.get("/health", headers={"Origin": EVIL})
        assert "access-control-allow-origin" not in r.headers
        r = e.client.get("/health", headers={"Origin": "http://localhost:5173"})
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert r.headers["access-control-allow-credentials"] == "true"
        r = e.client.options("/api/users", headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"})
        assert r.status_code == 400 and "access-control-allow-origin" not in r.headers


def test_cors_star_never_allows_credentials(tmp_path):
    with auth_env(tmp_path / "a", cors_origins=["*"]) as e:
        r = e.client.get("/health", headers={"Origin": EVIL, "Cookie": f"{SESSION_COOKIE}=x"})
        assert "access-control-allow-credentials" not in r.headers


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data; boundary=x"])
def test_simple_cross_site_bodies_are_not_parsed_as_json(env, content_type):
    """Форма или fetch(no-cors) чужого сайта шлёт только «простые» типы — JSON-маршруты их не принимают."""
    admin = env.login("admin")
    body = json.dumps({"username": "csrf", "display_name": "x", "role": "admin", "password": "pwned1"})
    r = env.client.post("/api/users", content=body, headers={**admin, "Content-Type": content_type})
    assert r.status_code == 422
    r = env.client.patch("/api/events/ev1", content=json.dumps({"status": "false_positive"}),
                         headers={**admin, "Content-Type": content_type})
    assert r.status_code == 422
    assert env.auth.store.get_user("csrf") is None
    assert env.engine.journal.get("ev1").status.value == "new"


@pytest.mark.parametrize("method,path", [
    ("POST", "/api/auth/logout"), ("POST", "/api/flywheel/capture/cam1"), ("DELETE", "/api/cameras/cam1"),
    ("POST", "/api/users"), ("PATCH", "/api/events/ev1"),
])
def test_foreign_origin_cannot_change_state(env, method, path):
    """Тот же сайт на другом порту (SameSite=Lax cookie отправит) или сайт без входа (GUARD_AUTH=0):
    изменяющий запрос с чужим Origin отклоняется до обработчика."""
    admin = env.login("admin")
    r = env.client.request(method, path, headers={**admin, "Origin": "http://localhost:3000"})
    assert r.status_code == 403 and r.json()["detail"] == "cross-origin request blocked"
    r = env.client.request(method, path, headers={**admin, "Origin": "null"})
    assert r.status_code == 403
    assert env.client.get("/api/auth/me", headers=admin).status_code == 200
    assert env.engine.get_camera("cam1") is not None


def test_same_origin_and_listed_origins_pass(env):
    admin = env.login("admin")
    for origin in ("http://testserver", "http://localhost:5173", "http://127.0.0.1:5173"):
        r = env.client.patch("/api/events/ev1", json={"operator_note": "ok"}, headers={**admin, "Origin": origin})
        assert r.status_code == 200, origin
    # без Origin (curl, скрипты) — как раньше
    assert env.client.patch("/api/events/ev1", json={"operator_note": "x"}, headers=admin).status_code == 200
    # чтение с чужим Origin не блокируется сервером: ответ не прочитает браузер (нет CORS-заголовков)
    r = env.client.get("/api/events", headers={**admin, "Origin": EVIL})
    assert r.status_code == 200 and "access-control-allow-origin" not in r.headers


def test_forwarded_host_counts_as_same_origin(env):
    admin = env.login("admin")
    r = env.client.patch("/api/events/ev1", json={"operator_note": "proxy"},
                         headers={**admin, "Origin": "https://demo.example", "X-Forwarded-Host": "demo.example"})
    assert r.status_code == 200


def test_foreign_origin_websocket_is_closed(env):
    op = env.login("operator")
    with pytest.raises(WebSocketDisconnect) as exc:
        with env.client.websocket_connect("/ws/events", headers={**op, "Origin": EVIL}) as ws:
            ws.receive_json()
    assert exc.value.code == 4403
    with env.client.websocket_connect("/ws/events", headers={**op, "Origin": "http://testserver"}) as ws:
        env.engine.bus.publish({"kind": "event", "event": {"id": "e1"}})
        assert ws.receive_json()["kind"] == "event"


def test_auth_off_still_blocks_foreign_origin(tmp_path):
    with auth_env(tmp_path, auth=False) as e:
        _seed(e, tmp_path)
        r = e.client.delete("/api/cameras/cam1", headers={"Origin": EVIL})
        assert r.status_code == 403
        assert e.engine.get_camera("cam1") is not None


# ---------------------------------------------------------------- подбор пароля


def test_bruteforce_gets_429_even_with_right_password(env):
    for _ in range(10):
        assert env.client.post("/api/auth/login", json={"username": "admin", "password": "x" * 5}).status_code == 401
    r = env.client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    env.clock.advance(61)
    assert env.client.post("/api/auth/login", json={"username": "admin", "password": "admin"}).status_code == 200


def test_password_spraying_from_one_address_is_limited(env):
    """Перебор логинов с одного адреса: общий лимит на адрес, а не только на пару «логин, адрес»."""
    codes = [
        env.client.post("/api/auth/login", json={"username": f"user{i}", "password": "123456"}).status_code
        for i in range(60)
    ]
    assert 429 in codes
    r = env.client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
    assert r.status_code == 429
    rows, _ = env.auth.store.query_audit(action="auth.login_failed", limit=1000)
    assert len(rows) <= 40  # журнал не раздувается перебором


# ---------------------------------------------------------------- утечки


def test_no_hashes_salts_or_passwords_in_responses_audit_and_logs(env, caplog):
    caplog.set_level(logging.DEBUG)
    admin = env.login("admin")
    secret = "S3cret-Pass-42"
    r = env.client.post("/api/users", headers=admin,
                        json={"username": "ivanov", "display_name": "Иванов", "role": "operator", "password": secret})
    assert r.status_code == 201
    env.client.patch("/api/users/ivanov", json={"password": secret + "x"}, headers=admin)
    env.client.post("/api/auth/login", json={"username": "ivanov", "password": "wrong-" + secret})
    bodies = [
        env.client.get("/api/users", headers=admin).text,
        env.client.get("/api/audit?limit=1000", headers=admin).text,
        env.client.get("/api/auth/me", headers=admin).text,
        env.client.get("/api/auth/config").text,
        caplog.text,
    ]
    for text in bodies:
        assert "scrypt$" not in text and "password_hash" not in text and secret not in text
        assert _token(admin) not in text


def test_camera_password_hidden_from_non_admin_and_kept_on_zone_edit(env):
    admin, op, safety = env.login("admin"), env.login("operator"), env.login("safety")
    body = camera_dict(env.folder, camera_id="cam2", source={"type": "rtsp", "uri": "rtsp://u:topsecret@10.0.0.5/s?token=abc"})
    assert env.client.post("/api/cameras", json=body, headers=admin).status_code == 201
    for h in (op, safety):
        for path in ("/api/cameras", "/api/cameras/cam2"):
            text = env.client.get(path, headers=h).text
            assert "topsecret" not in text and "abc" not in text, path
    assert "topsecret" in env.client.get("/api/cameras/cam2", headers=admin).text
    # инженер по ОТ сохраняет зоны с замаскированным адресом — адрес не меняется, права cameras.manage не нужно
    cam = env.client.get("/api/cameras/cam2", headers=safety).json()
    cam.pop("status", None)
    cam["zones"][0]["name"] = "Новая"
    r = env.client.put("/api/cameras/cam2", json=cam, headers=safety)
    assert r.status_code == 200, r.text
    assert env.engine.get_camera("cam2").source.uri == "rtsp://u:topsecret@10.0.0.5/s?token=abc"
    assert "topsecret" not in r.text
    # в журнале действий адрес без пароля и токена
    audit = env.client.get("/api/audit?limit=1000", headers=admin).text
    assert "topsecret" not in audit and "token=abc" not in audit


# ---------------------------------------------------------------- демо-учётки


def test_no_demo_accounts_without_demo_login(tmp_path, caplog):
    with auth_env(tmp_path, demo_login=False) as e:
        users = {u.username: u for u in e.auth.store.list_users()}
        assert not any(u.demo for u in users.values())
        for name in ("admin", "safety", "operator", "worker"):
            r = e.client.post("/api/auth/login", json={"username": name, "password": name})
            assert r.status_code in (401, 429), name
        assert e.client.get("/api/auth/config").json()["demo_users"] == []
        # первый администратор — со случайным паролем из файла рядом с базой
        secret_file = e.settings.auth_db_path.parent / "auth-initial-admin.txt"
        assert secret_file.exists() and (secret_file.stat().st_mode & 0o077) == 0
        password = secret_file.read_text(encoding="utf-8").strip().splitlines()[-1].split()[-1]
        assert password not in caplog.text
        assert e.client.post("/api/auth/login", json={"username": "admin", "password": password}).status_code == 200


def test_old_demo_accounts_are_locked_when_demo_login_turned_off(tmp_path):
    with auth_env(tmp_path, demo_login=True) as e:
        op = e.login("operator")
    with auth_env(tmp_path, demo_login=False) as e:
        assert e.client.get("/api/auth/me", headers=op).status_code == 401  # старая сессия демо-учётки
        r = e.client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        assert r.status_code == 401
        e.auth.store.update_user("operator", password="changed-pass")
        r = e.client.post("/api/auth/login", json={"username": "operator", "password": "changed-pass"})
        assert r.status_code == 200


# ---------------------------------------------------------------- сотрудники


@pytest.mark.parametrize("role", ["worker", "operator", "safety"])
def test_x_guard_role_header_does_not_raise_rights(env, role):
    h = {**env.login(role), "X-Guard-Role": "admin"}
    r = env.client.post("/api/employees", json={"full_name": "Подмена"}, headers=h)
    assert r.status_code == 403


def test_x_guard_role_header_does_not_grant_assignment(env):
    for role in ("worker", "safety"):
        h = {**env.login(role), "X-Guard-Role": "operator"}
        assert env.client.put("/api/events/ev1/employees", json={"employee_ids": []}, headers=h).status_code == 403
    assert env.client.get("/api/employees", headers={"X-Guard-Role": "admin"}).status_code == 401


# ---------------------------------------------------------------- проверка потока (SSRF)


@pytest.fixture()
def probe_calls(monkeypatch):
    from guard.sources import probe as probe_mod

    calls = []

    def fake(cfg, uri, timeout_s, **kw):
        calls.append((cfg.type, uri))
        return {"ok": True, "frames": 1}

    monkeypatch.setattr(probe_mod, "probe_source", fake)
    return calls


@pytest.mark.parametrize("source", [
    {"type": "rtsp", "uri": "file:///etc/passwd"},
    {"type": "rtsp", "uri": "/etc/passwd"},
    {"type": "rtsp", "uri": "/dev/video0"},
    {"type": "rtsp", "uri": "concat:/etc/passwd|/etc/hosts"},
    {"type": "rtsp", "uri": "http://169.254.169.254/latest/meta-data/"},
    {"type": "rtsp", "uri": "http://metadata.google.internal/computeMetadata/v1/"},
    {"type": "webcam", "uri": "/etc/passwd"},
    {"type": "webcam", "uri": "http://10.0.0.1/stream"},
    {"type": "file", "uri": "http://10.0.0.1/video.mp4"},
    {"type": "file", "uri": "file:///etc/passwd"},
    {"type": "file", "uri": "/dev/video0"},
    {"type": "images", "uri": "/proc/self"},
])
def test_probe_rejects_dangerous_sources(env, probe_calls, source):
    """Проверка потока отвечает как на неверный адрес (ok=false, bad_uri) и ничего не открывает; камеру — 422."""
    admin = env.login("admin")
    r = env.client.post("/api/cameras/probe", json={"source": source}, headers=admin)
    assert r.status_code == 200, (source, r.status_code, r.text)
    assert r.json()["ok"] is False and r.json()["error_code"] == "bad_uri", r.text
    assert not probe_calls
    body = camera_dict(env.folder, camera_id="bad", source=source)
    assert env.client.post("/api/cameras", json=body, headers=admin).status_code == 422


@pytest.mark.parametrize("source", [
    {"type": "rtsp", "uri": "rtsp://127.0.0.1:8554/demo"},
    {"type": "rtsp", "uri": "rtsp://user:pw@192.168.1.10:554/stream1"},
    {"type": "rtsp", "uri": "https://www.youtube.com/watch?v=abc"},
    {"type": "rtsp", "uri": "http://cam.local/video.mjpg"},
    {"type": "webcam", "uri": "0"},
    {"type": "file", "uri": "videos/demo.mp4"},
])
def test_probe_allows_demo_sources(env, probe_calls, source):
    admin = env.login("admin")
    r = env.client.post("/api/cameras/probe", json={"source": source}, headers=admin)
    assert r.status_code == 200, r.text
    assert probe_calls and probe_calls[-1][0] == source["type"]


def test_probe_is_admin_only_and_audited(env, probe_calls):
    safety, admin = env.login("safety"), env.login("admin")
    src = {"type": "rtsp", "uri": "rtsp://u:pw123@10.0.0.7/s"}
    assert env.client.post("/api/cameras/probe", json={"source": src}, headers=safety).status_code == 403
    assert env.client.post("/api/cameras/probe", json={"source": src}, headers=admin).status_code == 200
    rows, _ = env.auth.store.query_audit(action="camera.probe")
    assert len(rows) == 1 and "pw123" not in json.dumps(rows) and rows[0]["username"] == "admin"


# ---------------------------------------------------------------- журнал действий


def test_every_action_from_roles_md_is_audited(env, probe_calls):
    admin, op, safety = env.login("admin"), env.login("operator"), env.login("safety")
    env.client.post("/api/auth/login", json={"username": "worker", "password": "bad-pass"})
    c = env.client
    body = camera_dict(env.folder, camera_id="cam3")
    assert c.post("/api/cameras", json=body, headers=admin).status_code == 201
    body["name"] = "Склад"
    assert c.put("/api/cameras/cam3", json=body, headers=admin).status_code == 200
    body["zones"][0]["name"] = "Другая"
    body["rules"][0]["min_duration_s"] = 3.0
    assert c.put("/api/cameras/cam3", json=body, headers=safety).status_code == 200
    assert c.delete("/api/cameras/cam3", headers=admin).status_code == 204
    assert c.patch("/api/events/ev1", json={"status": "confirmed"}, headers=op).status_code == 200
    assert c.post("/api/users", json={"username": "petrov", "display_name": "П", "role": "worker", "password": "pw-1234"},
                  headers=admin).status_code == 201
    assert c.patch("/api/users/petrov", json={"role": "operator"}, headers=admin).status_code == 200
    r = c.post("/api/employees", json={"full_name": "Сидоров Сидор"}, headers=admin)
    assert r.status_code == 201
    emp = r.json()["id"]
    assert c.put("/api/events/ev1/employees", json={"employee_ids": [emp]}, headers=op).status_code == 200
    assert c.post("/api/cameras/probe", json={"source": {"type": "webcam", "uri": "0"}}, headers=admin).status_code == 200
    assert c.post("/api/auth/logout", headers=op).status_code == 200
    rows, _ = env.auth.store.query_audit(limit=1000)
    actions = {r["action"] for r in rows}
    expected = {
        "auth.login", "auth.login_failed", "auth.logout", "camera.create", "camera.update", "camera.delete",
        "zones.update", "rules.update", "event.update", "user.create", "user.update", "employee.create",
        "event.assign", "camera.probe",
    }
    assert expected <= actions, expected - actions
    assign = next(r for r in rows if r["action"] == "event.assign")
    assert assign["username"] == "operator" and assign["target"] == "ev1" and assign["details"]["employee_ids"] == [emp]


def test_cli_password_change_is_audited_and_banner_points_to_initial_admin(tmp_path, monkeypatch, capsys):
    import io

    from guard.auth import __main__ as cli
    from guard.auth.service import AuthService

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GUARD_AUTH", "1")
    monkeypatch.setenv("GUARD_DEMO_LOGIN", "0")
    service = AuthService(enabled=True, demo_login=False, db_path=tmp_path / "data" / "auth.db")
    assert service.open() == ["admin"]
    service.close()
    cli.main(["banner"])
    out = capsys.readouterr().out
    assert "auth-initial-admin.txt" in out
    monkeypatch.setattr("sys.stdin", io.StringIO("brand-new-9\n"))
    assert cli.main(["set-password", "admin", "--password-stdin"]) == 0
    service = AuthService(enabled=True, demo_login=False, db_path=tmp_path / "data" / "auth.db")
    rows, _ = service.store.query_audit(action="user.update")
    service.close()
    assert rows and rows[0]["username"] == "cli" and rows[0]["details"]["password_changed"] is True
    assert "brand-new-9" not in json.dumps(rows)


def test_huge_body_without_session_is_413(env, monkeypatch):
    """Тело читается до проверки входа — без предела запрос без сессии занял бы память сервера."""
    from guard.api import limits

    big = b"{" + b" " * (limits.MAX_BODY_BYTES + 1) + b"}"
    r = env.client.post("/api/auth/login", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413

    def chunks():
        for _ in range(9):
            yield b" " * (1024 * 1024)

    r = env.client.post("/api/cameras", content=chunks(), headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    # обычные запросы не затронуты
    assert env.client.post("/api/auth/login", json={"username": "worker", "password": "worker"}).status_code == 200


def test_event_media_link_never_serves_databases(env):
    """Ссылка события ведёт только на кадр или ролик: auth.db и guard.db лежат в том же data_dir."""
    op = env.login("operator")
    env.engine.journal.update("ev1", snapshot_path="auth.db", clip_path="guard.db")
    assert env.client.get("/api/events/ev1/snapshot.jpg", headers=op).status_code == 404
    assert env.client.get("/api/events/ev1/clip.mp4", headers=op).status_code == 404


def test_actions_catalog_is_documented():
    """Каждое действие журнала названо в docs/ROLES.md — интерфейсу и проверяющему видно, что пишется."""
    from pathlib import Path

    from guard.auth.audit import ACTIONS

    doc = (Path(__file__).resolve().parents[1] / "docs" / "ROLES.md").read_text(encoding="utf-8")
    missing = [a for a in ACTIONS if f"`{a}`" not in doc]
    assert not missing, missing


# ---------------------------------------------------------------- модули по отдельности


def _origin_ok(origin, host="127.0.0.1:8000", forwarded=None, allowed=("http://localhost:5173",)):
    from starlette.datastructures import Headers

    from guard.auth.origin import OriginGuard

    raw = {"host": host, "origin": origin}
    if forwarded:
        raw["x-forwarded-host"] = forwarded
    return OriginGuard(None, allowed).allowed_origin(origin, Headers(raw))


def test_origin_rules():
    assert _origin_ok("http://127.0.0.1:8000")
    assert _origin_ok("http://localhost:5174")  # dev-сервер интерфейса с прокси на том же компьютере
    assert _origin_ok("http://localhost:5173/", host="10.0.0.2:8000")  # из GUARD_CORS_ORIGINS
    assert _origin_ok("http://demo.example", host="demo.example:80")  # порт по умолчанию
    assert _origin_ok("https://demo.example", host="127.0.0.1:8000", forwarded="demo.example, proxy")
    assert not _origin_ok("http://localhost:5174", host="10.0.0.2:8000")
    assert not _origin_ok("http://10.0.0.9:8000", host="10.0.0.2:8000")
    assert not _origin_ok("https://demo.example", host="demo.example:80")  # другая схема — другой порт
    assert not _origin_ok("null") and not _origin_ok("") and not _origin_ok("file://")
    assert not _origin_ok("http://[bad", host="x")


def test_origin_star_disables_the_check():
    from starlette.datastructures import Headers

    from guard.auth.origin import OriginGuard

    assert OriginGuard(None, ["*"]).allowed_origin("http://evil.example", Headers({"host": "x"}))


def test_policy_lists_match_guard_sources():
    from guard.auth import audit, source_policy
    from guard.sources import stream_url

    assert audit.SECRET_QUERY_KEYS == stream_url.SECRET_QUERY_KEYS
    for scheme in source_policy.NETWORK_SCHEMES:
        assert stream_url.is_network(f"{scheme}://h/x"), scheme
    for scheme in ("file", "concat", "subfile", "pipe", "data", "gopher", "ftp"):
        assert not stream_url.is_network(f"{scheme}://h/x")
        assert scheme not in source_policy.NETWORK_SCHEMES


@pytest.mark.parametrize("uri,expected", [
    ("rtsp://u:p@h:554/s", "rtsp://u:***@h:554/s"),
    ("rtsp://u:p@[::1]:554/s", "rtsp://u:***@[::1]:554/s"),
    ("rtsp://u:p@h:99999/s", "rtsp://u:***@h:99999/s"),
    ("rtsp://u:p:q@h/s", "rtsp://u:***@h/s"),
    ("rtsp://u@h/s", "rtsp://u@h/s"),
    ("http://h/live.m3u8?token=abc&x=1", "http://h/live.m3u8?token=***&x=1"),
    ("videos/a.mp4", "videos/a.mp4"),
])
def test_redact_uri(uri, expected):
    from guard.auth.audit import redact_uri

    assert redact_uri(uri) == expected


def test_source_policy_resolves_relative_paths_against_data_dir(tmp_path):
    from guard.auth.source_policy import source_problem
    from guard.contracts import SourceConfig

    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    escape = "../" * len(deep.parts) + "dev/null"
    assert source_problem(SourceConfig(type="file", uri=escape), deep)
    assert source_problem(SourceConfig(type="file", uri="videos/a.mp4"), deep) is None
    (deep / "link.mp4").symlink_to("/dev/null")
    assert source_problem(SourceConfig(type="file", uri="link.mp4"), deep)


def test_limiter_counts_failures_per_address():
    from guard.auth.ratelimit import LoginLimiter
    from guard.auth.testing import FakeClock

    clock = FakeClock(0.0)
    lim = LoginLimiter(max_failures=10, block_s=60.0, clock=clock, ip_max_failures=5)
    for i in range(4):
        assert lim.failure(f"u{i}", "1.1.1.1") == 0
    assert lim.failure("u4", "1.1.1.1") == 60.0
    assert lim.retry_after("admin", "1.1.1.1") == 60.0
    assert lim.retry_after("admin", "2.2.2.2") == 0
    lim.success("u4", "1.1.1.1")  # своя учётка не снимает паузу адреса
    assert lim.retry_after("admin", "1.1.1.1") == 60.0
    clock.advance(61)
    assert lim.retry_after("admin", "1.1.1.1") == 0
