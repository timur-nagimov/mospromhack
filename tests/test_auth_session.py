"""Вход, выход, /api/auth/me, cookie guard_session, срок сессии, /api/auth/config и /api/auth/roles."""

import pytest

from guard.auth import passwords
from guard.auth.testing import SESSION_COOKIE, auth_env
from guard.pipeline import settings as settings_mod
from guard.pipeline.settings import Settings

HOUR = 3600.0


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        yield e


def test_login_sets_http_only_session_cookie_and_returns_me(env):
    r = env.client.post("/api/auth/login", json={"username": "operator", "password": "operator"})
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["username"] == "operator" and me["role"] == "operator"
    assert me["role_title"] == "Диспетчер смены" and me["display_name"] == "Диспетчер смены"
    assert "video.view" in me["permissions"] and "users.manage" not in me["permissions"]
    cookie = r.headers["set-cookie"]
    low = cookie.lower()
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "httponly" in low and "samesite=lax" in low and "path=/" in low
    assert "max-age" not in low  # сессионная cookie: срок держит сервер
    assert "password" not in r.text and "scrypt" not in r.text


def test_me_requires_session(env):
    r = env.client.get("/api/auth/me")
    assert r.status_code == 401
    assert r.json() == {"detail": "auth required"}


def test_me_with_session_then_logout(env):
    headers = env.login("safety")
    me = env.client.get("/api/auth/me", headers=headers).json()
    assert me["username"] == "safety" and "zones.edit" in me["permissions"]
    out = env.client.post("/api/auth/logout", headers=headers)
    assert out.status_code == 200
    assert SESSION_COOKIE in out.headers.get("set-cookie", "")  # cookie удаляется
    assert env.client.get("/api/auth/me", headers=headers).status_code == 401


def test_logout_requires_session(env):
    assert env.client.post("/api/auth/logout").status_code == 401


@pytest.mark.parametrize("username,password", [("admin", "wrong"), ("ghost", "ghost"), ("ADMIN", "Admin")])
def test_bad_credentials_give_401_without_details(env, username, password):
    r = env.client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 401
    assert r.json() == {"detail": "invalid credentials"}
    assert "set-cookie" not in r.headers


def test_username_is_case_insensitive(env):
    r = env.client.post("/api/auth/login", json={"username": " Admin ", "password": "admin"})
    assert r.status_code == 200 and r.json()["username"] == "admin"


def test_login_body_is_validated(env):
    assert env.client.post("/api/auth/login", json={"username": "admin"}).status_code == 422


def test_session_expires_after_12h_idle(env):
    headers = env.login("worker")
    env.clock.advance(11 * HOUR)
    assert env.client.get("/api/auth/me", headers=headers).status_code == 200  # активность сдвигает срок
    env.clock.advance(11 * HOUR)
    assert env.client.get("/api/auth/me", headers=headers).status_code == 200
    env.clock.advance(12 * HOUR + 1)
    assert env.client.get("/api/auth/me", headers=headers).status_code == 401


def test_garbage_cookie_is_401(env):
    for value in ("x", "a" * 1000, ""):
        r = env.client.get("/api/stats", headers={"Cookie": f"{SESSION_COOKIE}={value}"})
        assert r.status_code == 401


def test_last_login_is_recorded(env):
    env.login("worker")
    admin = env.login("admin")
    users = {u["username"]: u for u in env.client.get("/api/users", headers=admin).json()}
    assert users["worker"]["last_login_at"] == env.clock()
    assert users["operator"]["last_login_at"] is None


def test_config_is_public_and_lists_demo_users(env):
    r = env.client.get("/api/auth/config")
    assert r.status_code == 200
    data = r.json()
    assert data["auth_enabled"] is True and data["demo_login"] is True
    assert data["demo_users"] == [
        {"username": "worker", "display_name": "Сотрудник", "role": "worker"},
        {"username": "operator", "display_name": "Диспетчер смены", "role": "operator"},
        {"username": "safety", "display_name": "Инженер по ОТ", "role": "safety"},
        {"username": "admin", "display_name": "Администратор", "role": "admin"},
    ]


def test_demo_user_with_changed_password_leaves_demo_list(env):
    admin = env.login("admin")
    assert env.client.patch("/api/users/worker", json={"password": "secret-77"}, headers=admin).status_code == 200
    names = [u["username"] for u in env.client.get("/api/auth/config").json()["demo_users"]]
    assert names == ["operator", "safety", "admin"]


def test_config_without_demo_login(tmp_path):
    with auth_env(tmp_path, demo_login=False) as e:
        data = e.client.get("/api/auth/config").json()
        assert data == {"auth_enabled": True, "demo_login": False, "demo_users": []}
        # демо-учёток с паролем, равным логину, нет: только admin с паролем из файла рядом с базой
        r = e.client.post("/api/auth/login", json={"username": "admin", "password": "admin"})
        assert r.status_code == 401
        password = e.auth.initial_admin_file.read_text(encoding="utf-8").split()[-1]
        e.login("admin", password)


def test_roles_catalog(env):
    headers = env.login("worker")
    data = env.client.get("/api/auth/roles", headers=headers).json()
    assert [r["id"] for r in data["roles"]] == ["worker", "operator", "safety", "admin"]
    worker = data["roles"][0]
    assert worker["title"] == "Сотрудник" and worker["permissions"] == ["analytics.view", "events.view", "employees.view"]
    assert worker["description"]
    perms = data["permissions"]
    assert len(perms) == 15 and perms[0] == {
        "id": "analytics.view", "title": perms[0]["title"], "description": perms[0]["description"],
    }
    assert {p["id"] for p in perms} >= {"zones.edit", "users.manage", "flywheel.train"}


def test_auth_disabled_makes_everyone_admin(tmp_path):
    with auth_env(tmp_path, auth=False) as e:
        me = e.client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "admin" and me.json()["role"] == "admin"
        assert "users.manage" in me.json()["permissions"]
        assert e.client.get("/api/auth/config").json()["auth_enabled"] is False
        assert e.client.get("/api/users").status_code == 200
        assert e.client.post("/api/auth/logout").status_code == 200


def test_settings_in_code_keep_auth_off_and_env_turns_it_on(tmp_path, monkeypatch):
    assert Settings(
        data_dir=tmp_path, models_dir=tmp_path, cameras_config=tmp_path / "c.yaml", models_config=tmp_path / "m.yaml",
        db_path=tmp_path / "g.db", media_dir=tmp_path / "media", frontend_dist=tmp_path / "dist",
    ).auth is False
    monkeypatch.setattr(settings_mod, "load_dotenv_guard", lambda *a, **k: None)
    for name in ("GUARD_AUTH", "GUARD_DEMO_LOGIN", "GUARD_AUTH_DB", "GUARD_CORS_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GUARD_DATA_DIR", str(tmp_path / "d"))
    s = Settings.from_env()
    assert s.auth is True and s.demo_login is True
    assert s.auth_db_path == tmp_path / "d" / "auth.db"
    monkeypatch.setenv("GUARD_AUTH", "0")
    monkeypatch.setenv("GUARD_DEMO_LOGIN", "0")
    monkeypatch.setenv("GUARD_AUTH_DB", str(tmp_path / "other.db"))
    monkeypatch.setenv("GUARD_CORS_ORIGINS", "http://a.example, http://b.example")
    s = Settings.from_env()
    assert s.auth is False and s.demo_login is False and s.auth_db_path == tmp_path / "other.db"
    assert s.cors_origins == ["http://a.example", "http://b.example"]


def test_cors_allows_credentials_only_for_explicit_origins(tmp_path):
    with auth_env(tmp_path, cors_origins=["http://localhost:5173"]) as e:
        r = e.client.get("/api/auth/config", headers={"Origin": "http://localhost:5173"})
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
        assert r.headers["access-control-allow-credentials"] == "true"
        evil = e.client.get("/api/auth/config", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in evil.headers
    with auth_env(tmp_path / "star", cors_origins=["*"]) as e:
        r = e.client.get("/api/auth/config", headers={"Origin": "http://evil.example", "Cookie": "x=1"})
        assert "access-control-allow-credentials" not in r.headers
