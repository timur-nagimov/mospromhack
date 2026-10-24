"""/api/users: список, создание, изменение; защита последнего администратора; закрытие сессий."""

import pytest

from guard.auth import passwords
from guard.auth.testing import auth_env


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        e.admin = e.login("admin")
        yield e


def _create(env, username="ivanov", role="worker", password="pass1234", display_name="Иванов И. И."):
    return env.client.post(
        "/api/users",
        json={"username": username, "display_name": display_name, "role": role, "password": password},
        headers=env.admin,
    )


def test_list_users_without_secrets(env):
    r = env.client.get("/api/users", headers=env.admin)
    assert r.status_code == 200
    users = r.json()
    assert [u["username"] for u in users] == ["admin", "operator", "safety", "worker"]
    assert set(users[0]) == {"username", "display_name", "role", "active", "created_at", "last_login_at"}
    assert "scrypt" not in r.text


def test_create_user_and_login(env):
    r = _create(env)
    assert r.status_code == 201, r.text
    assert r.json()["username"] == "ivanov" and r.json()["role"] == "worker" and r.json()["active"] is True
    headers = env.login("ivanov", "pass1234")
    assert env.client.get("/api/auth/me", headers=headers).json()["display_name"] == "Иванов И. И."


@pytest.mark.parametrize(
    "kwargs",
    [
        {"role": "ghost"},
        {"password": "123"},
        {"username": "a"},
        {"username": "Иванов"},
        {"username": "bad name"},
        {"display_name": ""},
    ],
)
def test_create_validation(env, kwargs):
    assert _create(env, **kwargs).status_code == 422


def test_create_duplicate_is_409(env):
    assert _create(env).status_code == 201
    assert _create(env, username="IVANOV").status_code == 409


def test_role_change_applies_to_live_session(env):
    _create(env)
    user = env.login("ivanov", "pass1234")
    assert env.client.get("/api/cameras", headers=user).status_code == 403
    r = env.client.patch("/api/users/ivanov", json={"role": "operator"}, headers=env.admin)
    assert r.status_code == 200 and r.json()["role"] == "operator"
    assert env.client.get("/api/cameras", headers=user).status_code == 200


def test_password_change_closes_user_sessions(env):
    _create(env)
    user = env.login("ivanov", "pass1234")
    r = env.client.patch("/api/users/ivanov", json={"password": "new-pass-1"}, headers=env.admin)
    assert r.status_code == 200
    assert env.client.get("/api/auth/me", headers=user).status_code == 401
    env.login("ivanov", "new-pass-1")
    bad = env.client.post("/api/auth/login", json={"username": "ivanov", "password": "pass1234"})
    assert bad.status_code == 401


def test_deactivation_closes_sessions_and_blocks_login(env):
    _create(env)
    user = env.login("ivanov", "pass1234")
    r = env.client.patch("/api/users/ivanov", json={"active": False}, headers=env.admin)
    assert r.status_code == 200 and r.json()["active"] is False
    assert env.client.get("/api/auth/me", headers=user).status_code == 401
    r = env.client.post("/api/auth/login", json={"username": "ivanov", "password": "pass1234"})
    assert r.status_code == 401 and r.json() == {"detail": "invalid credentials"}
    assert env.client.patch("/api/users/ivanov", json={"active": True}, headers=env.admin).status_code == 200
    env.login("ivanov", "pass1234")


def test_own_password_change_keeps_current_session(env):
    other = env.login("admin")
    r = env.client.patch("/api/users/admin", json={"password": "admin-new"}, headers=env.admin)
    assert r.status_code == 200
    assert env.client.get("/api/auth/me", headers=env.admin).status_code == 200
    assert env.client.get("/api/auth/me", headers=other).status_code == 401


def test_last_active_admin_is_protected(env):
    r = env.client.patch("/api/users/admin", json={"role": "operator"}, headers=env.admin)
    assert r.status_code == 409 and r.json() == {"detail": "last active admin"}
    r = env.client.patch("/api/users/admin", json={"active": False}, headers=env.admin)
    assert r.status_code == 409
    assert _create(env, username="root2", role="admin").status_code == 201
    r = env.client.patch("/api/users/admin", json={"role": "safety"}, headers=env.admin)
    assert r.status_code == 200
    assert env.client.get("/api/users", headers=env.admin).status_code == 403  # права уже safety


def test_patch_validation(env):
    assert env.client.patch("/api/users/worker", json={}, headers=env.admin).status_code == 422
    assert env.client.patch("/api/users/worker", json={"role": "boss"}, headers=env.admin).status_code == 422
    assert env.client.patch("/api/users/worker", json={"password": "1"}, headers=env.admin).status_code == 422
    assert env.client.patch("/api/users/ghost", json={"display_name": "x"}, headers=env.admin).status_code == 404


def test_users_need_users_manage(env):
    safety = env.login("safety")
    r = env.client.get("/api/users", headers=safety)
    assert r.status_code == 403 and r.json() == {"detail": "forbidden", "permission": "users.manage"}
