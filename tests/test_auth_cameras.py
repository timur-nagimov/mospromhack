"""PUT /api/cameras/{id}: только zones и rules — нужно zones.edit, любые другие поля — cameras.manage."""

import copy

import pytest

from guard.auth import passwords
from guard.auth.testing import auth_env, camera_dict
from guard.contracts import CameraConfig
from guard.pipeline.testing import write_images

NEW_ZONE = {"zone_id": "z2", "name": "Проход", "kind": "danger", "polygon": [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6]]}


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


@pytest.fixture()
def env(tmp_path):
    with auth_env(tmp_path) as e:
        e.folder = write_images(tmp_path / "imgs", n=2)
        e.camera = camera_dict(e.folder)
        e.engine.add_camera(CameraConfig(**e.camera))
        yield e


def _body(env, **changes):
    body = copy.deepcopy(env.camera)
    body.update(changes)
    return body


def _audit(env, action):
    admin = env.login("admin")
    items = env.client.get("/api/audit", params={"action": action}, headers=admin).json()["items"]
    return items


def test_safety_changes_zones_only(env):
    body = _body(env, zones=[*env.camera["zones"], NEW_ZONE])
    r = env.client.put("/api/cameras/cam1", json=body, headers=env.login("safety"))
    assert r.status_code == 200, r.text
    assert [z.zone_id for z in env.engine.get_camera("cam1").zones] == ["z1", "z2"]
    items = _audit(env, "zones.update")
    assert len(items) == 1
    assert items[0]["username"] == "safety" and items[0]["role"] == "safety" and items[0]["target"] == "cam1"
    assert items[0]["details"]["zones"] == ["z1", "z2"]
    assert _audit(env, "camera.update") == [] and _audit(env, "rules.update") == []


def test_safety_changes_rules_only(env):
    rules = [{"rule_id": "s1", "type": "zone_intrusion", "zones": ["z1"], "severity": "critical"}]
    r = env.client.put("/api/cameras/cam1", json=_body(env, rules=rules), headers=env.login("safety"))
    assert r.status_code == 200, r.text
    assert env.engine.get_camera("cam1").rules[0].severity.value == "critical"
    assert [i["target"] for i in _audit(env, "rules.update")] == ["cam1"]


def test_read_only_status_field_does_not_count_as_change(env):
    body = _body(env, zones=[*env.camera["zones"], NEW_ZONE], status={"running": False})
    r = env.client.put("/api/cameras/cam1", json=body, headers=env.login("safety"))
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    "change",
    [
        {"source": {"type": "rtsp", "uri": "rtsp://cam.local/stream"}},
        {"name": "Другое имя"},
        {"site": "Цех 2"},
        {"sample_fps": 2},
        {"enabled": True},
    ],
    ids=["source", "name", "site", "sample_fps", "enabled"],
)
def test_safety_cannot_change_other_fields(env, change):
    before = env.engine.get_camera("cam1")
    r = env.client.put("/api/cameras/cam1", json=_body(env, **change), headers=env.login("safety"))
    assert r.status_code == 403
    assert r.json() == {"detail": "forbidden", "permission": "cameras.manage"}
    assert env.engine.get_camera("cam1") == before


def test_safety_cannot_mix_zones_with_source(env):
    body = _body(env, zones=[NEW_ZONE], source={"type": "rtsp", "uri": "rtsp://cam.local/stream"})
    r = env.client.put("/api/cameras/cam1", json=body, headers=env.login("safety"))
    assert r.status_code == 403 and r.json()["permission"] == "cameras.manage"


def test_admin_changes_source_and_audit_hides_credentials(env):
    body = _body(env, source={"type": "rtsp", "uri": "rtsp://viewer:s3cret@10.0.0.7:554/main"},
                 zones=[*env.camera["zones"], NEW_ZONE])
    r = env.client.put("/api/cameras/cam1", json=body, headers=env.login("admin"))
    assert r.status_code == 200, r.text
    updates = _audit(env, "camera.update")
    assert len(updates) == 1 and updates[0]["details"]["fields"] == ["source"]
    assert "s3cret" not in str(updates[0]["details"])
    assert "rtsp://viewer:***@10.0.0.7:554/main" in str(updates[0]["details"])
    assert [i["target"] for i in _audit(env, "zones.update")] == ["cam1"]


@pytest.mark.parametrize("role", ["worker", "operator"])
def test_roles_without_zone_rights_get_403(env, role):
    body = _body(env, zones=[NEW_ZONE])
    r = env.client.put("/api/cameras/cam1", json=body, headers=env.login(role))
    assert r.status_code == 403 and r.json() == {"detail": "forbidden", "permission": "zones.edit"}


def test_missing_camera_needs_cameras_manage(env):
    """Зонами несуществующую камеру не изменить: safety получает 403, администратор — 404 от движка."""
    body = _body(env, camera_id="ghost")
    r = env.client.put("/api/cameras/ghost", json=body, headers=env.login("safety"))
    assert r.status_code == 403 and r.json()["permission"] == "cameras.manage"
    assert env.client.put("/api/cameras/ghost", json=body, headers=env.login("admin")).status_code == 404


def test_unchanged_put_is_allowed_and_not_audited(env):
    r = env.client.put("/api/cameras/cam1", json=_body(env), headers=env.login("safety"))
    assert r.status_code == 200
    assert _audit(env, "zones.update,rules.update,camera.update") == []


def test_create_and_delete_are_admin_only_and_audited(env):
    body = camera_dict(env.folder, camera_id="cam2")
    safety, admin = env.login("safety"), env.login("admin")
    r = env.client.post("/api/cameras", json=body, headers=safety)
    assert r.status_code == 403 and r.json()["permission"] == "cameras.manage"
    assert env.client.post("/api/cameras", json=body, headers=admin).status_code == 201
    assert env.client.delete("/api/cameras/cam2", headers=safety).status_code == 403
    assert env.client.delete("/api/cameras/cam2", headers=admin).status_code == 204
    created, deleted = _audit(env, "camera.create"), _audit(env, "camera.delete")
    assert [(i["target"], i["username"]) for i in created] == [("cam2", "admin")]
    assert created[0]["details"]["source"]["type"] == "images"
    assert [i["target"] for i in deleted] == ["cam2"]
