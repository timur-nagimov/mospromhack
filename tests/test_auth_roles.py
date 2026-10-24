"""Роли и права: матрица в guard/auth/roles.py совпадает с таблицей docs/ROLES.md."""

import re
from pathlib import Path

from guard.auth import roles

ROLES_MD = Path(__file__).resolve().parents[1] / "docs" / "ROLES.md"

OPERATOR = {"analytics.view", "events.view", "video.view", "events.handle", "cameras.view", "system.view"}
EXPECTED = {
    "worker": {"analytics.view", "events.view"},
    "operator": OPERATOR,
    "safety": OPERATOR | {"zones.edit", "flywheel.review", "audit.view"},
    "admin": OPERATOR
    | {"zones.edit", "flywheel.review", "audit.view", "cameras.manage", "flywheel.train", "users.manage"},
}


def test_matrix_matches_contract():
    assert {r: set(roles.permissions_for(r)) for r in roles.ROLE_IDS} == EXPECTED


def test_roles_go_from_least_to_most_privileged():
    assert roles.ROLE_IDS == ("worker", "operator", "safety", "admin")
    for lower, higher in zip(roles.ROLE_IDS, roles.ROLE_IDS[1:]):
        assert roles.permissions_for(lower) < roles.permissions_for(higher)


def test_admin_has_every_permission():
    assert roles.permissions_for("admin") == frozenset(roles.PERMISSIONS)


def test_russian_titles_and_descriptions():
    titles = {r: roles.ROLES[r].title for r in roles.ROLE_IDS}
    assert titles == {
        "worker": "Сотрудник",
        "operator": "Диспетчер смены",
        "safety": "Инженер по ОТ",
        "admin": "Администратор",
    }
    for role in roles.ROLES.values():
        assert re.search("[а-яё]", role.description)
    for perm in roles.PERMISSIONS.values():
        assert perm.title and re.search("[а-яё]", perm.description)


def test_unknown_role_has_no_permissions():
    assert roles.permissions_for("ghost") == frozenset()
    assert not roles.has_permission("ghost", "events.view")
    assert roles.has_permission("worker", "events.view")
    assert not roles.has_permission("worker", "video.view")


def test_sorted_permissions_follow_catalog_order():
    assert roles.sorted_permissions({"users.manage", "analytics.view"}) == ["analytics.view", "users.manage"]


def _doc_matrix() -> dict[str, set[str]]:
    """Таблица «Права» из docs/ROLES.md: строки вида | `perm` | ... | ✓ | | ✓ | ✓ |."""
    text = ROLES_MD.read_text(encoding="utf-8")
    section = text.split("## Права", 1)[1].split("\n## ", 1)[0]
    found: dict[str, set[str]] = {r: set() for r in roles.ROLE_IDS}
    for line in section.splitlines():
        m = re.match(r"\|\s*`([a-z]+\.[a-z]+)`\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        marks = cells[-4:]  # worker, operator, safety, admin
        for role, mark in zip(roles.ROLE_IDS, marks):
            if mark == "✓":
                found[role].add(m.group(1))
    return found


def test_code_and_roles_md_agree():
    assert _doc_matrix() == {r: set(roles.permissions_for(r)) for r in roles.ROLE_IDS}
