"""PUT /api/cameras/{id}: что изменилось относительно текущей конфигурации и какое право для этого нужно.

Только zones и rules — zones.edit (инженер по ОТ). Любое другое поле, включая источник, имя, участок,
частоту и включение, — cameras.manage (администратор).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from guard.auth import audit
from guard.auth.roles import CAMERAS_MANAGE, ZONES_EDIT
from guard.contracts import CameraConfig
from guard.rules.base import with_default_severity

ZONE_FIELDS = frozenset({"zones", "rules"})


@dataclass(frozen=True)
class CameraEdit:
    fields: tuple[str, ...]  # изменённые поля, кроме zones и rules
    zones: bool
    rules: bool

    @property
    def permission(self) -> str:
        return CAMERAS_MANAGE if self.fields else ZONES_EDIT


def hide_source_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Камера для роли без cameras.manage: пароль и токены в адресе источника заменены на ***."""
    source = data.get("source")
    if not isinstance(source, dict) or not source.get("uri"):
        return data
    return {**data, "source": {**source, "uri": audit.redact_uri(source["uri"])}}


def restore_hidden_source(current: CameraConfig, new: CameraConfig) -> CameraConfig:
    """PUT с адресом, замаскированным hide_source_secrets, — адрес не меняется (зоны правит инженер по ОТ)."""
    old_uri, new_uri = current.source.uri, new.source.uri
    if new_uri == old_uri or new_uri != audit.redact_uri(old_uri):
        return new
    return new.model_copy(update={"source": new.source.model_copy(update={"uri": old_uri})})


def camera_edit(current: CameraConfig, new: CameraConfig) -> CameraEdit:
    """Сравнить с текущей конфигурацией. Правила сравниваются после подстановки severity по умолчанию —
    так же, как их сохранит движок."""
    before = current.model_dump(mode="json")
    after = with_default_severity(new).model_dump(mode="json")
    fields = tuple(k for k in after if k not in ZONE_FIELDS and before.get(k) != after.get(k))
    return CameraEdit(fields, before["zones"] != after["zones"], before["rules"] != after["rules"])


def audit_entries(edit: CameraEdit, current: CameraConfig, new: CameraConfig) -> list[tuple[str, dict[str, Any]]]:
    """Записи журнала: camera.update по прочим полям, zones.update и rules.update — отдельно."""
    entries: list[tuple[str, dict[str, Any]]] = []
    if edit.fields:
        details: dict[str, Any] = {"fields": list(edit.fields)}
        for name in edit.fields:
            if name == "source":
                details["source"] = audit.source_details(new.source)
                details["previous_source"] = audit.source_details(current.source)
            elif name in ("name", "site", "sample_fps", "enabled"):
                details[name] = {"from": getattr(current, name), "to": getattr(new, name)}
        entries.append((audit.CAMERA_UPDATE, details))
    if edit.zones:
        entries.append((audit.ZONES_UPDATE, {
            "zones": [z.zone_id for z in new.zones], "previous": [z.zone_id for z in current.zones],
        }))
    if edit.rules:
        entries.append((audit.RULES_UPDATE, {
            "rules": [r.rule_id for r in new.rules], "previous": [r.rule_id for r in current.rules],
        }))
    return entries
