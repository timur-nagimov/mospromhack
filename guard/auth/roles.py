"""Роли и права — источник правды для матрицы из docs/ROLES.md.

Право — строка вида «область.действие». Роль — набор прав; роли упорядочены от меньших прав к большим,
и каждая следующая включает предыдущую.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

ANALYTICS_VIEW = "analytics.view"
EVENTS_VIEW = "events.view"
VIDEO_VIEW = "video.view"
EVENTS_HANDLE = "events.handle"
CAMERAS_VIEW = "cameras.view"
SYSTEM_VIEW = "system.view"
ZONES_EDIT = "zones.edit"
FLYWHEEL_REVIEW = "flywheel.review"
AUDIT_VIEW = "audit.view"
CAMERAS_MANAGE = "cameras.manage"
FLYWHEEL_TRAIN = "flywheel.train"
USERS_MANAGE = "users.manage"
EMPLOYEES_VIEW = "employees.view"
EMPLOYEES_MANAGE = "employees.manage"
EVENTS_ASSIGN = "events.assign"

ADMIN = "admin"


@dataclass(frozen=True)
class Permission:
    id: str
    title: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "description": self.description}


@dataclass(frozen=True)
class Role:
    id: str
    title: str
    description: str
    permissions: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "permissions": sorted_permissions(self.permissions),
        }


_CATALOG = (
    Permission(ANALYTICS_VIEW, "Аналитика", "Сводка нарушений по участкам, типам и сменам, отчёты за смену"),
    Permission(EVENTS_VIEW, "Журнал событий", "Список событий и их метаданные без снимков и видео, поток событий"),
    Permission(VIDEO_VIEW, "Видео и снимки", "Кадры и видео камер, снимки и клипы событий (изображения людей)"),
    Permission(EVENTS_HANDLE, "Разбор событий", "Подтвердить событие, отметить ложным, закрыть, написать заметку"),
    Permission(CAMERAS_VIEW, "Камеры: просмотр", "Список камер с зонами и правилами, типы правил"),
    Permission(SYSTEM_VIEW, "Производительность", "Метрики конвейера: кадры в секунду, задержки, загрузка ускорителя"),
    Permission(ZONES_EDIT, "Зоны и правила", "Изменить опасные зоны и правила камеры"),
    Permission(FLYWHEEL_REVIEW, "Данные: проверка", "Очередь проверки кадров маховика данных, отправка кадра в выборку"),
    Permission(AUDIT_VIEW, "Журнал действий", "Кто, когда и что менял в системе"),
    Permission(CAMERAS_MANAGE, "Камеры: управление", "Добавить и удалить камеру, сменить источник и другие настройки"),
    Permission(FLYWHEEL_TRAIN, "Обучение моделей", "Выгрузить датасет, запустить обучение, проверку и выкатку модели"),
    Permission(USERS_MANAGE, "Пользователи", "Создать учётную запись, сменить роль или пароль, отключить"),
    Permission(EMPLOYEES_VIEW, "Сотрудники", "Справочник сотрудников и персональная статистика (docs/EMPLOYEES.md)"),
    Permission(EMPLOYEES_MANAGE, "Сотрудники: управление", "Добавить сотрудника предприятия"),
    Permission(EVENTS_ASSIGN, "Виновники инцидента", "Сохранить список виновников инцидента"),
)
PERMISSIONS: dict[str, Permission] = {p.id: p for p in _CATALOG}
_ORDER = {p.id: i for i, p in enumerate(_CATALOG)}

_WORKER = frozenset({ANALYTICS_VIEW, EVENTS_VIEW, EMPLOYEES_VIEW})
_OPERATOR = _WORKER | {VIDEO_VIEW, EVENTS_HANDLE, CAMERAS_VIEW, SYSTEM_VIEW, EVENTS_ASSIGN}
# Виновников инцидента записывают только диспетчер и администратор; инженер по ОТ читает список (docs/ROLES.md)
_SAFETY = (_OPERATOR - {EVENTS_ASSIGN}) | {ZONES_EDIT, FLYWHEEL_REVIEW, AUDIT_VIEW}
_ADMIN = frozenset(PERMISSIONS)

ROLES: dict[str, Role] = {
    r.id: r
    for r in (
        Role("worker", "Сотрудник", "Рабочий, бригадир: смотрит дашборды и журнал событий без снимков и видео", _WORKER),
        Role(
            "operator",
            "Диспетчер смены",
            "Мастер смены, диспетчер: следит за камерами, разбирает события, выгружает отчёты",
            _OPERATOR,
        ),
        Role(
            "safety",
            "Инженер по ОТ",
            "Специалист по охране труда: всё, что диспетчер, плюс зоны и правила камер, проверка кадров, журнал действий",
            _SAFETY,
        ),
        Role(
            ADMIN,
            "Администратор",
            "ИТ, владелец системы: всё, плюс камеры, учётные записи, обучение и выкатка моделей",
            _ADMIN,
        ),
    )
}
ROLE_IDS: tuple[str, ...] = tuple(ROLES)


def permissions_for(role: str) -> frozenset[str]:
    """Права роли; неизвестная роль — пустой набор."""
    found = ROLES.get(role)
    return found.permissions if found else frozenset()


def has_permission(role: str, permission: str) -> bool:
    return permission in permissions_for(role)


def role_title(role: str) -> str:
    found = ROLES.get(role)
    return found.title if found else role


def sorted_permissions(permissions: Iterable[str]) -> list[str]:
    """Права в порядке каталога (как в таблице docs/ROLES.md)."""
    return sorted(permissions, key=lambda p: (_ORDER.get(p, len(_ORDER)), p))
