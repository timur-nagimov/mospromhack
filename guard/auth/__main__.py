"""Консоль входа: python -m guard.auth banner | set-password <логин> | routes.

banner        — что печатают scripts/run_dev.sh и run_tpu.sh перед стартом: включён ли вход и демо-учётки;
set-password  — сменить пароль без интерфейса (например, если забыт пароль администратора);
                сессии пользователя закрываются;
routes        — таблица «маршрут → право» по текущему коду (для docs/ROLES.md).
"""

from __future__ import annotations

import argparse
import getpass
import sqlite3
import sys
import tempfile
from pathlib import Path

from guard.auth.passwords import password_problem
from guard.auth.roles import role_title
from guard.auth.service import INITIAL_ADMIN_FILE
from guard.auth.store import DEMO_USERS, AuthStore, UserNotFound
from guard.pipeline.settings import Settings


def _demo_accounts(db_path: Path) -> list[tuple[str, str]]:
    """(логин, название роли) демо-учёток с паролем, равным логину. Базы ещё нет — все четыре."""
    if not db_path.is_file():
        return [(name, role_title(name)) for name in DEMO_USERS]
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            rows = dict(conn.execute("SELECT username, role FROM users WHERE demo = 1 AND active = 1").fetchall())
    except sqlite3.Error:
        return []
    return [(name, role_title(rows[name])) for name in DEMO_USERS if name in rows]


def banner() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as exc:  # ошибку настройки покажет сам сервер
        print(f"Вход: настройки не прочитаны ({exc})")
        return 0
    if not settings.auth:
        print("Вход выключен (GUARD_AUTH=0): все запросы идут как admin. Для реального объекта включите вход.")
        return 0
    print(f"Вход включён (GUARD_AUTH=1), учётные записи и журнал действий: {settings.auth_db_path}")
    if not settings.demo_login:
        print("Демо-вход выключен (GUARD_DEMO_LOGIN=0): демо-учётки с паролем, равным логину, не входят.")
        initial = settings.auth_db_path.parent / INITIAL_ADMIN_FILE
        if initial.is_file():
            print(f"Пароль первого администратора (admin) — в файле {initial}. Смените пароль и удалите файл.")
        return 0
    accounts = _demo_accounts(settings.auth_db_path)
    if accounts:
        print("Демо-учётки (логин / пароль):")
        for name, title in accounts:
            print(f"  {name} / {name} — {title}")
    else:
        print("Демо-учёток с паролем по умолчанию нет.")
    print("Для реального объекта: GUARD_DEMO_LOGIN=0 и смените пароли "
          "(страница «Пользователи» или python -m guard.auth set-password <логин>).")
    return 0


def _read_password(from_stdin: bool) -> str | None:
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("Новый пароль: ")
    if first != getpass.getpass("Повторите: "):
        print("Пароли не совпадают", file=sys.stderr)
        return None
    return first


def set_password(username: str, from_stdin: bool) -> int:
    password = _read_password(from_stdin)
    if password is None:
        return 2
    problem = password_problem(password)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    db_path = Settings.from_env().auth_db_path
    with AuthStore(db_path) as store:
        try:
            user = store.update_user(username, password=password)
        except UserNotFound:
            print(f"Пользователь {username} не найден в {db_path}", file=sys.stderr)
            return 1
        closed = store.delete_user_sessions(user.username)
        store.add_audit("user.update", username="cli", target=user.username,
                        details={"password_changed": True, "via": "cli", "sessions_closed": closed})
    print(f"Пароль {user.username} изменён, закрыто сессий: {closed}")
    return 0


def route_rows() -> list[tuple[str, str, str]]:
    """(метод, путь, доступ) для каждого маршрута API приложения."""
    from fastapi.routing import iter_route_contexts

    from guard.api.app import create_app
    from guard.auth.deps import route_policy

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        settings = Settings(
            data_dir=root, models_dir=root, cameras_config=root / "cameras.yaml", models_config=root / "models.yaml",
            db_path=root / "guard.db", media_dir=root / "media", frontend_dist=root / "dist",
        )
        app = create_app(settings)
    rows = []
    for rc in iter_route_contexts(app.routes):
        path = rc.path_format
        if not path or not path.startswith(("/api", "/ws", "/health")):
            continue
        methods = ["WS"] if rc.methods is None else sorted(rc.methods - {"HEAD"})
        rows.extend((method, path, route_policy(rc.original_route, method).describe()) for method in methods)
    return sorted(rows, key=lambda r: (r[1], r[0]))


def routes() -> int:
    print("| Метод | Путь | Доступ |")
    print("|---|---|---|")
    for method, path, access in route_rows():
        print(f"| {method} | {path} | {access} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m guard.auth", description="Вход и роли «Зазеркалья»")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("banner", help="состояние входа и демо-учётки")
    pw = sub.add_parser("set-password", help="сменить пароль пользователя")
    pw.add_argument("username")
    pw.add_argument("--password-stdin", action="store_true", help="прочитать пароль из первой строки stdin")
    sub.add_parser("routes", help="таблица «маршрут → право»")
    args = parser.parse_args(argv)
    if args.command == "banner":
        return banner()
    if args.command == "set-password":
        return set_password(args.username, args.password_stdin)
    return routes()


if __name__ == "__main__":
    sys.exit(main())
