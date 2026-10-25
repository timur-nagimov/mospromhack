// Мок /api/auth/*, /api/users*, /api/audit по контракту docs/ROLES.md: демо-учётки
// (пароль равен логину), сессия в localStorage, проверка прав каждого запроса
// и журнал действий с историей за несколько дней.

import {
  ALL_PERMISSIONS,
  forbiddenText,
  isRoleId,
  OPEN_ACCESS_USER,
  PERMISSION_META,
  ROLE_META,
  ROLE_ORDER,
  ROLE_PERMISSIONS,
  type AuditEntry,
  type AuditQuery,
  type AuthConfig,
  type Me,
  type Permission,
  type RoleId,
  type RolesCatalog,
  type UserCreate,
  type UserPatch,
  type UserRecord,
} from '../../domain/auth';
import { nowSec, startOfDay } from '../../domain/format';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import { authSignals } from '../authSignals';
import { ApiError, type AuditApi, type AuthApi, type UsersApi } from '../types';
import { mulberry32 } from './random';

const SESSION_KEY = 'guard.mock.session';
const AUTH_FLAG_KEY = 'guard.mock.auth';
const DEMO_FLAG_KEY = 'guard.mock.demo';
/** Адрес «этого компьютера» в журнале действий мок-режима */
const CLIENT_IP = '10.20.0.24';
const USERNAME_RE = /^[a-z0-9][a-z0-9._-]{1,31}$/;
const MIN_PASSWORD = 4;

interface MockUser extends UserRecord {
  role: RoleId;
  password: string;
}

export interface MockAuthOptions {
  /** false — вход выключен, все запросы идут как admin (как GUARD_AUTH=0) */
  auth?: boolean;
  demoLogin?: boolean;
}

interface Deps {
  respond: <T>(value: T, min?: number, max?: number) => Promise<T>;
  fail: (status: number, message: string) => Promise<never>;
  cameras: () => CameraConfig[];
  events: () => GuardEvent[];
}

/** localStorage есть не везде: тесты создают мок в Node с минимальным window */
const storage = {
  get(key: string): string | null {
    try {
      return typeof localStorage === 'undefined' ? null : localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string | null): void {
    try {
      if (typeof localStorage === 'undefined') return;
      if (value == null) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      // приватный режим — сессия живёт до перезагрузки
    }
  },
};

/** ?auth=0 / ?demo=0 в адресе мок-режима; выбор запоминается, как ?mock=1 */
function resolveFlag(param: string, key: string, fallback: boolean): boolean {
  let value: string | null = null;
  try {
    value = typeof window !== 'undefined' && window.location ? new URLSearchParams(window.location.search).get(param) : null;
  } catch {
    value = null;
  }
  if (value === '0' || value === '1') {
    storage.set(key, value);
    return value === '1';
  }
  const saved = storage.get(key);
  return saved == null ? fallback : saved === '1';
}

function seedUsers(now: number): MockUser[] {
  const installed = startOfDay(now) - 12 * 86400 + 10 * 3600;
  const user = (username: string, display_name: string, role: RoleId, extra: Partial<MockUser> = {}): MockUser => ({
    username,
    display_name,
    role,
    active: true,
    created_at: installed,
    last_login_at: null,
    password: username,
    ...extra,
  });
  return [
    user('worker', 'Сергей Козлов', 'worker'),
    user('operator', 'Ольга Иванова', 'operator'),
    user('safety', 'Марина Соколова', 'safety'),
    user('admin', 'Дмитрий Орлов', 'admin', { created_at: installed - 600 }),
    user('petrov', 'Андрей Петров', 'operator', { created_at: startOfDay(now) - 2 * 86400 + 11 * 3600 + 44 * 60, password: 'night-shift' }),
    user('sidorov', 'Илья Сидоров', 'worker', { created_at: installed + 3600, active: false, password: 'sidorov-2026' }),
  ];
}

/**
 * История за три дня: смены диспетчеров, решения по событиям, правки зон и
 * правил инженером по ОТ, действия администратора, неудачный вход.
 */
function seedAudit(now: number, users: MockUser[], cameras: CameraConfig[], events: GuardEvent[]): AuditEntry[] {
  const rng = mulberry32(20261024);
  const roleOf = (username: string) => users.find((u) => u.username === username)?.role ?? null;
  const ip: Record<string, string> = {
    operator: '10.20.0.11',
    petrov: '10.20.0.11',
    safety: '10.20.0.15',
    admin: '10.20.0.2',
    worker: '10.20.1.40',
  };
  const out: AuditEntry[] = [];
  const add = (ts: number, username: string | null, action: string, target: string | null, details: unknown = null, from?: string) => {
    if (ts > now - 30) return;
    out.push({ ts, username, role: username ? roleOf(username) : null, action, target, details, ip: from ?? (username ? ip[username] : null) ?? '10.20.0.90' });
  };
  const cam = (id: string) => cameras.find((c) => c.camera_id === id);
  const day0 = startOfDay(now);
  const at = (day: number, h: number, m: number, s = 0) => day0 - day * 86400 + h * 3600 + m * 60 + s;

  for (let day = 2; day >= 0; day -= 1) {
    // дневная смена: вход диспетчера и разбор событий за её время
    add(at(day, 7, 52, 14), 'operator', 'auth.login', 'operator');
    add(at(day, 8, 3, 40), 'worker', 'auth.login', 'worker');
    const shiftEvents = events
      .filter((e) => e.confirmed_at >= at(day, 8, 0) && e.confirmed_at < at(day, 19, 30) && e.status !== 'new')
      .slice(0, 7);
    for (const e of shiftEvents) {
      const details = e.operator_note ? { status: e.status, operator_note: e.operator_note } : { status: e.status };
      add(e.confirmed_at + 40 + Math.floor(rng() * 300), 'operator', 'event.update', e.id, details);
    }
    add(at(day, 9, 10, 5), 'safety', 'auth.login', 'safety');
    for (let i = 0; i < 3; i += 1) {
      add(at(day, 10, 5 + i * 4, 12), 'safety', 'flywheel.review', `fs-${(day * 37 + i * 11 + 1000).toString(36)}`, {
        status: i === 1 ? 'rejected' : 'accepted',
      });
    }
    add(at(day, 19, 58, 2), 'operator', 'auth.logout', 'operator');
    add(at(day, 19, 59, 30), 'petrov', 'auth.login', 'petrov');
  }

  // правки зон и правил — дело инженера по ОТ
  const c1 = cam('cam-01');
  const c2 = cam('cam-02');
  if (c1) add(at(1, 9, 24, 50), 'safety', 'zones.update', c1.camera_id, { zones: c1.zones.map((z) => z.name), active_hours: '07:55-12:00' });
  if (c2) {
    const rule = c2.rules.find((r) => r.type === 'ppe_missing') ?? c2.rules[0];
    if (rule) add(at(2, 9, 31, 7), 'safety', 'rules.update', c2.camera_id, { rule_id: rule.rule_id, min_duration_s: 3 });
  }
  // администратор: учётные записи, камера, обучение
  add(at(2, 11, 40, 3), 'admin', 'auth.login', 'admin');
  add(at(2, 11, 44, 31), 'admin', 'user.create', 'petrov', { role: 'operator', display_name: 'Андрей Петров' });
  const c6 = cam('cam-06');
  if (c6) add(at(2, 12, 2, 19), 'admin', 'camera.create', c6.camera_id, { name: c6.name, source: c6.source.type });
  add(at(1, 11, 41, 55), 'admin', 'auth.login', 'admin');
  add(at(1, 11, 46, 9), 'admin', 'user.update', 'sidorov', { active: false });
  add(at(1, 12, 10, 44), 'admin', 'flywheel.job', 'train', { kind: 'train', model_id: 'fall-v1' });
  // опечатка в пароле с чужого терминала — сервер не уточняет, что неверно
  add(at(1, 13, 20, 8), 'operator', 'auth.login_failed', 'operator', null, '10.20.0.77');
  add(at(0, 0, 47, 31), null, 'auth.login_failed', 'master', null, '10.20.3.12');

  return out.sort((a, b) => b.ts - a.ts);
}

export interface MockAuth {
  auth: AuthApi;
  users: UsersApi;
  audit: AuditApi;
  /** null — можно; иначе ошибка 401 или 403, которую надо вернуть клиенту */
  check(permission?: Permission): ApiError | null;
  /** Отказ: уведомить слушателей (переход на вход, «Недостаточно прав») и вернуть ошибку */
  deny(error: ApiError, path: string): Promise<never>;
  /** Обёртка метода API: сначала вход и право, потом сам ответ */
  guard<A extends unknown[], R>(permission: Permission, path: string, fn: (...args: A) => Promise<R>): (...args: A) => Promise<R>;
  /** Есть ли право у текущей сессии (фильтр сообщений фейкового WS) */
  allows(permission: Permission): boolean;
  /** Запись в журнал действий от имени текущего пользователя */
  log(action: string, target: string | null, details?: unknown): void;
}

export function createAuthMock(deps: Deps, options: MockAuthOptions = {}): MockAuth {
  const { respond, fail } = deps;
  const authEnabled = options.auth ?? resolveFlag('auth', AUTH_FLAG_KEY, true);
  const demoLogin = options.demoLogin ?? resolveFlag('demo', DEMO_FLAG_KEY, true);
  const now = nowSec();
  const users = seedUsers(now);
  let session: string | null = authEnabled ? storage.get(SESSION_KEY) : null;
  const audit = seedAudit(now, users, deps.cameras(), deps.events());

  // последний вход демо-учёток — из истории журнала
  for (const u of users) {
    const last = audit.find((a) => a.action === 'auth.login' && a.username === u.username);
    if (last) u.last_login_at = last.ts;
  }

  const findUser = (username: string) => users.find((u) => u.username === username);

  function current(): MockUser | null {
    if (!session) return null;
    const user = findUser(session);
    return user && user.active ? user : null;
  }

  function permissionsOf(role: string): Permission[] {
    return isRoleId(role) ? ROLE_PERMISSIONS[role] : [];
  }

  function meOf(user: MockUser): Me {
    return {
      username: user.username,
      display_name: user.display_name,
      role: user.role,
      role_title: ROLE_META[user.role].title,
      permissions: [...permissionsOf(user.role)],
    };
  }

  function record(user: MockUser): UserRecord {
    return {
      username: user.username,
      display_name: user.display_name,
      role: user.role,
      active: user.active,
      created_at: user.created_at,
      last_login_at: user.last_login_at,
    };
  }

  function setSession(username: string | null) {
    session = username;
    storage.set(SESSION_KEY, username);
  }

  function log(action: string, target: string | null, details: unknown = null, actor?: { username: string | null; role: string | null }) {
    const user = authEnabled ? current() : null;
    audit.unshift({
      ts: nowSec(),
      username: actor ? actor.username : user?.username ?? (authEnabled ? null : 'admin'),
      role: actor ? actor.role : user?.role ?? (authEnabled ? null : 'admin'),
      action,
      target,
      details,
      ip: CLIENT_IP,
    });
  }

  function check(permission?: Permission): ApiError | null {
    if (!authEnabled) return null;
    const user = current();
    if (!user) return new ApiError(401, 'auth required');
    if (permission && !permissionsOf(user.role).includes(permission)) return new ApiError(403, forbiddenText(permission), permission);
    return null;
  }

  function deny(error: ApiError, path: string): Promise<never> {
    return new Promise((_, reject) =>
      window.setTimeout(() => {
        authSignals.report(error, path);
        reject(error);
      }, 90),
    );
  }

  /** Обёртка метода API: сначала вход и право, потом сам ответ */
  function guarded<A extends unknown[], R>(permission: Permission, path: string, fn: (...args: A) => Promise<R>) {
    return (...args: A): Promise<R> => {
      const error = check(permission);
      return error ? deny(error, path) : fn(...args);
    };
  }

  const activeAdmins = () => users.filter((u) => u.role === 'admin' && u.active);

  const config = (): AuthConfig => ({
    auth_enabled: authEnabled,
    demo_login: authEnabled && demoLogin,
    demo_users:
      authEnabled && demoLogin
        ? ROLE_ORDER.map((role) => findUser(role)).filter((u): u is MockUser => Boolean(u)).map((u) => ({ username: u.username, display_name: u.display_name, role: u.role }))
        : [],
  });

  const catalog = (): RolesCatalog => ({
    roles: ROLE_ORDER.map((id) => ({ id, title: ROLE_META[id].title, description: ROLE_META[id].description, permissions: [...ROLE_PERMISSIONS[id]] })),
    permissions: ALL_PERMISSIONS.map((id) => ({ id, title: PERMISSION_META[id].title, description: PERMISSION_META[id].description })),
  });

  const auth: AuthApi = {
    config: () => respond(config(), 60, 140),
    login: (username, password) => {
      const name = username.trim().toLowerCase();
      if (!authEnabled) return respond(OPEN_ACCESS_USER);
      const user = findUser(name);
      if (!user || !user.active || user.password !== password) {
        log('auth.login_failed', name || null, null, { username: user ? name : null, role: user?.role ?? null });
        // неверный логин и неверный пароль неотличимы — как на сервере
        return deny(new ApiError(401, 'Неверный логин или пароль'), '/api/auth/login');
      }
      setSession(user.username);
      user.last_login_at = nowSec();
      log('auth.login', user.username);
      return respond(meOf(user), 150, 320);
    },
    logout: () => {
      const error = check();
      if (error) return deny(error, '/api/auth/logout');
      log('auth.logout', session);
      setSession(null);
      return respond(undefined as void, 60, 140);
    },
    me: () => {
      if (!authEnabled) return respond(OPEN_ACCESS_USER);
      const error = check();
      if (error) return deny(error, '/api/auth/me');
      return respond(meOf(current() as MockUser), 60, 140);
    },
    roles: guardedAny('/api/auth/roles', () => respond(catalog())),
  };

  /** Только вход, без конкретного права */
  function guardedAny<R>(path: string, fn: () => Promise<R>) {
    return (): Promise<R> => {
      const error = check();
      return error ? deny(error, path) : fn();
    };
  }

  const usersApi: UsersApi = {
    list: guarded('users.manage', '/api/users', () => respond(users.map(record))),
    create: guarded('users.manage', '/api/users', (body: UserCreate) => {
      const username = body.username.trim().toLowerCase();
      if (!USERNAME_RE.test(username)) return fail(422, 'username: латиница, цифры, «.», «-», «_», от 2 до 32 символов');
      if (!body.display_name.trim()) return fail(422, 'display_name: обязательное поле');
      if (!isRoleId(body.role)) return fail(422, `role: неизвестная роль ${body.role}`);
      if (body.password.length < MIN_PASSWORD) return fail(422, `password: не короче ${MIN_PASSWORD} символов`);
      if (findUser(username)) return fail(409, `Пользователь ${username} уже существует`);
      const user: MockUser = {
        username,
        display_name: body.display_name.trim(),
        role: body.role,
        active: true,
        created_at: nowSec(),
        last_login_at: null,
        password: body.password,
      };
      users.push(user);
      log('user.create', username, { role: user.role, display_name: user.display_name });
      return respond(record(user), 150, 300);
    }),
    update: guarded('users.manage', '/api/users', (username: string, patch: UserPatch) => {
      const user = findUser(username);
      if (!user) return fail(404, `Пользователь ${username} не найден`);
      if (patch.role !== undefined && !isRoleId(patch.role)) return fail(422, `role: неизвестная роль ${patch.role}`);
      if (patch.display_name !== undefined && !patch.display_name.trim()) return fail(422, 'display_name: обязательное поле');
      if (patch.password !== undefined && patch.password.length < MIN_PASSWORD) return fail(422, `password: не короче ${MIN_PASSWORD} символов`);
      const demotes = (patch.role !== undefined && patch.role !== 'admin') || patch.active === false;
      if (user.role === 'admin' && user.active && demotes && activeAdmins().length <= 1) {
        return fail(409, 'Нельзя понизить или отключить последнего активного администратора');
      }
      const details: Record<string, unknown> = {};
      if (patch.display_name !== undefined && patch.display_name.trim() !== user.display_name) {
        user.display_name = patch.display_name.trim();
        details.display_name = user.display_name;
      }
      if (patch.role !== undefined && patch.role !== user.role) {
        details.role = patch.role;
        user.role = patch.role as RoleId;
      }
      if (patch.active !== undefined && patch.active !== user.active) {
        user.active = patch.active;
        details.active = patch.active;
      }
      if (patch.password !== undefined) {
        user.password = patch.password;
        details.password = 'изменён';
      }
      log('user.update', username, details);
      // отключение и смена пароля закрывают сессии пользователя
      if (session === username && (patch.active === false || patch.password !== undefined)) setSession(null);
      return respond(record(user), 150, 300);
    }),
  };

  const auditApi: AuditApi = {
    list: guarded('audit.view', '/api/audit', (q: AuditQuery) => {
      const filtered = audit.filter(
        (a) =>
          (!q.username || a.username === q.username) &&
          (!q.action || a.action === q.action) &&
          (q.from == null || a.ts >= q.from) &&
          (q.to == null || a.ts <= q.to),
      );
      const offset = q.offset ?? 0;
      const limit = q.limit ?? 50;
      return respond({ items: filtered.slice(offset, offset + limit), total: filtered.length });
    }),
  };

  return {
    auth,
    users: usersApi,
    audit: auditApi,
    check,
    deny,
    guard: guarded,
    allows: (permission) => check(permission) == null,
    log: (action, target, details) => log(action, target, details),
  };
}
