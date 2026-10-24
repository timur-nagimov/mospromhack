// Роли и права: зеркало контракта docs/ROLES.md. Источник правды — сервер
// (guard/auth/roles.py, GET /api/auth/roles); здесь — подписи для интерфейса
// и запасная матрица на случай, если каталог ролей не загрузился.

import { toEpoch } from './flywheel';

export type Permission =
  | 'analytics.view'
  | 'events.view'
  | 'video.view'
  | 'events.handle'
  | 'cameras.view'
  | 'system.view'
  | 'zones.edit'
  | 'flywheel.review'
  | 'audit.view'
  | 'cameras.manage'
  | 'flywheel.train'
  | 'users.manage';

export type RoleId = 'worker' | 'operator' | 'safety' | 'admin';

export const ROLE_ORDER: RoleId[] = ['worker', 'operator', 'safety', 'admin'];

export interface PermissionMeta {
  title: string;
  /** Что открывает — простыми словами, для матрицы и экрана «Нет доступа» */
  description: string;
  /** Право открывает изображения людей — персональные данные (152-ФЗ) */
  personalData?: boolean;
}

export const PERMISSION_META: Record<Permission, PermissionMeta> = {
  'analytics.view': { title: 'Аналитика', description: 'Сводка нарушений по участкам, типам и сменам, отчёты' },
  'events.view': { title: 'Журнал событий', description: 'Время, тип, камера и статус события — без снимков и видео' },
  'video.view': { title: 'Видео и снимки', description: 'Мониторинг камер вживую, снимки и клипы событий', personalData: true },
  'events.handle': { title: 'Разбор событий', description: 'Подтвердить, отметить ложным, передать мастеру, заметка' },
  'cameras.view': { title: 'Камеры: просмотр', description: 'Список камер, зоны и правила без права менять' },
  'zones.edit': { title: 'Зоны и правила', description: 'Опасные зоны, расписания и параметры правил камер' },
  'cameras.manage': { title: 'Подключение камер', description: 'Добавить и удалить камеру, сменить источник (RTSP и др.)' },
  'flywheel.review': { title: 'Проверка кадров', description: 'Очередь маховика данных: принять или отклонить разметку' },
  'flywheel.train': { title: 'Обучение и выкатка', description: 'Сборка датасета, дообучение, гейт и выкатка моделей' },
  'system.view': { title: 'Производительность', description: 'FPS, задержки по этапам, загрузка ускорителя LinQ' },
  'audit.view': { title: 'Журнал действий', description: 'Кто, когда и что изменил в системе, с какого адреса' },
  'users.manage': { title: 'Пользователи', description: 'Учётные записи, роли, пароли, отключение' },
};

/** Группы прав для матрицы: строки идут по участкам ответственности */
export const PERMISSION_GROUPS: { title: string; permissions: Permission[] }[] = [
  { title: 'Дашборды и журнал', permissions: ['analytics.view', 'events.view'] },
  { title: 'Видео и реакция на нарушение', permissions: ['video.view', 'events.handle'] },
  { title: 'Камеры, зоны и правила', permissions: ['cameras.view', 'zones.edit', 'cameras.manage'] },
  { title: 'Данные и модели', permissions: ['flywheel.review', 'flywheel.train'] },
  { title: 'Система и доступ', permissions: ['system.view', 'audit.view', 'users.manage'] },
];

export const ALL_PERMISSIONS: Permission[] = PERMISSION_GROUPS.flatMap((g) => g.permissions);

export interface RoleMeta {
  title: string;
  /** Кто это на производстве */
  who: string;
  /** Что делает в системе — одной строкой для кнопок входа */
  summary: string;
  description: string;
  /** Уровень допуска I–IV: порядок колонок матрицы и полоски в интерфейсе */
  level: number;
}

export const ROLE_META: Record<RoleId, RoleMeta> = {
  worker: {
    title: 'Сотрудник',
    who: 'рабочий, бригадир',
    summary: 'Дашборды и журнал без снимков',
    description: 'Смотрит дашборды: сводку нарушений по участку, типам и сменам. Видит журнал событий без снимков и видео',
    level: 1,
  },
  operator: {
    title: 'Диспетчер смены',
    who: 'мастер смены, диспетчер',
    summary: 'Камеры вживую и разбор событий',
    description: 'Следит за камерами вживую, разбирает события: подтверждает, отмечает ложные, закрывает, пишет заметки',
    level: 2,
  },
  safety: {
    title: 'Инженер по ОТ',
    who: 'специалист по охране труда',
    summary: 'Плюс зоны и правила, проверка кадров',
    description: 'Всё, что диспетчер, плюс опасные зоны и правила камер, проверка кадров маховика, журнал действий',
    level: 3,
  },
  admin: {
    title: 'Администратор',
    who: 'ИТ, владелец системы',
    summary: 'Плюс камеры, учётные записи, модели',
    description: 'Всё, плюс подключение и удаление камер, учётные записи, обучение и выкатка моделей',
    level: 4,
  },
};

/**
 * Запасная матрица по docs/ROLES.md. Интерфейс берёт права пользователя из
 * /api/auth/me, а матрицу — из /api/auth/roles; эта таблица нужна мок-серверу
 * и подсказкам, если каталог ролей недоступен.
 */
export const ROLE_PERMISSIONS: Record<RoleId, Permission[]> = {
  worker: ['analytics.view', 'events.view'],
  operator: ['analytics.view', 'events.view', 'video.view', 'events.handle', 'cameras.view', 'system.view'],
  safety: [
    'analytics.view',
    'events.view',
    'video.view',
    'events.handle',
    'cameras.view',
    'system.view',
    'zones.edit',
    'flywheel.review',
    'audit.view',
  ],
  admin: [...ALL_PERMISSIONS],
};

export function isRoleId(value: string): value is RoleId {
  return (ROLE_ORDER as string[]).includes(value);
}

export function roleTitle(role: string | null | undefined): string {
  if (!role) return '—';
  return isRoleId(role) ? ROLE_META[role].title : role;
}

export function roleLevel(role: string | null | undefined): number {
  return role && isRoleId(role) ? ROLE_META[role].level : 0;
}

export function permissionTitle(permission: string): string {
  return PERMISSION_META[permission as Permission]?.title ?? permission;
}

/** Текст уведомления о 403: «Недостаточно прав: Зоны и правила (zones.edit)» */
export function forbiddenText(permission: string): string {
  const title = PERMISSION_META[permission as Permission]?.title;
  return title ? `Недостаточно прав: ${title} (${permission})` : `Недостаточно прав: ${permission}`;
}

/** Инициалы для аватара: «Ольга Иванова» → «ОИ», «admin» → «AD» */
export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
  return (words[0] ?? '?').slice(0, 2).toUpperCase();
}

// ---------- контракт API (docs/ROLES.md, раздел «API»)

export interface DemoUser {
  username: string;
  display_name: string;
  role: string;
}

export interface AuthConfig {
  auth_enabled: boolean;
  demo_login: boolean;
  demo_users: DemoUser[];
}

/** Бэкенд без /api/auth/* (старее контракта) или GUARD_AUTH=0 — вход выключен */
export const OPEN_CONFIG: AuthConfig = { auth_enabled: false, demo_login: false, demo_users: [] };

export interface Me {
  username: string;
  display_name: string;
  role: string;
  role_title: string;
  permissions: string[];
}

/** При выключенном входе (GUARD_AUTH=0) все запросы идут как admin — интерфейс тоже */
export const OPEN_ACCESS_USER: Me = {
  username: 'admin',
  display_name: 'Администратор',
  role: 'admin',
  role_title: ROLE_META.admin.title,
  permissions: [...ALL_PERMISSIONS],
};

/**
 * Бэкенд старее контракта (нет /api/auth/*): интерфейс как до ролей — без
 * разделов «Пользователи» и «Журнал действий», их API на таком сервере нет.
 */
export const LEGACY_ACCESS_USER: Me = {
  ...OPEN_ACCESS_USER,
  permissions: ALL_PERMISSIONS.filter((p) => p !== 'users.manage' && p !== 'audit.view'),
};

export interface RoleInfo {
  id: string;
  title: string;
  description: string;
  permissions: string[];
}

export interface PermissionInfo {
  id: string;
  title: string;
  description: string;
}

export interface RolesCatalog {
  roles: RoleInfo[];
  permissions: PermissionInfo[];
}

export interface UserRecord {
  username: string;
  display_name: string;
  role: string;
  active: boolean;
  /** epoch, секунды */
  created_at: number | null;
  last_login_at: number | null;
}

export interface UserCreate {
  username: string;
  display_name: string;
  role: string;
  password: string;
}

export interface UserPatch {
  display_name?: string;
  role?: string;
  password?: string;
  active?: boolean;
}

export interface AuditEntry {
  /** epoch, секунды */
  ts: number;
  username: string | null;
  role: string | null;
  action: string;
  target: string | null;
  details: unknown;
  ip: string | null;
}

export interface AuditPage {
  items: AuditEntry[];
  total: number;
}

export interface AuditQuery {
  limit?: number;
  offset?: number;
  username?: string;
  action?: string;
  from?: number;
  to?: number;
}

// ---------- журнал действий

export type AuditGroup = 'auth' | 'cameras' | 'events' | 'users' | 'flywheel';

export interface AuditActionMeta {
  title: string;
  group: AuditGroup;
  /** Сигнал цветом — только для неудачного входа и удалений */
  tone?: 'bad';
}

export const AUDIT_ACTIONS: Record<string, AuditActionMeta> = {
  'auth.login': { title: 'Вход', group: 'auth' },
  'auth.login_failed': { title: 'Неудачный вход', group: 'auth', tone: 'bad' },
  'auth.logout': { title: 'Выход', group: 'auth' },
  'camera.create': { title: 'Камера добавлена', group: 'cameras' },
  'camera.update': { title: 'Камера изменена', group: 'cameras' },
  'camera.delete': { title: 'Камера удалена', group: 'cameras', tone: 'bad' },
  'zones.update': { title: 'Зоны изменены', group: 'cameras' },
  'rules.update': { title: 'Правила изменены', group: 'cameras' },
  'event.update': { title: 'Решение по событию', group: 'events' },
  'user.create': { title: 'Пользователь создан', group: 'users' },
  'user.update': { title: 'Пользователь изменён', group: 'users' },
  'flywheel.review': { title: 'Проверка кадра', group: 'flywheel' },
  'flywheel.job': { title: 'Задача маховика', group: 'flywheel' },
};

export const AUDIT_GROUP_TITLES: Record<AuditGroup, string> = {
  auth: 'Вход и выход',
  cameras: 'Камеры, зоны и правила',
  events: 'События',
  users: 'Учётные записи',
  flywheel: 'Маховик данных',
};

export function auditActionMeta(action: string): AuditActionMeta {
  const known = AUDIT_ACTIONS[action];
  if (known) return known;
  const prefix = action.split('.')[0];
  const group: AuditGroup =
    prefix === 'auth' ? 'auth' : prefix === 'user' ? 'users' : prefix === 'event' ? 'events' : prefix === 'flywheel' ? 'flywheel' : 'cameras';
  return { title: action, group };
}

// ---------- нормализация ответов: принимаем варианты формы, которые допускает контракт

type Obj = Record<string, unknown>;

const isObj = (value: unknown): value is Obj => typeof value === 'object' && value !== null && !Array.isArray(value);
const str = (value: unknown): string | null => (typeof value === 'string' && value ? value : typeof value === 'number' ? String(value) : null);

function strList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => (isObj(v) ? str(v.id) ?? str(v.name) ?? str(v.permission) : str(v))).filter((v): v is string => Boolean(v));
}

/** Канонический порядок: известные права — по группам, неизвестные — в конце */
function sortPermissions(ids: Iterable<string>): string[] {
  const order = new Map<string, number>(ALL_PERMISSIONS.map((p, i) => [p, i]));
  return [...new Set(ids)].sort((a, b) => (order.get(a) ?? 999) - (order.get(b) ?? 999) || a.localeCompare(b));
}

export function normalizeConfig(raw: unknown): AuthConfig {
  if (!isObj(raw)) return OPEN_CONFIG;
  const demo = Array.isArray(raw.demo_users) ? raw.demo_users : [];
  return {
    auth_enabled: raw.auth_enabled !== false,
    demo_login: Boolean(raw.demo_login),
    demo_users: demo
      .filter(isObj)
      .map((u) => ({ username: str(u.username) ?? '', display_name: str(u.display_name) ?? str(u.username) ?? '', role: str(u.role) ?? '' }))
      .filter((u) => u.username)
      .sort((a, b) => roleLevel(a.role) - roleLevel(b.role)),
  };
}

export function normalizeMe(raw: unknown): Me {
  const o = isObj(raw) ? raw : {};
  const role = str(o.role) ?? '';
  const username = str(o.username) ?? '';
  return {
    username,
    display_name: str(o.display_name) ?? username,
    role,
    role_title: str(o.role_title) ?? roleTitle(role),
    permissions: strList(o.permissions),
  };
}

function toRole(o: Obj, fallbackId?: string): RoleInfo | null {
  const id = str(o.id) ?? str(o.role) ?? str(o.key) ?? fallbackId ?? str(o.name);
  if (!id) return null;
  const known = isRoleId(id) ? ROLE_META[id] : null;
  const name = str(o.name);
  return {
    id,
    title: str(o.title) ?? str(o.role_title) ?? (name && name !== id ? name : null) ?? known?.title ?? id,
    description: str(o.description) ?? known?.description ?? '',
    permissions: sortPermissions(strList(o.permissions)),
  };
}

/**
 * GET /api/auth/roles: массив ролей или {roles, permissions}; роли — массивом
 * или словарём по id; права — строками или объектами {id, title, description}.
 */
export function normalizeRolesCatalog(raw: unknown): RolesCatalog {
  const rolesRaw = Array.isArray(raw) ? raw : isObj(raw) ? (raw.roles ?? raw.items) : null;
  let roles: RoleInfo[] = [];
  if (Array.isArray(rolesRaw)) roles = rolesRaw.filter(isObj).map((r) => toRole(r)).filter((r): r is RoleInfo => r != null);
  else if (isObj(rolesRaw))
    roles = Object.entries(rolesRaw)
      .map(([id, r]) => (isObj(r) ? toRole(r, id) : Array.isArray(r) ? toRole({ permissions: r }, id) : null))
      .filter((r): r is RoleInfo => r != null);
  roles.sort((a, b) => (roleLevel(a.id) || 99) - (roleLevel(b.id) || 99));

  const declared = new Map<string, PermissionInfo>();
  const permsRaw = isObj(raw) ? raw.permissions : null;
  if (Array.isArray(permsRaw)) {
    for (const p of permsRaw) {
      const id = isObj(p) ? str(p.id) ?? str(p.name) ?? str(p.permission) : str(p);
      if (!id) continue;
      declared.set(id, {
        id,
        title: (isObj(p) ? str(p.title) : null) ?? permissionTitle(id),
        description: (isObj(p) ? str(p.description) : null) ?? PERMISSION_META[id as Permission]?.description ?? '',
      });
    }
  } else if (isObj(permsRaw)) {
    for (const [id, p] of Object.entries(permsRaw)) {
      declared.set(id, {
        id,
        title: (isObj(p) ? str(p.title) : str(p)) ?? permissionTitle(id),
        description: (isObj(p) ? str(p.description) : null) ?? PERMISSION_META[id as Permission]?.description ?? '',
      });
    }
  }
  const ids = sortPermissions([...declared.keys(), ...roles.flatMap((r) => r.permissions)]);
  const permissions = ids.map(
    (id) => declared.get(id) ?? { id, title: permissionTitle(id), description: PERMISSION_META[id as Permission]?.description ?? '' },
  );
  return { roles, permissions };
}

/** Каталог по docs/ROLES.md — пока /api/auth/roles не ответил или недоступен */
export function fallbackRolesCatalog(): RolesCatalog {
  return {
    roles: ROLE_ORDER.map((id) => ({ id, title: ROLE_META[id].title, description: ROLE_META[id].description, permissions: [...ROLE_PERMISSIONS[id]] })),
    permissions: ALL_PERMISSIONS.map((id) => ({ id, title: PERMISSION_META[id].title, description: PERMISSION_META[id].description })),
  };
}

export function normalizeUser(raw: unknown): UserRecord {
  const o = isObj(raw) ? raw : {};
  const username = str(o.username) ?? '';
  return {
    username,
    display_name: str(o.display_name) ?? username,
    role: str(o.role) ?? '',
    active: o.active !== false && o.active !== 0,
    created_at: toEpoch(o.created_at as string | number | null | undefined),
    last_login_at: toEpoch(o.last_login_at as string | number | null | undefined),
  };
}

export function normalizeAuditEntry(raw: unknown): AuditEntry {
  const o = isObj(raw) ? raw : {};
  let details: unknown = o.details ?? null;
  if (typeof details === 'string' && /^[[{]/.test(details.trim())) {
    try {
      details = JSON.parse(details);
    } catch {
      // оставляем строкой
    }
  }
  return {
    ts: toEpoch(o.ts as string | number | null | undefined) ?? 0,
    username: str(o.username),
    role: str(o.role),
    action: str(o.action) ?? '',
    target: str(o.target),
    details,
    ip: str(o.ip),
  };
}

export function normalizeAuditPage(raw: unknown): AuditPage {
  if (Array.isArray(raw)) return { items: raw.map(normalizeAuditEntry), total: raw.length };
  const o = isObj(raw) ? raw : {};
  const items = Array.isArray(o.items) ? o.items.map(normalizeAuditEntry) : [];
  const total = typeof o.total === 'number' ? o.total : items.length;
  return { items, total };
}
