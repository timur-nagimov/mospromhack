/** Локальный выбор роли для интерфейса; матрица разделов из docs/ROLES.md. */
export const ROLES = {
  worker: { title: 'Сотрудник', shortTitle: 'Сотрудник' },
  operator: { title: 'Диспетчер смены', shortTitle: 'Диспетчер' },
  safety: { title: 'Инженер по ОТ', shortTitle: 'Инженер ОТ' },
  admin: { title: 'Администратор', shortTitle: 'Админ' },
} as const;

export type Role = keyof typeof ROLES;

export type SectionPermission =
  | 'analytics.view'
  | 'events.view'
  | 'video.view'
  | 'cameras.view'
  | 'system.view'
  | 'employees.view'
  | 'flywheel.review';

const WORKER: readonly SectionPermission[] = ['analytics.view', 'events.view', 'employees.view'];
const OPERATOR: readonly SectionPermission[] = [...WORKER, 'video.view', 'cameras.view', 'system.view'];
const SAFETY: readonly SectionPermission[] = [...OPERATOR, 'flywheel.review'];

const PERMISSIONS: Record<Role, readonly SectionPermission[]> = {
  worker: WORKER,
  operator: OPERATOR,
  safety: SAFETY,
  admin: SAFETY,
};

export function isRole(value: unknown): value is Role {
  return typeof value === 'string' && Object.hasOwn(ROLES, value);
}

export function canViewSection(role: Role, permission: SectionPermission): boolean {
  return PERMISSIONS[role].includes(permission);
}

export function roleHome(role: Role): string {
  return canViewSection(role, 'video.view') ? '/' : '/analytics';
}
