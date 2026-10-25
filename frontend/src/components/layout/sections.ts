import { ChartColumn, Cctv, DatabaseZap, FileClock, Gauge, LayoutGrid, ScrollText, UserCog, UsersRound, type LucideIcon } from 'lucide-react';
import type { Permission } from '../../domain/auth';

/** Раздел интерфейса: пункт меню, заголовок шапки и право, которое его открывает */
export interface Section {
  path: string;
  /** Подпись в колонне навигации (мягкий перенос — для узкой колонны) */
  label: string;
  title: string;
  sub: string;
  icon: LucideIcon;
  permission: Permission;
  /** Административный раздел — отдельной группой внизу колонны */
  admin?: boolean;
}

export const SECTIONS: Section[] = [
  {
    path: '/',
    label: 'Мониторинг',
    title: 'Мониторинг',
    sub: 'Камеры и лента тревог в реальном времени',
    icon: LayoutGrid,
    permission: 'video.view',
  },
  {
    path: '/journal',
    label: 'Журнал',
    title: 'Журнал событий',
    sub: 'Доказательная база службы охраны труда',
    icon: ScrollText,
    permission: 'events.view',
  },
  {
    path: '/analytics',
    label: 'Аналитика',
    title: 'Аналитика',
    sub: 'Нарушения по типам, участкам и времени',
    icon: ChartColumn,
    permission: 'analytics.view',
  },
  {
    path: '/cameras',
    label: 'Камеры и правила',
    title: 'Камеры и правила',
    sub: 'Зоны, расписания и правила без кода',
    icon: Cctv,
    permission: 'cameras.view',
  },
  {
    path: '/performance',
    label: 'Производи­тельность',
    title: 'Производительность',
    sub: 'FPS, задержки по этапам и загрузка ускорителя',
    icon: Gauge,
    permission: 'system.view',
  },
  {
    path: '/data',
    label: 'Данные',
    title: 'Данные',
    sub: 'Маховик данных: сбор, проверка, датасеты и дообучение моделей',
    icon: DatabaseZap,
    permission: 'flywheel.review',
  },
  {
    path: '/employees',
    label: 'Сотрудники',
    title: 'Сотрудники',
    sub: 'Полный справочник сотрудников предприятия',
    icon: UsersRound,
    permission: 'employees.view',
  },
  {
    path: '/users',
    label: 'Пользо­ватели',
    title: 'Пользователи',
    sub: 'Учётные записи, роли и права доступа',
    icon: UserCog,
    permission: 'users.manage',
    admin: true,
  },
  {
    path: '/audit',
    label: 'Журнал действий',
    title: 'Журнал действий',
    sub: 'Кто, когда и что изменил в системе',
    icon: FileClock,
    permission: 'audit.view',
    admin: true,
  },
];

/** Раздел по адресу: первый сегмент пути (/cameras/cam-01 → «Камеры и правила») */
export function sectionFor(pathname: string): Section | undefined {
  const head = `/${pathname.split('/')[1] ?? ''}`;
  return SECTIONS.find((s) => s.path === head);
}

/**
 * Стартовая страница — первая доступная: Мониторинг (video.view), иначе Аналитика,
 * иначе любой открытый раздел. null — ни одного раздела (роль без прав).
 */
export function startPath(has: (permission: Permission) => boolean): string | null {
  const preferred = ['/', '/analytics'];
  for (const path of preferred) {
    const section = SECTIONS.find((s) => s.path === path);
    if (section && has(section.permission)) return path;
  }
  return SECTIONS.find((s) => has(s.permission))?.path ?? null;
}

/** Куда вернуться после входа (?next=): только внутренний путь приложения */
export function safeNextPath(value: string | null): string | null {
  if (!value || !value.startsWith('/') || value.startsWith('//') || value.startsWith('/login')) return null;
  return value;
}
