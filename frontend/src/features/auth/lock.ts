import { useCallback } from 'react';
import { useRolesCatalog } from '../../api/authQueries';
import { needRoleText, type Permission } from '../../domain/auth';
import { useCan } from '../../state/auth';

/**
 * Для кнопок, которые лучше показать неактивными, чем спрятать:
 * lock(perm) → null, если право есть, иначе подсказка «Нужна роль: …».
 */
export function useLock(): (permission: Permission) => string | null {
  const can = useCan();
  const { catalog } = useRolesCatalog();
  return useCallback((permission: Permission) => (can(permission) ? null : needRoleText(permission, catalog.roles)), [can, catalog]);
}

/** Действия ролей простыми словами — для карточки «Что может эта роль» */
export const ROLE_ACTIONS: { permission: Permission; text: string }[] = [
  { permission: 'video.view', text: 'Смотреть видео камер, снимки и клипы событий' },
  { permission: 'events.handle', text: 'Разбирать события: подтвердить, ложное, передать мастеру' },
  { permission: 'events.assign', text: 'Указывать виновников инцидента' },
  { permission: 'zones.edit', text: 'Рисовать опасные зоны, менять расписания и правила' },
  { permission: 'cameras.manage', text: 'Подключать, проверять и удалять камеры' },
  { permission: 'flywheel.review', text: 'Проверять кадры для дообучения моделей' },
  { permission: 'flywheel.train', text: 'Собирать датасеты, обучать и выкатывать модели' },
  { permission: 'employees.manage', text: 'Добавлять сотрудников в справочник' },
  { permission: 'users.manage', text: 'Создавать учётные записи, менять роли и пароли' },
  { permission: 'audit.view', text: 'Читать журнал действий: кто и что менял' },
];
