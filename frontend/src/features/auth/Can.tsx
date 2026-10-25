import type { ReactNode } from 'react';
import type { Permission } from '../../domain/auth';
import { usePermission } from '../../state/auth';
import { NoAccess } from './NoAccess';

/**
 * Показывает содержимое только при наличии права. Интерфейс прячет недоступное,
 * а решает всё равно сервер: запрос без права получит 403.
 */
export function Can({ perm, children, fallback = null }: { perm: Permission; children: ReactNode; fallback?: ReactNode }) {
  return <>{usePermission(perm) ? children : fallback}</>;
}

/** Охрана маршрута: без права — экран «Нет доступа» с названием права */
export function RequirePermission({ perm, children }: { perm: Permission; children: ReactNode }) {
  return usePermission(perm) ? <>{children}</> : <NoAccess permission={perm} />;
}
