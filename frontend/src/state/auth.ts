// Состояние входа: конфигурация /api/auth/config, текущий пользователь /api/auth/me
// и его права. Интерфейс только прячет недоступное — решает всегда сервер.

import { useCallback } from 'react';
import type { AuthConfig, Me, Permission } from '../domain/auth';
import { createStore, useStore } from './createStore';

export type AuthStatus = 'loading' | 'error' | 'anonymous' | 'authenticated';

/** Почему пользователь на странице входа: сам вышел или сессия кончилась (401) */
export type SignOutReason = 'logout' | 'expired' | null;

export interface AuthState {
  status: AuthStatus;
  config: AuthConfig | null;
  user: Me | null;
  permissions: ReadonlySet<string>;
  /** Ошибка начальной загрузки (сервер недоступен) */
  error: string | null;
  reason: SignOutReason;
}

const NO_PERMISSIONS: ReadonlySet<string> = new Set();

export const authStore = createStore<AuthState>({
  status: 'loading',
  config: null,
  user: null,
  permissions: NO_PERMISSIONS,
  error: null,
  reason: null,
});

function samePermissions(a: ReadonlySet<string>, b: string[]): boolean {
  return a.size === b.length && b.every((p) => a.has(p));
}

export const authActions = {
  loading() {
    authStore.set((s) => (s.status === 'loading' ? s : { ...s, status: 'loading', error: null }));
  },

  failed(error: string) {
    authStore.set((s) => ({ ...s, status: 'error', error }));
  },

  signedIn(config: AuthConfig, user: Me) {
    authStore.set((s) => ({
      status: 'authenticated',
      config,
      user,
      // тот же набор прав — та же ссылка: подписчики useCan не перерисовываются зря
      permissions: s.user?.username === user.username && samePermissions(s.permissions, user.permissions) ? s.permissions : new Set(user.permissions),
      error: null,
      reason: null,
    }));
  },

  signedOut(config: AuthConfig, reason: SignOutReason) {
    authStore.set(() => ({ status: 'anonymous', config, user: null, permissions: NO_PERMISSIONS, error: null, reason }));
  },
};

/** Проверка права вне React (таймеры, обработчики транспорта) */
export function can(permission: Permission): boolean {
  return authStore.get().permissions.has(permission);
}

export const useAuth = <T>(selector: (s: AuthState) => T): T => useStore(authStore, selector);

/** true — у текущего пользователя есть право */
export function usePermission(permission: Permission): boolean {
  return useStore(authStore, (s) => s.permissions.has(permission));
}

/** Функция проверки прав для компонентов, где проверок несколько */
export function useCan(): (permission: Permission) => boolean {
  const permissions = useStore(authStore, (s) => s.permissions);
  return useCallback((permission: Permission) => permissions.has(permission), [permissions]);
}
