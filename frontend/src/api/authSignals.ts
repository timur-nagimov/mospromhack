import type { ApiError } from './types';

/**
 * Сигналы транспорта о доступе: HTTP-клиент и мок-сервер сообщают сюда о
 * каждом 401 и 403, а AuthProvider реагирует — переход на вход или уведомление.
 * Слой api не зависит от React: только подписка на события.
 */
export interface AuthSignalListener {
  /** 401 от любого запроса, кроме самой попытки входа */
  unauthorized?(): void;
  /** 403 проверки прав: сервер назвал недостающее право */
  forbidden?(permission: string): void;
}

const LOGIN_PATH = '/api/auth/login';
const listeners = new Set<AuthSignalListener>();

export const authSignals = {
  listen(listener: AuthSignalListener): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },

  /** path — путь запроса: неудачный вход (401 на /api/auth/login) обрабатывает форма */
  report(error: ApiError, path: string): void {
    if (error.status === 401 && !path.startsWith(LOGIN_PATH)) {
      for (const l of listeners) l.unauthorized?.();
    } else if (error.status === 403 && error.permission) {
      const permission = error.permission;
      for (const l of listeners) l.forbidden?.(permission);
    }
  },
};
