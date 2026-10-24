import { useQueryClient, type QueryClient } from '@tanstack/react-query';
import { createContext, useContext, useEffect, useMemo, useRef, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../../api';
import { authSignals } from '../../api/authSignals';
import { ApiError } from '../../api/types';
import { toast } from '../../components/ui/toast';
import { forbiddenText, LEGACY_ACCESS_USER, OPEN_ACCESS_USER, OPEN_CONFIG, roleTitle, type AuthConfig, type Me } from '../../domain/auth';
import { sectionFor, startPath } from '../../components/layout/sections';
import { authActions, authStore, useAuth } from '../../state/auth';
import { liveActions } from '../../state/live';

const RETRY_MS = 5000;
/** Одно и то же «Недостаточно прав» не чаще раза в 4 с */
const FORBIDDEN_TOAST_GAP_MS = 4000;
/** После 403 сверяем права с сервером: роль могли сменить, пока страница открыта */
const ME_REFRESH_DELAY_MS = 1200;

export interface AuthControls {
  /** Вход по логину и паролю; ошибка (401 и др.) — исключением, форма покажет её сама */
  login(username: string, password: string): Promise<Me>;
  logout(): Promise<void>;
  /** Демо: выйти и войти демо-учёткой (пароль равен логину) */
  switchUser(username: string): Promise<boolean>;
  retry(): void;
}

const AuthControlsContext = createContext<AuthControls | null>(null);

/** Без AuthProvider (тестовые стенды отдельных компонентов) действия входа недоступны */
const DETACHED: AuthControls = {
  login: () => Promise.reject(new Error('Вход недоступен: нет AuthProvider')),
  logout: async () => undefined,
  switchUser: async () => false,
  retry: () => undefined,
};

export function useAuthControls(): AuthControls {
  return useContext(AuthControlsContext) ?? DETACHED;
}

/** Текущий маршрут HashRouter: «#/cameras/cam-01?x=1» → «/cameras/cam-01» */
function currentRoute(): string {
  return window.location.hash.replace(/^#/, '').split('?')[0] || '/';
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Всё, что загружено под прежним пользователем, — в мусор: кэш запросов и живая лента */
function resetClientState(qc: QueryClient): void {
  qc.clear();
  liveActions.reset();
}

/** /api/auth/config, затем /api/auth/me. Бэкенд без /api/auth (404) — вход выключен, роль admin */
async function loadSession(): Promise<void> {
  authActions.loading();
  let config: AuthConfig;
  try {
    config = await api().auth.config();
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) authActions.signedIn(OPEN_CONFIG, LEGACY_ACCESS_USER);
    else authActions.failed(message(err));
    return;
  }
  if (!config.auth_enabled) {
    authActions.signedIn(config, OPEN_ACCESS_USER);
    return;
  }
  try {
    authActions.signedIn(config, await api().auth.me());
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) authActions.signedOut(config, null);
    else authActions.failed(message(err));
  }
}

/**
 * Сессия пользователя: загрузка при старте, реакция на 401 и 403 любого запроса,
 * вход, выход и демо-смена роли. Права живут в authStore (state/auth.ts).
 */
export function AuthProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const navigate = useNavigate();
  const status = useAuth((s) => s.status);
  // Пока идёт смена роли, 401 от фоновых запросов прежней сессии — ожидаемы
  const switching = useRef(false);
  const lastForbidden = useRef(new Map<string, number>());
  const meTimer = useRef<number | null>(null);

  useEffect(() => {
    void loadSession();
  }, []);

  // Сервер недоступен при старте — пробуем снова, пока не ответит
  useEffect(() => {
    if (status !== 'error') return;
    const timer = window.setTimeout(() => void loadSession(), RETRY_MS);
    return () => window.clearTimeout(timer);
  }, [status]);

  useEffect(() => {
    const refreshMe = () => {
      if (meTimer.current != null) return;
      meTimer.current = window.setTimeout(async () => {
        meTimer.current = null;
        const { status: current, config } = authStore.get();
        if (current !== 'authenticated' || !config?.auth_enabled) return;
        try {
          authActions.signedIn(config, await api().auth.me());
        } catch {
          // 401 обработает сигнал unauthorized
        }
      }, ME_REFRESH_DELAY_MS);
    };

    const unlisten = authSignals.listen({
      unauthorized: () => {
        const { status: current, config } = authStore.get();
        if (switching.current || current !== 'authenticated' || !config?.auth_enabled) return;
        resetClientState(qc);
        authActions.signedOut(config, 'expired');
      },
      forbidden: (permission) => {
        const now = Date.now();
        const last = lastForbidden.current.get(permission) ?? 0;
        if (now - last > FORBIDDEN_TOAST_GAP_MS) {
          lastForbidden.current.set(permission, now);
          toast(forbiddenText(permission), 'lock');
        }
        refreshMe();
      },
    });
    return () => {
      unlisten();
      if (meTimer.current != null) window.clearTimeout(meTimer.current);
      meTimer.current = null;
    };
  }, [qc]);

  const controls = useMemo<AuthControls>(() => {
    const configOrOpen = () => authStore.get().config ?? OPEN_CONFIG;

    const login = async (username: string, password: string) => {
      const me = await api().auth.login(username, password);
      resetClientState(qc);
      authActions.signedIn(configOrOpen(), me);
      return me;
    };

    const logout = async () => {
      try {
        await api().auth.logout();
      } catch (err) {
        // сессия могла уже истечь (401) — локально выходим в любом случае
        if (!(err instanceof ApiError && err.status === 401)) toast(`Выход на сервере не подтверждён: ${message(err)}`, 'bad');
      }
      resetClientState(qc);
      authActions.signedOut(configOrOpen(), 'logout');
      navigate('/login', { replace: true });
    };

    const switchUser = async (username: string) => {
      switching.current = true;
      try {
        try {
          await api().auth.logout();
        } catch {
          // прежняя сессия уже закрыта — неважно
        }
        try {
          const me = await api().auth.login(username, username);
          resetClientState(qc);
          authActions.signedIn(configOrOpen(), me);
          // остаёмся в разделе, если новой роли он открыт, иначе — её стартовая страница
          const has = (p: string) => me.permissions.includes(p);
          const section = sectionFor(currentRoute());
          if (!section || !has(section.permission)) navigate(startPath(has) ?? '/', { replace: true });
          toast(`Вы вошли как ${me.role_title || roleTitle(me.role)}: ${me.display_name}`);
          return true;
        } catch (err) {
          resetClientState(qc);
          authActions.signedOut(configOrOpen(), 'logout');
          navigate('/login', { replace: true });
          toast(`Не удалось войти как ${username}: ${message(err)}`, 'bad');
          return false;
        }
      } finally {
        switching.current = false;
      }
    };

    return { login, logout, switchUser, retry: () => void loadSession() };
  }, [qc, navigate]);

  return <AuthControlsContext.Provider value={controls}>{children}</AuthControlsContext.Provider>;
}
