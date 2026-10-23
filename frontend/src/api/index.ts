import { createHttpApi } from './http';
import type { GuardApi } from './types';

export type { GuardApi } from './types';

let instance: GuardApi | null = null;

/**
 * Режим мок-данных включается:
 *  - при сборке/запуске с VITE_MOCK=1 (npm run dev:mock);
 *  - в любой сборке параметром ?mock=1 (запоминается до ?mock=0) — запасной
 *    вариант для демо без бэкенда.
 */
export function resolveMockMode(): boolean {
  const params = new URLSearchParams(window.location.search);
  const flag = params.get('mock');
  if (flag === '1') {
    localStorage.setItem('guard.mock', '1');
    return true;
  }
  if (flag === '0') {
    // явный параметр в адресе сильнее режима сборки
    localStorage.removeItem('guard.mock');
    return false;
  }
  if (import.meta.env.VITE_MOCK === '1') return true;
  return localStorage.getItem('guard.mock') === '1';
}

/** Инициализация до первого рендера; мок грузится отдельным чанком */
export async function initApi(): Promise<GuardApi> {
  if (instance) return instance;
  if (resolveMockMode()) {
    const { createMockApi } = await import('./mock/server');
    instance = createMockApi();
  } else {
    instance = createHttpApi();
  }
  return instance;
}

export function api(): GuardApi {
  if (!instance) throw new Error('API не инициализирован: вызовите initApi() до рендера');
  return instance;
}
