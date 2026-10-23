import { useSyncExternalStore } from 'react';

/** Минимальный внешний стор для useSyncExternalStore (без зависимостей) */
export interface Store<S> {
  get(): S;
  set(updater: (state: S) => S): void;
  subscribe(listener: () => void): () => void;
}

export function createStore<S>(initial: S): Store<S> {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set(updater) {
      const next = updater(state);
      if (next === state) return;
      state = next;
      for (const l of listeners) l();
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

/** Подписка на срез стора; селектор должен возвращать стабильные ссылки */
export function useStore<S, T>(store: Store<S>, selector: (state: S) => T): T {
  return useSyncExternalStore(store.subscribe, () => selector(store.get()));
}
