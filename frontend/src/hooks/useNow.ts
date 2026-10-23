import { useSyncExternalStore } from 'react';

// Общий секундный тикер: один setInterval на всё приложение

let now = Date.now() / 1000;
const listeners = new Set<() => void>();
let timer: number | null = null;

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  if (timer == null) {
    timer = window.setInterval(() => {
      now = Date.now() / 1000;
      for (const l of listeners) l();
    }, 1000);
  }
  return () => {
    listeners.delete(listener);
    if (listeners.size === 0 && timer != null) {
      window.clearInterval(timer);
      timer = null;
    }
  };
}

/** Текущее время (epoch-секунды), обновляется раз в секунду */
export function useNow(): number {
  return useSyncExternalStore(subscribe, () => now);
}
