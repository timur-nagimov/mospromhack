import { createStore, useStore } from './createStore';

// Пользовательские настройки пульта, сохраняются в localStorage

export interface Settings {
  sound: boolean;
}

const KEY = 'guard.settings';

function load(): Settings {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { sound: true, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    // повреждённые настройки — берём значения по умолчанию
  }
  return { sound: true };
}

export const settingsStore = createStore<Settings>(load());

settingsStore.subscribe(() => {
  try {
    localStorage.setItem(KEY, JSON.stringify(settingsStore.get()));
  } catch {
    // приватный режим — без сохранения
  }
});

export const useSettings = <T>(selector: (s: Settings) => T): T => useStore(settingsStore, selector);

export function toggleSound(): void {
  settingsStore.set((s) => ({ ...s, sound: !s.sound }));
}
