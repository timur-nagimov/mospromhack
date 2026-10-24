import { isRole, type Role } from '../domain/roles';
import { createStore, useStore } from './createStore';

const KEY = 'guard.role';

function loadRole(): Role {
  try {
    const saved = localStorage.getItem(KEY);
    if (isRole(saved)) return saved;
  } catch {
    // Без localStorage выбор работает до закрытия страницы.
  }
  // Сохраняем полный набор разделов при первом открытии существующего пульта.
  return 'admin';
}

const roleStore = createStore<Role>(loadRole());

roleStore.subscribe(() => {
  try {
    localStorage.setItem(KEY, roleStore.get());
  } catch {
    // Приватный режим — без сохранения.
  }
});

export function useRole(): Role {
  return useStore(roleStore, (role) => role);
}

export function getRole(): Role {
  return roleStore.get();
}

export function selectRole(role: Role): void {
  roleStore.set(() => role);
}
