// Защита несохранённых правок камеры при переходах внутри приложения
// (beforeunload ловит только закрытие вкладки; HashRouter без data router не даёт useBlocker)

let unsaved = false;

export function setUnsaved(value: boolean): void {
  unsaved = value;
}

/** true — можно уходить */
export function confirmLeave(): boolean {
  if (!unsaved) return true;
  const ok = window.confirm('Есть несохранённые изменения камеры. Уйти без сохранения?');
  if (ok) unsaved = false;
  return ok;
}
