import { CircleAlert, CircleCheck, Lock } from 'lucide-react';
import { createStore, useStore } from '../../state/createStore';

interface Toast {
  id: number;
  /** lock — отказ сервера по правам (403) */
  kind: 'ok' | 'bad' | 'lock';
  text: string;
}

const toastStore = createStore<Toast[]>([]);
let nextId = 1;

const ICONS = { ok: CircleCheck, bad: CircleAlert, lock: Lock } as const;

/** Короткое уведомление о результате действия (сохранено, ошибка, нет прав) */
export function toast(text: string, kind: Toast['kind'] = 'ok'): void {
  // Не дублируем то, что уже на экране: после уведомления «Недостаточно прав: …»
  // обработчик ошибки действия добавил бы «Не сохранено: Недостаточно прав: …»
  if (toastStore.get().some((t) => t.text === text || (kind !== 'ok' && t.kind !== 'ok' && text.includes(t.text)))) return;
  const id = nextId++;
  toastStore.set((list) => [...list.slice(-3), { id, kind, text }]);
  window.setTimeout(() => toastStore.set((list) => list.filter((t) => t.id !== id)), kind === 'ok' ? 3200 : 6000);
}

export function ToastHost() {
  const items = useStore(toastStore, (s) => s);
  return (
    <div className="toast-stack" aria-live="polite">
      {items.map((t) => {
        const Icon = ICONS[t.kind];
        return (
          <div key={t.id} className={`toast toast--${t.kind}`}>
            <Icon size={17} />
            {t.text}
          </div>
        );
      })}
    </div>
  );
}
