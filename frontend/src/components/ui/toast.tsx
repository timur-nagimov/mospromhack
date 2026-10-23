import { CircleAlert, CircleCheck } from 'lucide-react';
import { createStore, useStore } from '../../state/createStore';

interface Toast {
  id: number;
  kind: 'ok' | 'bad';
  text: string;
}

const toastStore = createStore<Toast[]>([]);
let nextId = 1;

/** Короткое уведомление о результате действия (сохранено, ошибка) */
export function toast(text: string, kind: Toast['kind'] = 'ok'): void {
  const id = nextId++;
  toastStore.set((list) => [...list.slice(-3), { id, kind, text }]);
  window.setTimeout(() => toastStore.set((list) => list.filter((t) => t.id !== id)), kind === 'bad' ? 6000 : 3200);
}

export function ToastHost() {
  const items = useStore(toastStore, (s) => s);
  return (
    <div className="toast-stack" aria-live="polite">
      {items.map((t) => (
        <div key={t.id} className={`toast toast--${t.kind}`}>
          {t.kind === 'ok' ? <CircleCheck size={17} /> : <CircleAlert size={17} />}
          {t.text}
        </div>
      ))}
    </div>
  );
}
