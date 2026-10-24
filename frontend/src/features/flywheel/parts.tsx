// Мелкие общие элементы страницы «Данные»: бейджи источника и статусов, запрет обучения

import { CircleCheck, CircleDashed, CircleX, Loader2 } from 'lucide-react';
import {
  jobPhase,
  modelStatusLabel,
  sampleStatusColor,
  sampleStatusLabel,
  sourceColor,
  sourceLabel,
  SOURCE_HINTS,
  type SampleSource,
} from '../../domain/flywheel';
import { createStore, useStore } from '../../state/createStore';

export function SourceTag({ source }: { source: string }) {
  return (
    <span className="badge fw-src" style={{ ['--c' as string]: sourceColor(source) }} title={SOURCE_HINTS[source as SampleSource] ?? source}>
      <span className="badge__mark" aria-hidden="true" />
      {sourceLabel(source)}
    </span>
  );
}

export function SampleStatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge fw-st fw-st--${status}`} style={{ ['--c' as string]: sampleStatusColor(status) }}>
      <span className="badge__mark" aria-hidden="true" />
      {sampleStatusLabel(status)}
    </span>
  );
}

export function HoldoutBadge() {
  return (
    <span className="badge fw-holdout" title="Кадр отложенного ролика: в обучающую выборку не экспортируется">
      отложенная выборка
    </span>
  );
}

const MODEL_STATUS_COLOR: Record<string, string> = {
  deployed: 'var(--st-confirmed)',
  candidate: 'var(--series-1)',
  rejected: 'var(--bad)',
  archived: 'var(--text-3)',
};

export function ModelStatusBadge({ status }: { status: string }) {
  return (
    <span className="badge" style={{ ['--c' as string]: MODEL_STATUS_COLOR[status] ?? 'var(--text-3)' }}>
      <span className="badge__mark" aria-hidden="true" />
      {modelStatusLabel(status)}
    </span>
  );
}

export function JobPhaseIcon({ status, size = 15 }: { status: string; size?: number }) {
  const phase = jobPhase(status);
  if (phase === 'active') return <Loader2 size={size} className="fw-spin fw-phase--active" aria-label="идёт" />;
  if (phase === 'done') return <CircleCheck size={size} className="fw-phase--done" aria-label="готово" />;
  if (phase === 'failed') return <CircleX size={size} className="fw-phase--failed" aria-label="ошибка" />;
  return <CircleDashed size={size} className="fw-phase--unknown" aria-label={status} />;
}

/**
 * Запрет задач train и deploy: сервер отвечает 403, пока не задан
 * GUARD_FLYWHEEL_ALLOW_TRAIN=1. Запоминаем причину до перезагрузки страницы,
 * чтобы кнопки «Дообучить» и «Выкатить» не предлагали заведомо отклонённое действие.
 */
const trainGuardStore = createStore<{ reason: string | null }>({ reason: null });

export const trainGuard = {
  forbid(reason: string) {
    trainGuardStore.set((s) => (s.reason === reason ? s : { reason }));
  },
};

export function useTrainForbidden(allowTrain: boolean | undefined): string | null {
  const reason = useStore(trainGuardStore, (s) => s.reason);
  if (reason) return reason;
  if (allowTrain === false) return 'Задачи train и deploy выключены: нужен GUARD_FLYWHEEL_ALLOW_TRAIN=1';
  return null;
}
