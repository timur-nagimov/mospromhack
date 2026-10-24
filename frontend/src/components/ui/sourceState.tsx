import type { SourceBrief, SourceState } from '../../domain/types';

/** Состояние источника камеры: значок на плитке мониторинга и в списке камер (docs/STREAMS.md) */
export const SOURCE_STATE_META: Record<SourceState, { label: string; tone: 'ok' | 'warn' | 'bad' }> = {
  online: { label: 'В сети', tone: 'ok' },
  connecting: { label: 'Подключение', tone: 'warn' },
  reconnecting: { label: 'Переподключение', tone: 'warn' },
  offline: { label: 'Нет сигнала', tone: 'bad' },
  error: { label: 'Ошибка источника', tone: 'bad' },
};

export interface SourceView {
  state: SourceState | 'disabled' | 'unknown';
  tone: 'ok' | 'warn' | 'bad' | 'off';
  label: string;
  /** подсказка: к/с, разрешение, последний кадр, переподключения, последняя ошибка */
  title: string;
}

function ageText(s: number | null): string | null {
  if (s == null) return null;
  if (s < 1) return 'последний кадр только что';
  if (s < 90) return `последний кадр ${Math.round(s)} с назад`;
  return `последний кадр ${Math.round(s / 60)} мин назад`;
}

/**
 * Что показать по камере. source — из метрик (WS); без него (старый бэкенд, метрик ещё нет)
 * решаем по fallbackOnline, как раньше: идут ли обработанные кадры.
 */
export function sourceView(enabled: boolean, source: SourceBrief | undefined, fallbackOnline: boolean): SourceView {
  if (!enabled) return { state: 'disabled', tone: 'off', label: 'Выключена', title: 'Камера выключена' };
  if (!source) {
    return fallbackOnline
      ? { state: 'unknown', tone: 'ok', label: 'В сети', title: 'Поток обрабатывается' }
      : { state: 'unknown', tone: 'bad', label: 'Нет данных', title: 'Нет данных' };
  }
  const meta = SOURCE_STATE_META[source.state] ?? SOURCE_STATE_META.error;
  const lines = [meta.label];
  const details = [
    source.fps_in > 0 ? `${source.fps_in.toFixed(1)} к/с` : null,
    source.width && source.height ? `${source.width}×${source.height}` : null,
    ageText(source.last_frame_age_s),
  ].filter(Boolean);
  if (details.length) lines.push(details.join(' · '));
  if (source.drops || source.reconnects) lines.push(`обрывов ${source.drops}, переподключений ${source.reconnects}`);
  if (source.last_error && source.state !== 'online') lines.push(`Последняя ошибка: ${source.last_error}`);
  else if (source.last_error) lines.push(`Была ошибка: ${source.last_error}`);
  return { state: source.state, tone: meta.tone, label: meta.label, title: lines.join('\n') };
}

const DOT_CLASS: Record<SourceView['tone'], string> = {
  ok: 'live-dot',
  warn: 'live-dot live-dot--warn',
  bad: 'live-dot live-dot--bad',
  off: 'live-dot live-dot--off',
};

export function SourceStateDot({ view }: { view: SourceView }) {
  return <span className={DOT_CLASS[view.tone]} title={view.title} aria-label={view.label} role="img" />;
}

/** Подпись состояния, когда поток не в сети: «Переподключение», «Нет сигнала» */
export function SourceStateChip({ view }: { view: SourceView }) {
  if (view.tone === 'ok') return null;
  return (
    <span className={`source-chip source-chip--${view.tone}`} title={view.title}>
      {view.label}
    </span>
  );
}
