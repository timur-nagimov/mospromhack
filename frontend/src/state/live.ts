// Живое состояние: лента событий из WS, счётчик неразобранных, метрики и их история

import type { ConnectionStatus } from '../api/types';
import type { FlywheelSampleBrief } from '../domain/flywheel';
import type { GuardEvent, MetricsSnapshot } from '../domain/types';
import { isCriticalEvent } from '../domain/violations';
import { createStore, useStore } from './createStore';

const FEED_LIMIT = 200;
const HISTORY_LIMIT = 90; // 3 минуты при снимке раз в 2 с
const ALARM_FALLBACK_S = 600; // после разбора тревоги баннер переходит на другую критическую не старше 10 мин
const FLYWHEEL_FRESH_LIMIT = 300;

export interface MetricsPoint {
  /** epoch, мс */
  at: number;
  fps: Record<string, number>;
  totalP95: number | null;
}

export interface LiveState {
  connection: ConnectionStatus;
  feed: GuardEvent[];
  unreviewed: number | null;
  metrics: MetricsSnapshot | null;
  metricsAt: number | null;
  history: MetricsPoint[];
  /** Последняя критическая тревога, ещё не открытая оператором */
  alarm: GuardEvent | null;
  /** id событий, пришедших в этой сессии, — для анимации появления */
  freshIds: ReadonlySet<string>;
  /** Сэмплы маховика из WS flywheel_sample в этой сессии — подсветка в очереди проверки */
  flywheelFresh: ReadonlySet<string>;
  flywheelLast: FlywheelSampleBrief | null;
  /** epoch, мс: когда пришёл последний сэмпл */
  flywheelLastAt: number | null;
}

const initialLiveState = (): LiveState => ({
  connection: 'connecting',
  feed: [],
  unreviewed: null,
  metrics: null,
  metricsAt: null,
  history: [],
  alarm: null,
  freshIds: new Set(),
  flywheelFresh: new Set(),
  flywheelLast: null,
  flywheelLastAt: null,
});

export const liveStore = createStore<LiveState>(initialLiveState());

export const useLive = <T>(selector: (s: LiveState) => T): T => useStore(liveStore, selector);

function sortFeed(items: GuardEvent[]): GuardEvent[] {
  return items.sort((a, b) => b.confirmed_at - a.confirmed_at).slice(0, FEED_LIMIT);
}

export const liveActions = {
  /** Выход или смена пользователя: лента и метрики прежней сессии не переносятся */
  reset() {
    liveStore.set(() => initialLiveState());
  },

  setConnection(connection: ConnectionStatus) {
    liveStore.set((s) => (s.connection === connection ? s : { ...s, connection }));
  },

  /** Начальная загрузка ленты: слияние с тем, что уже пришло по WS */
  seedFeed(items: GuardEvent[]) {
    liveStore.set((s) => {
      const byId = new Map<string, GuardEvent>();
      for (const e of items) byId.set(e.id, e);
      for (const e of s.feed) byId.set(e.id, e);
      return { ...s, feed: sortFeed([...byId.values()]) };
    });
  },

  setUnreviewed(total: number) {
    liveStore.set((s) => ({ ...s, unreviewed: total }));
  },

  /** true — событие новое (не дубль повторной доставки WS) */
  addEvent(event: GuardEvent): boolean {
    if (liveStore.get().feed.some((e) => e.id === event.id)) return false;
    liveStore.set((s) => {
      if (s.feed.some((e) => e.id === event.id)) return s;
      const fresh = new Set(s.freshIds);
      fresh.add(event.id);
      return {
        ...s,
        feed: sortFeed([event, ...s.feed]),
        unreviewed: s.unreviewed == null ? s.unreviewed : s.unreviewed + (event.status === 'new' ? 1 : 0),
        alarm: isCriticalEvent(event) && event.status === 'new' ? event : s.alarm,
        freshIds: fresh,
      };
    });
    return true;
  },

  /**
   * Обновление события (WS event_update или ответ PATCH).
   * false — события не было в ленте, счётчик неразобранных надо сверить с сервером.
   */
  upsertEvent(event: GuardEvent): boolean {
    const known = liveStore.get().feed.some((e) => e.id === event.id);
    liveStore.set((s) => {
      const prev = s.feed.find((e) => e.id === event.id);
      let unreviewed = s.unreviewed;
      if (prev && unreviewed != null) {
        if (prev.status === 'new' && event.status !== 'new') unreviewed = Math.max(0, unreviewed - 1);
        if (prev.status !== 'new' && event.status === 'new') unreviewed += 1;
      }
      const feed = prev ? s.feed.map((e) => (e.id === event.id ? event : e)) : sortFeed([event, ...s.feed]);
      let alarm = s.alarm;
      if (alarm && alarm.id === event.id && event.status !== 'new') {
        // гасим разобранную тревогу, но не теряем другую свежую критическую
        const edge = Date.now() / 1000 - ALARM_FALLBACK_S;
        alarm =
          feed.find((e) => e.id !== event.id && e.status === 'new' && e.ended_at == null && isCriticalEvent(e) && e.confirmed_at >= edge) ?? null;
      }
      return { ...s, feed, unreviewed, alarm };
    });
    return known;
  },

  /** true — сэмпл новый (не повторная доставка WS) */
  addFlywheelSample(sample: FlywheelSampleBrief): boolean {
    if (liveStore.get().flywheelFresh.has(sample.id)) return false;
    liveStore.set((s) => {
      const fresh = new Set(s.flywheelFresh);
      fresh.add(sample.id);
      if (fresh.size > FLYWHEEL_FRESH_LIMIT) fresh.delete(fresh.values().next().value as string);
      return { ...s, flywheelFresh: fresh, flywheelLast: sample, flywheelLastAt: Date.now() };
    });
    return true;
  },

  dismissAlarm() {
    liveStore.set((s) => (s.alarm ? { ...s, alarm: null } : s));
  },

  setMetrics(metrics: MetricsSnapshot) {
    liveStore.set((s) => {
      const fps: Record<string, number> = {};
      let totalP95: number | null = null;
      for (const [id, m] of Object.entries(metrics.cameras ?? {})) {
        fps[id] = m.fps_processed;
        const p95 = m.timings_ms?.total?.p95;
        if (p95 != null) totalP95 = Math.max(totalP95 ?? 0, p95);
      }
      const point: MetricsPoint = { at: Date.now(), fps, totalP95 };
      const history = [...s.history, point].slice(-HISTORY_LIMIT);
      return { ...s, metrics, metricsAt: Date.now(), history };
    });
  },
};

/** Активное нарушение камеры: событие ещё не закрыто и свежее 5 минут */
export function activeEventFor(feed: GuardEvent[], cameraId: string, now: number): GuardEvent | null {
  let best: GuardEvent | null = null;
  for (const e of feed) {
    if (e.camera_id !== cameraId || e.ended_at != null) continue;
    if (now - e.confirmed_at > 300) continue;
    if (!best) best = e;
    else if (isCriticalEvent(e) && !isCriticalEvent(best)) best = e;
  }
  return best;
}
