import { weekdayIndex } from '../../domain/format';
import type { EventStatus, GuardEvent, StatsItem } from '../../domain/types';

/** Матрица день недели × час (7×24) */
export function weekdayHourMatrix(events: GuardEvent[]): number[][] {
  const m = Array.from({ length: 7 }, () => new Array<number>(24).fill(0));
  for (const e of events) {
    const d = new Date(e.confirmed_at * 1000);
    m[weekdayIndex(e.confirmed_at)][d.getHours()] += 1;
  }
  return m;
}

export interface StatusSplit {
  type: string;
  counts: Record<EventStatus, number>;
  total: number;
  reviewed: number;
  /** доля ложных среди разобранных */
  falseRate: number | null;
}

export function statusByType(events: GuardEvent[]): StatusSplit[] {
  const map = new Map<string, StatusSplit>();
  for (const e of events) {
    let row = map.get(e.type);
    if (!row) {
      row = { type: e.type, counts: { new: 0, confirmed: 0, forwarded: 0, false: 0 }, total: 0, reviewed: 0, falseRate: null };
      map.set(e.type, row);
    }
    row.counts[e.status] += 1;
    row.total += 1;
  }
  for (const row of map.values()) {
    row.reviewed = row.total - row.counts.new;
    row.falseRate = row.reviewed > 0 ? row.counts.false / row.reviewed : null;
  }
  return [...map.values()].sort((a, b) => (b.falseRate ?? -1) - (a.falseRate ?? -1));
}

export function sumStatuses(rows: StatusSplit[]): Record<EventStatus, number> {
  const acc: Record<EventStatus, number> = { new: 0, confirmed: 0, forwarded: 0, false: 0 };
  for (const r of rows) for (const k of Object.keys(acc) as EventStatus[]) acc[k] += r.counts[k];
  return acc;
}

/** Заполнить пропущенные дни нулями — ось времени без «дыр» */
export function fillDays(items: StatsItem[], from: number | undefined, to: number): { key: string; count: number }[] {
  const byKey = new Map(items.map((i) => [i.key.slice(0, 10), i.count]));
  const keys = [...byKey.keys()].sort();
  const startTs = from ?? (keys[0] ? new Date(`${keys[0]}T00:00:00`).getTime() / 1000 : to);
  const out: { key: string; count: number }[] = [];
  const d = new Date(startTs * 1000);
  d.setHours(0, 0, 0, 0);
  const end = new Date(to * 1000);
  let guard = 0;
  while (d <= end && guard < 400) {
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
    out.push({ key, count: byKey.get(key) ?? 0 });
    d.setDate(d.getDate() + 1);
    guard += 1;
  }
  return out;
}
