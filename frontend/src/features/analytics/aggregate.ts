import { moscowDayKey, moscowStartOfDay } from '../../domain/moscow';
import type { PeriodPreset } from '../../domain/period';
import type { EventStatus, StatsItem } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';

/** Все запросы страницы используют одни и те же включительные границы МСК. */
export function analyticsRange(preset: PeriodPreset, now: number): { from?: number; to: number } {
  const to = Math.floor(now / 60) * 60;
  const today = moscowStartOfDay(to);
  switch (preset) {
    case 'today': return { from: today, to };
    case '7d': return { from: today - 6 * 86400, to };
    case '30d': return { from: today - 29 * 86400, to };
    default: return { to };
  }
}

/** Матрица из полных серверных агрегатов, ключ «день недели:час» в МСК. */
export function weekdayHourMatrix(items: StatsItem[]): number[][] {
  const matrix = Array.from({ length: 7 }, () => new Array<number>(24).fill(0));
  for (const item of items) {
    const [day, hour] = item.key.split(':').map(Number);
    if (Number.isInteger(day) && day >= 0 && day < 7 && Number.isInteger(hour) && hour >= 0 && hour < 24) {
      matrix[day][hour] += item.count;
    }
  }
  return matrix;
}

export interface StatusSplit {
  type: string;
  counts: Record<EventStatus, number>;
  total: number;
  reviewed: number;
  falseRate: number | null;
}

export function statusByType(groups: Record<EventStatus, StatsItem[]>): StatusSplit[] {
  const map = new Map<string, StatusSplit>();
  for (const status of Object.keys(groups) as EventStatus[]) {
    for (const item of groups[status]) {
      let row = map.get(item.key);
      if (!row) {
        row = { type: item.key, counts: { new: 0, confirmed: 0, forwarded: 0, false: 0 }, total: 0, reviewed: 0, falseRate: null };
        map.set(item.key, row);
      }
      row.counts[status] += item.count;
      row.total += item.count;
    }
  }
  for (const row of map.values()) {
    row.reviewed = row.total - row.counts.new;
    row.falseRate = row.reviewed > 0 ? row.counts.false / row.reviewed : null;
  }
  return [...map.values()].sort((a, b) => (b.falseRate ?? -1) - (a.falseRate ?? -1));
}

export function sumStatuses(rows: StatusSplit[]): Record<EventStatus, number> {
  const acc: Record<EventStatus, number> = { new: 0, confirmed: 0, forwarded: 0, false: 0 };
  for (const row of rows) for (const key of Object.keys(acc) as EventStatus[]) acc[key] += row.counts[key];
  return acc;
}

/** Объединение критических типов и важности без двойного счёта пересечения. */
export function criticalCount(types: StatsItem[], criticalTypes: StatsItem[]): number {
  return types.reduce((sum, item) => sum + (getViolationMeta(item.key).critical ? item.count : 0), 0)
    + criticalTypes.reduce((sum, item) => sum + (getViolationMeta(item.key).critical ? 0 : item.count), 0);
}

/** Сервер уже прислал календарные дни МСК; перебираем ключи независимо от TZ браузера. */
export function fillDays(items: StatsItem[], from: number | undefined, to: number): { key: string; count: number }[] {
  const byKey = new Map(items.map((item) => [item.key, item.count]));
  const keys = [...byKey.keys()].sort();
  const start = from == null ? keys[0] ?? moscowDayKey(to) : moscowDayKey(from);
  const end = Date.parse(`${moscowDayKey(to)}T00:00:00Z`);
  const out: { key: string; count: number }[] = [];
  for (let ts = Date.parse(`${start}T00:00:00Z`); ts <= end; ts += 86400000) {
    const key = new Date(ts).toISOString().slice(0, 10);
    out.push({ key, count: byKey.get(key) ?? 0 });
  }
  return out;
}
