import type { Point, Zone } from './types';

const RANGE_RE = /^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]|24):([0-5]\d)$/;

export interface HoursRange {
  from: number; // минуты от полуночи
  to: number;
}

export function parseHoursRange(value: string): HoursRange | null {
  const m = value.trim().match(RANGE_RE);
  if (!m) return null;
  const from = Number(m[1]) * 60 + Number(m[2]);
  const to = Number(m[3]) * 60 + Number(m[4]);
  if (to > 24 * 60 || from === to) return null;
  return { from, to };
}

export function isValidHoursRange(value: string): boolean {
  return parseHoursRange(value) !== null;
}

/** Зона активна, если локальное время попадает в один из интервалов (с переходом через полночь) */
export function isZoneActive(zone: Pick<Zone, 'active_hours'>, ts: number): boolean {
  if (!zone.active_hours || zone.active_hours.length === 0) return true;
  const d = new Date(ts * 1000);
  const minutes = d.getHours() * 60 + d.getMinutes();
  return zone.active_hours.some((raw) => {
    const r = parseHoursRange(raw);
    if (!r) return false;
    return r.from < r.to ? minutes >= r.from && minutes < r.to : minutes >= r.from || minutes < r.to;
  });
}

export function describeHours(zone: Pick<Zone, 'active_hours'>): string {
  if (!zone.active_hours || zone.active_hours.length === 0) return 'Круглосуточно';
  return zone.active_hours.map((r) => r.replace('-', '–')).join(', ');
}

export function polygonCentroid(points: Point[]): Point {
  if (points.length === 0) return [0.5, 0.5];
  const sx = points.reduce((s, p) => s + p[0], 0);
  const sy = points.reduce((s, p) => s + p[1], 0);
  return [sx / points.length, sy / points.length];
}

export function roundPoint([x, y]: Point): Point {
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  return [Math.round(clamp(x) * 10000) / 10000, Math.round(clamp(y) * 10000) / 10000];
}

export function makeId(prefix: string): string {
  return `${prefix}-${Math.random().toString(36).slice(2, 7)}`;
}
