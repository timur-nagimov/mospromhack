// Форматирование для русского интерфейса. Время событий — epoch в секундах.

const timeFmt = new Intl.DateTimeFormat('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
const hmFmt = new Intl.DateTimeFormat('ru-RU', { hour: '2-digit', minute: '2-digit' });
const dateFmt = new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit' });
const fullDateFmt = new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });
const longDateFmt = new Intl.DateTimeFormat('ru-RU', { weekday: 'short', day: 'numeric', month: 'long' });
const weekdayFmt = new Intl.DateTimeFormat('ru-RU', { weekday: 'short' });

export const nowSec = (): number => Date.now() / 1000;

export function formatTime(ts: number): string {
  return timeFmt.format(ts * 1000);
}

export function formatHM(ts: number): string {
  return hmFmt.format(ts * 1000);
}

export function formatDate(ts: number): string {
  return dateFmt.format(ts * 1000);
}

export function formatFullDate(ts: number): string {
  return fullDateFmt.format(ts * 1000);
}

export function formatLongDate(ts: number): string {
  return longDateFmt.format(ts * 1000);
}

export function formatWeekday(ts: number): string {
  return weekdayFmt.format(ts * 1000);
}

export function formatDateTime(ts: number): string {
  return `${formatFullDate(ts)} ${formatTime(ts)}`;
}

/** «Сегодня 14:05:12», «Вчера 09:12:40» или полная дата */
export function formatSmartDateTime(ts: number, now = nowSec()): string {
  const d = new Date(ts * 1000);
  const n = new Date(now * 1000);
  const sameDay = d.toDateString() === n.toDateString();
  if (sameDay) return `Сегодня ${formatTime(ts)}`;
  const y = new Date(n);
  y.setDate(n.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `Вчера ${formatTime(ts)}`;
  return formatDateTime(ts);
}

export function plural(n: number, forms: [string, string, string]): string {
  const abs = Math.abs(n) % 100;
  const last = abs % 10;
  if (abs > 10 && abs < 20) return forms[2];
  if (last > 1 && last < 5) return forms[1];
  if (last === 1) return forms[0];
  return forms[2];
}

export function formatRelative(ts: number, now = nowSec()): string {
  const diff = Math.max(0, now - ts);
  if (diff < 5) return 'только что';
  if (diff < 60) return `${Math.floor(diff)} с назад`;
  if (diff < 3600) return `${Math.floor(diff / 60)} мин назад`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} ч назад`;
  const days = Math.floor(diff / 86400);
  return `${days} ${plural(days, ['день', 'дня', 'дней'])} назад`;
}

const num1 = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const num0 = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });
const numAuto = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 });

export function formatNumber(value: number, digits?: 0 | 1): string {
  if (!Number.isFinite(value)) return '—';
  if (digits === 0) return num0.format(value);
  if (digits === 1) return num1.format(value);
  return numAuto.format(value);
}

/** Компактная запись: 1 284 → 1,3 тыс. */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 10000) return `${num1.format(value / 1000)} тыс.`;
  return num0.format(value);
}

export function formatPercent(ratio: number, digits: 0 | 1 = 0): string {
  if (!Number.isFinite(ratio)) return '—';
  return `${digits === 0 ? num0.format(ratio * 100) : num1.format(ratio * 100)} %`;
}

export function formatMs(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—';
  if (ms >= 1000) return `${num1.format(ms / 1000)} с`;
  if (ms >= 10) return `${num0.format(ms)} мс`;
  return `${num1.format(ms)} мс`;
}

export function formatSeconds(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(sec)) return '—';
  if (sec < 60) return `${num1.format(sec)} с`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  if (m < 60) return s ? `${m} мин ${s} с` : `${m} мин`;
  const h = Math.floor(m / 60);
  return `${h} ч ${m % 60} мин`;
}

/** Таймер вида 00:42 / 1:02:03 */
export function formatClock(sec: number): string {
  const total = Math.max(0, Math.floor(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(s).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export function shortId(id: string): string {
  return id.length > 14 ? id.slice(0, 8) : id;
}

/** Начало суток (локальное время), epoch-секунды */
export function startOfDay(ts: number): number {
  const d = new Date(ts * 1000);
  d.setHours(0, 0, 0, 0);
  return d.getTime() / 1000;
}

/** Ключ дня YYYY-MM-DD в локальном времени */
export function dayKey(ts: number): string {
  const d = new Date(ts * 1000);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** Номер дня недели с понедельника: 0 = пн … 6 = вс */
export function weekdayIndex(ts: number): number {
  return (new Date(ts * 1000).getDay() + 6) % 7;
}

export const WEEKDAYS_SHORT = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'];

/** Для <input type="datetime-local"> */
export function toLocalInputValue(ts: number): string {
  const d = new Date(ts * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function fromLocalInputValue(value: string): number | undefined {
  if (!value) return undefined;
  const t = new Date(value).getTime();
  return Number.isFinite(t) ? t / 1000 : undefined;
}
