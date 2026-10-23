import { startOfDay } from './format';

export type PeriodPreset = 'today' | '24h' | '7d' | '30d' | 'all' | 'custom';

export const PERIOD_LABELS: Record<PeriodPreset, string> = {
  today: 'Сегодня',
  '24h': '24 часа',
  '7d': '7 дней',
  '30d': '30 дней',
  all: 'Всё время',
  custom: 'Свой период',
};

export interface Range {
  from?: number;
  to?: number;
}

/**
 * Границы периода. now округляется до минуты, чтобы ключ запроса
 * не менялся каждую секунду (иначе — лишние перезапросы).
 */
export function periodRange(preset: PeriodPreset, nowSec: number, custom: Range = {}): Range {
  const now = Math.floor(nowSec / 60) * 60;
  switch (preset) {
    case 'today':
      return { from: startOfDay(now) };
    case '24h':
      return { from: now - 86400 };
    case '7d':
      return { from: startOfDay(now) - 6 * 86400 };
    case '30d':
      return { from: startOfDay(now) - 29 * 86400 };
    case 'custom':
      return custom;
    case 'all':
    default:
      return {};
  }
}
