/** Календарь аналитики совпадает с серверными бакетами Europe/Moscow. */
const dateFormatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'Europe/Moscow', year: 'numeric', month: '2-digit', day: '2-digit',
});
const hourFormatter = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'Europe/Moscow', hour: '2-digit', hourCycle: 'h23',
});

export function moscowDayKey(ts: number): string {
  const parts = dateFormatter.formatToParts(ts * 1000);
  const part = (name: string) => parts.find((p) => p.type === name)?.value;
  return `${part('year')}-${part('month')}-${part('day')}`;
}

export function moscowHour(ts: number): number {
  return Number(hourFormatter.format(ts * 1000));
}

export function moscowWeekdayHour(ts: number): string {
  const day = new Date(`${moscowDayKey(ts)}T00:00:00Z`).getUTCDay();
  return `${(day + 6) % 7}:${moscowHour(ts)}`;
}

/** В текущем календаре МСК всегда UTC+3, без сезонного перевода часов. */
export function moscowStartOfDay(ts: number): number {
  return Date.parse(`${moscowDayKey(ts)}T00:00:00+03:00`) / 1000;
}
