// Смены: дневная 08:00–20:00, ночная 20:00–08:00 (местное время)

export interface Shift {
  start: number; // epoch, с
  label: string;
}

export function currentShift(now: number): Shift {
  const d = new Date(now * 1000);
  const h = d.getHours();
  const start = new Date(d);
  start.setMinutes(0, 0, 0);
  if (h >= 8 && h < 20) {
    start.setHours(8);
    return { start: start.getTime() / 1000, label: 'Дневная смена с 08:00' };
  }
  if (h < 8) start.setDate(start.getDate() - 1);
  start.setHours(20);
  return { start: start.getTime() / 1000, label: 'Ночная смена с 20:00' };
}
