import type { TooltipProps } from 'recharts';

interface Options {
  /** Формат значения */
  format?: (value: number, name: string) => string;
  /** Заголовок по метке оси */
  title?: (label: string | number) => string;
  /** Скрыть строки с нулём */
  hideZero?: boolean;
}

/** Подсказка: значение — главное, имя серии — вторично; ключ серии — короткая линия */
export function makeTooltip({ format, title, hideZero }: Options = {}) {
  return function ChartTooltip({ active, payload, label }: TooltipProps<number, string>) {
    if (!active || !payload || payload.length === 0) return null;
    const rows = payload.filter((p) => !(hideZero && !p.value));
    return (
      <div className="tip">
        <div className="tip__title">{title ? title(label as string) : label}</div>
        {rows.map((p) => (
          <div className="tip__row" key={String(p.dataKey)}>
            <span className="tip__key">
              <i style={{ ['--k' as string]: (p.payload?.fill as string) || p.color }} />
              {p.name}
            </span>
            <span className="tip__val">{format ? format(Number(p.value), String(p.name)) : p.value}</span>
          </div>
        ))}
      </div>
    );
  };
}

export const axisProps = {
  tick: { fill: '#8d8b85', fontSize: 11.5 },
  tickLine: false,
  axisLine: { stroke: 'var(--chart-axis)' },
} as const;

export const gridProps = {
  stroke: 'var(--chart-grid)',
  strokeDasharray: undefined,
} as const;

export const SURFACE = '#111317';
