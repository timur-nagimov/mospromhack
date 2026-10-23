import { ChartColumn, Table2 } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import './charts.css';

export interface TableView {
  columns: string[];
  rows: (string | number)[][];
}

interface Props {
  title: string;
  sub?: ReactNode;
  /** Табличный двойник графика (доступность: значения без наведения) */
  table?: TableView;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  loading?: boolean;
  style?: React.CSSProperties;
}

export function ChartCard({ title, sub, table, actions, children, className, loading, style }: Props) {
  const [asTable, setAsTable] = useState(false);
  return (
    <section className={`panel chart-card${className ? ` ${className}` : ''}`} style={style}>
      <header className="chart-card__head">
        <div>
          <h2 className="panel__title">{title}</h2>
          {sub && <p className="panel__sub">{sub}</p>}
        </div>
        <div className="chart-card__actions">
          {actions}
          {table && (
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => setAsTable((v) => !v)}
              aria-pressed={asTable}
              title={asTable ? 'Показать график' : 'Показать таблицей'}
            >
              {asTable ? <ChartColumn size={15} /> : <Table2 size={15} />}
              {asTable ? 'График' : 'Таблица'}
            </button>
          )}
        </div>
      </header>
      <div className="chart-card__body loading-veil" data-loading={Boolean(loading)}>
        {asTable && table ? (
          <div className="table-wrap chart-card__table">
            <table className="table">
              <thead>
                <tr>
                  {table.columns.map((c, i) => (
                    <th key={c} className={i > 0 ? 'right' : undefined}>
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, ri) => (
                  <tr key={ri}>
                    {row.map((v, ci) => (
                      <td key={ci} className={ci > 0 ? 'right' : undefined}>
                        {v}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          children
        )}
      </div>
    </section>
  );
}

/** Легенда: прямоугольник для столбцов/областей, линия для линий */
export function Legend({ items, shape = 'rect' }: { items: { label: string; color: string }[]; shape?: 'rect' | 'line' }) {
  return (
    <ul className="legend">
      {items.map((it) => (
        <li key={it.label}>
          <i className={`legend__key legend__key--${shape}`} style={{ background: it.color }} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}
