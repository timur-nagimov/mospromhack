import { formatMs } from '../../domain/format';
import { STAGES } from './stages';

/** Компактная полоса «куда ушло время» по этапам обработки (div-реализация, без библиотек) */
export function StageBar({ values, showLegend = true }: { values: Record<string, number | undefined>; showLegend?: boolean }) {
  const parts = STAGES.map((s) => ({ ...s, value: values[s.key] ?? 0 })).filter((p) => p.value > 0);
  const total = parts.reduce((sum, p) => sum + p.value, 0);
  if (total <= 0) return <span className="muted">нет замеров</span>;
  return (
    <div className="stagebar">
      <div className="stagebar__track" role="img" aria-label={parts.map((p) => `${p.label} ${formatMs(p.value)}`).join(', ')}>
        {parts.map((p) => (
          <span
            key={p.key}
            className="stagebar__seg"
            style={{ flexGrow: p.value, background: p.color }}
            title={`${p.label}: ${formatMs(p.value)}`}
          />
        ))}
      </div>
      {showLegend && (
        <ul className="stagebar__legend">
          {parts.map((p) => (
            <li key={p.key}>
              <i style={{ background: p.color }} />
              <span>{p.label}</span>
              <b className="num">{formatMs(p.value)}</b>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
