import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useCameraMap, useEvents, useStats } from '../../api/queries';
import { ChartCard, Legend } from '../../components/charts/ChartCard';
import { axisProps, gridProps, makeTooltip } from '../../components/charts/ChartTooltip';
import { ViolationIcon } from '../../components/ui/badges';
import { formatDate, formatPercent, formatWeekday, plural, WEEKDAYS_SHORT } from '../../domain/format';
import { STATUS_LABELS } from '../../domain/labels';
import { PERIOD_LABELS, periodRange, type PeriodPreset } from '../../domain/period';
import type { EventStatus } from '../../domain/types';
import { getViolationMeta, isCriticalEvent } from '../../domain/violations';
import { useNow } from '../../hooks/useNow';
import { fillDays, statusByType, sumStatuses, weekdayHourMatrix } from './aggregate';
import './analytics.css';

const PRESETS: PeriodPreset[] = ['today', '7d', '30d', 'all'];
const EVENTS_LIMIT = 5000;

/** Цвета решений оператора — те же, что у бейджей статуса */
const STATUS_COLORS: Record<EventStatus, string> = {
  confirmed: '#3dbb72',
  forwarded: '#5b9cf5',
  false: '#c9c6bd',
  new: '#3a3d43',
};
const STATUS_SEQ: EventStatus[] = ['confirmed', 'forwarded', 'false', 'new'];

const HEAT_LOW = '#16263b';
const HEAT_HIGH = '#a9cdf6';

function heatColor(value: number, max: number): string {
  if (value <= 0 || max <= 0) return 'var(--bg-3)';
  const t = 0.12 + 0.88 * (value / max);
  return `color-mix(in oklab, ${HEAT_HIGH} ${Math.round(t * 100)}%, ${HEAT_LOW})`;
}

function Heatmap({ matrix }: { matrix: number[][] }) {
  const [hover, setHover] = useState<{ d: number; h: number } | null>(null);
  const max = Math.max(0, ...matrix.flat());
  const peak = useMemo(() => {
    let best = { d: 0, h: 0, v: -1 };
    matrix.forEach((row, d) => row.forEach((v, h) => v > best.v && (best = { d, h, v })));
    return best;
  }, [matrix]);
  const shown = hover ?? (peak.v > 0 ? { d: peak.d, h: peak.h } : null);
  return (
    <div>
      <div className="heatmap" role="grid" aria-label="Нарушения по дням недели и часам">
        <span className="heatmap__corner" />
        {Array.from({ length: 24 }, (_, h) => (
          <span key={h} className="heatmap__hour">
            {h % 3 === 0 ? String(h).padStart(2, '0') : ''}
          </span>
        ))}
        {matrix.map((row, d) => (
          <div key={d} role="row" style={{ display: 'contents' }}>
            <span className="heatmap__day">{WEEKDAYS_SHORT[d]}</span>
            {row.map((v, h) => (
              <button
                key={h}
                type="button"
                className="heatmap__cell"
                style={{ background: heatColor(v, max) }}
                aria-label={`${WEEKDAYS_SHORT[d]}, ${h}:00–${h + 1}:00 — ${v}`}
                onMouseEnter={() => setHover({ d, h })}
                onFocus={() => setHover({ d, h })}
                onMouseLeave={() => setHover(null)}
                onBlur={() => setHover(null)}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="heatmap__foot">
        <span className="heatmap__readout">
          {shown ? (
            <>
              {hover ? '' : 'Пик: '}
              {WEEKDAYS_SHORT[shown.d]}, {String(shown.h).padStart(2, '0')}:00–{String(shown.h + 1).padStart(2, '0')}:00 —{' '}
              <b className="num">{matrix[shown.d][shown.h]}</b>{' '}
              {plural(matrix[shown.d][shown.h], ['нарушение', 'нарушения', 'нарушений'])}
            </>
          ) : (
            'Нет данных за период'
          )}
        </span>
        <span className="scale">
          0
          <span className="scale__bar" style={{ background: `linear-gradient(90deg, ${HEAT_LOW}, ${HEAT_HIGH})` }} />
          {max}
        </span>
      </div>
    </div>
  );
}

export function AnalyticsPage() {
  const now = useNow();
  const [preset, setPreset] = useState<PeriodPreset>('30d');
  const minute = Math.floor(now / 60);
  const range = useMemo(() => periodRange(preset, minute * 60), [preset, minute]);
  const cameraMap = useCameraMap();

  const byType = useStats('type', range.from, range.to);
  const byCamera = useStats('camera', range.from, range.to);
  const byDay = useStats('day', range.from, range.to);
  const events = useEvents({ from: range.from, to: range.to, limit: EVENTS_LIMIT });

  const eventItems = useMemo(() => events.data?.items ?? [], [events.data]);
  const truncated = (events.data?.total ?? 0) > eventItems.length;

  const typeRows = useMemo(
    () =>
      [...(byType.data?.items ?? [])]
        .sort((a, b) => b.count - a.count)
        .map((i) => ({ key: i.key, name: getViolationMeta(i.key).short, title: getViolationMeta(i.key).title, count: i.count })),
    [byType.data],
  );
  const total = typeRows.reduce((s, r) => s + r.count, 0);

  const sites = useMemo(() => {
    const groups = new Map<string, { site: string; total: number; cameras: { id: string; name: string; count: number }[] }>();
    for (const item of byCamera.data?.items ?? []) {
      const cam = cameraMap.get(item.key);
      const site = cam?.site ?? 'Без участка';
      const g = groups.get(site) ?? { site, total: 0, cameras: [] };
      g.total += item.count;
      g.cameras.push({ id: item.key, name: cam?.name ?? item.key, count: item.count });
      groups.set(site, g);
    }
    const list = [...groups.values()].sort((a, b) => b.total - a.total);
    list.forEach((g) => g.cameras.sort((a, b) => b.count - a.count));
    return list;
  }, [byCamera.data, cameraMap]);
  const siteMax = Math.max(1, ...sites.map((s) => s.total));

  const dayRows = useMemo(() => {
    const filled = fillDays(byDay.data?.items ?? [], range.from, now);
    return filled.map((d) => {
      const ts = new Date(`${d.key}T12:00:00`).getTime() / 1000;
      return { ...d, label: formatDate(ts), weekday: formatWeekday(ts) };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [byDay.data, range.from, minute]);

  const matrix = useMemo(() => weekdayHourMatrix(eventItems), [eventItems]);
  const statusRows = useMemo(() => statusByType(eventItems), [eventItems]);
  const statusTotals = useMemo(() => sumStatuses(statusRows), [statusRows]);
  const reviewed = statusTotals.confirmed + statusTotals.forwarded + statusTotals.false;
  const precision = reviewed > 0 ? (statusTotals.confirmed + statusTotals.forwarded) / reviewed : null;
  const critical = eventItems.filter((e) => isCriticalEvent(e)).length;

  const typeTooltip = useMemo(
    () => makeTooltip({ title: (l) => typeRows.find((r) => r.name === l)?.title ?? String(l), format: (v) => `${v}` }),
    [typeRows],
  );
  const dayTooltip = useMemo(
    () =>
      makeTooltip({
        title: (l) => {
          const row = dayRows.find((r) => r.label === l);
          return row ? `${row.weekday}, ${row.label}` : String(l);
        },
      }),
    [dayRows],
  );

  const loading = byType.isFetching || byCamera.isFetching || byDay.isFetching || events.isFetching;

  return (
    <div className="page analytics">
      <div className="analytics__filters reveal">
        <div className="segmented" role="group" aria-label="Период">
          {PRESETS.map((p) => (
            <button key={p} type="button" aria-pressed={preset === p} onClick={() => setPreset(p)}>
              {PERIOD_LABELS[p]}
            </button>
          ))}
        </div>
        <span className="muted">Все графики ниже — за выбранный период</span>
        {loading && <span className="spinner" style={{ width: 14, height: 14 }} />}
      </div>

      <div className="stat-row reveal" style={{ ['--i' as string]: 1 }}>
        <div className="stat">
          <span className="stat__label">Нарушений</span>
          <span className="stat__value">{total}</span>
          <span className="stat__note">{PERIOD_LABELS[preset].toLowerCase()}</span>
        </div>
        <div className="stat">
          <span className="stat__label">Критических</span>
          <span className="stat__value">{critical}</span>
          <span className="stat__note">падения, сближения с техникой и правила с критической важностью</span>
        </div>
        <div className="stat stat--wide">
          <span className="stat__label">Точность системы по решениям оператора</span>
          <span className="stat__value">{precision != null ? formatPercent(precision) : '—'}</span>
          <div className="meter" aria-hidden="true">
            <span style={{ width: `${(precision ?? 0) * 100}%` }} />
          </div>
          <span className="stat__note">
            подтверждено и передано мастеру из {reviewed} разобранных · ложных {reviewed ? formatPercent(statusTotals.false / reviewed) : '—'}
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">Не разобрано</span>
          <span className="stat__value">{statusTotals.new}</span>
          <span className="stat__note">не попадают в оценку точности</span>
        </div>
      </div>

      <div className="analytics__grid">
        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 2 } as React.CSSProperties}
          title="Нарушения по типам"
          sub="Сколько раз сработало каждое правило"
          loading={byType.isFetching}
          table={{ columns: ['Тип нарушения', 'Событий', 'Доля'], rows: typeRows.map((r) => [r.title, r.count, formatPercent(total ? r.count / total : 0)]) }}
        >
          {typeRows.length === 0 ? (
            <div className="empty">Нет данных за период</div>
          ) : (
            <ResponsiveContainer width="100%" height={typeRows.length * 32 + 24}>
              <BarChart data={typeRows} layout="vertical" margin={{ top: 0, right: 40, bottom: 0, left: 0 }} barCategoryGap={6}>
                <CartesianGrid {...gridProps} horizontal={false} />
                <XAxis type="number" {...axisProps} allowDecimals={false} hide />
                <YAxis type="category" dataKey="name" {...axisProps} axisLine={false} width={150} tick={{ fill: '#a9a7a0', fontSize: 12 }} />
                <Tooltip content={typeTooltip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <Bar dataKey="count" name="Событий" fill="var(--series-1)" radius={[0, 4, 4, 0]} barSize={16} isAnimationActive={false}>
                  <LabelList dataKey="count" position="right" fill="#edebe6" fontSize={12} fontWeight={600} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 3 } as React.CSSProperties}
          title="По участкам и камерам"
          sub="Где нарушают чаще всего"
          loading={byCamera.isFetching}
          table={{
            columns: ['Участок / камера', 'Событий'],
            rows: sites.flatMap((s) => [[s.site, s.total], ...s.cameras.map((c) => [`   ${c.name}`, c.count])]),
          }}
        >
          {sites.length === 0 ? (
            <div className="empty">Нет данных за период</div>
          ) : (
            <div className="hbars">
              {sites.map((s) => (
                <div key={s.site}>
                  <div className="hbar hbar--group" title={`${s.site}: ${s.total}`}>
                    <span className="hbar__label">{s.site}</span>
                    <span className="hbar__track">
                      <span className="hbar__fill" style={{ width: `${(s.total / siteMax) * 100}%`, display: 'block' }} />
                    </span>
                    <span className="hbar__value">{s.total}</span>
                  </div>
                  {s.cameras.map((c) => (
                    <div key={c.id} className="hbar hbar--child" title={`${c.name}: ${c.count}`}>
                      <span className="hbar__label">{c.name}</span>
                      <span className="hbar__track">
                        <span className="hbar__fill" style={{ width: `${(c.count / siteMax) * 100}%`, display: 'block' }} />
                      </span>
                      <span className="hbar__value">{c.count}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </ChartCard>

        <ChartCard
          className="analytics__wide reveal"
          style={{ ['--i' as string]: 4 } as React.CSSProperties}
          title="Когда нарушают: день недели × час"
          sub={
            truncated
              ? `По последним ${eventItems.length} событиям периода — для полной картины нужна серверная агрегация`
              : 'Плотность нарушений по времени суток — для планирования обходов и инструктажей'
          }
          loading={events.isFetching}
          table={{
            columns: ['День', ...Array.from({ length: 24 }, (_, h) => `${h}`)],
            rows: matrix.map((row, d) => [WEEKDAYS_SHORT[d], ...row]),
          }}
        >
          <Heatmap matrix={matrix} />
        </ChartCard>

        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 5 } as React.CSSProperties}
          title="По дням"
          sub="Динамика нарушений за период"
          loading={byDay.isFetching}
          table={{ columns: ['День', 'Событий'], rows: dayRows.map((d) => [`${d.weekday}, ${d.label}`, d.count]) }}
        >
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={dayRows} margin={{ top: 16, right: 18, bottom: 0, left: -18 }} barCategoryGap="22%">
              <CartesianGrid {...gridProps} vertical={false} />
              <XAxis dataKey="label" {...axisProps} interval="preserveStartEnd" minTickGap={18} />
              <YAxis {...axisProps} axisLine={false} allowDecimals={false} />
              <Tooltip content={dayTooltip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar dataKey="count" name="Нарушений" fill="var(--series-1)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 6 } as React.CSSProperties}
          title="Решения оператора по типам"
          sub="Доля ложных срабатываний среди разобранных — оценка точности каждого правила"
          loading={events.isFetching}
          table={{
            columns: ['Тип', ...STATUS_SEQ.map((s) => STATUS_LABELS[s]), 'Ложных'],
            rows: statusRows.map((r) => [
              getViolationMeta(r.type).title,
              ...STATUS_SEQ.map((s) => r.counts[s]),
              r.falseRate != null ? formatPercent(r.falseRate) : '—',
            ]),
          }}
        >
          <Legend items={STATUS_SEQ.map((s) => ({ label: STATUS_LABELS[s], color: STATUS_COLORS[s] }))} />
          <div className="status-rows">
            <div className="status-row status-row--head">
              <span />
              <span />
              <span>ложных</span>
            </div>
            {statusRows.map((r) => (
              <div key={r.type} className="status-row" title={STATUS_SEQ.map((s) => `${STATUS_LABELS[s]}: ${r.counts[s]}`).join('\n')}>
                <span className="status-row__label">
                  <ViolationIcon type={r.type} severity="low" size="sm" />
                  {getViolationMeta(r.type).short}
                </span>
                <span className="stack100" aria-label={STATUS_SEQ.map((s) => `${STATUS_LABELS[s]} ${r.counts[s]}`).join(', ')}>
                  {STATUS_SEQ.map((s) =>
                    r.counts[s] > 0 ? <span key={s} style={{ flexGrow: r.counts[s], background: STATUS_COLORS[s] }} /> : null,
                  )}
                </span>
                <span className="status-row__rate num">
                  {r.falseRate != null ? formatPercent(r.falseRate) : '—'}
                  <small> из {r.reviewed}</small>
                </span>
              </div>
            ))}
          </div>
        </ChartCard>
      </div>
    </div>
  );
}
