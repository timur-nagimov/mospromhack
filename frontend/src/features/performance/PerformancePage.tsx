import { Cpu } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { useCameras } from '../../api/queries';
import { ChartCard, Legend } from '../../components/charts/ChartCard';
import { axisProps, gridProps, makeTooltip, SURFACE } from '../../components/charts/ChartTooltip';
import { STAGES } from '../../components/charts/stages';
import { formatMs, formatNumber, formatPercent } from '../../domain/format';
import { backendLabel, SOURCE_TYPE_LABELS, STAGE_PHASE_LABELS } from '../../domain/labels';
import type { CameraMetrics, MetricsSnapshot, Percentiles } from '../../domain/types';
import { useLive } from '../../state/live';
import './performance.css';

/** Категориальная палитра (тёмный режим), порядок фиксирован */
const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
const INPUT_GRAY = '#4b4f56';

function averageTimings(cams: CameraMetrics[]): Record<string, Percentiles> {
  const out: Record<string, Percentiles> = {};
  const keys = new Set(cams.flatMap((c) => Object.keys(c.timings_ms ?? {})));
  for (const key of keys) {
    const vals = cams.map((c) => c.timings_ms?.[key]).filter((v): v is Percentiles => v != null);
    if (vals.length === 0) continue;
    out[key] = {
      p50: vals.reduce((s, v) => s + v.p50, 0) / vals.length,
      p95: vals.reduce((s, v) => s + v.p95, 0) / vals.length,
    };
  }
  return out;
}

/** Загрузка ускорителя: из backend.stats() или оценка Σ(fps × infer p50) */
function acceleratorLoad(m: MetricsSnapshot | null): { value: number | null; estimated: boolean } {
  if (!m) return { value: null, estimated: false };
  const reported = m.inference?.utilization;
  if (typeof reported === 'number') return { value: reported, estimated: false };
  const models = m.inference?.models ?? {};
  let busyMs = 0;
  for (const s of Object.values(models)) {
    const infer = s.timings_ms?.infer?.p50 ?? 0;
    const batch = Math.max(1, s.max_batch_seen ?? 1);
    busyMs += ((s.fps ?? 0) / batch) * infer;
  }
  return busyMs > 0 ? { value: Math.min(1, busyMs / 1000), estimated: true } : { value: null, estimated: true };
}

export function PerformancePage() {
  const metrics = useLive((s) => s.metrics);
  const metricsAt = useLive((s) => s.metricsAt);
  const history = useLive((s) => s.history);
  const { data: cameras = [] } = useCameras();
  const [cameraSel, setCameraSel] = useState<string>('all');

  const camEntries = useMemo(() => Object.entries(metrics?.cameras ?? {}), [metrics]);
  const nameOf = (id: string) => cameras.find((c) => c.camera_id === id)?.name ?? id;
  const colorOf = (id: string) => {
    const idx = cameras.findIndex((c) => c.camera_id === id);
    return SERIES[(idx >= 0 ? idx : camEntries.findIndex(([k]) => k === id)) % SERIES.length];
  };

  const fpsTotal = camEntries.reduce((s, [, m]) => s + (m.fps_processed ?? 0), 0);
  const fpsInTotal = camEntries.reduce((s, [, m]) => s + (m.fps_in ?? 0), 0);
  const maxLag = Math.max(0, ...camEntries.map(([, m]) => m.lag_s ?? 0));
  const allTimings = averageTimings(camEntries.map(([, m]) => m));
  const load = acceleratorLoad(metrics);
  const models = Object.entries(metrics?.inference?.models ?? {});
  const errors =
    camEntries.reduce((s, [, m]) => s + (m.errors ?? 0), 0) + models.reduce((s, [, m]) => s + (m.errors ?? 0), 0);

  const selected = cameraSel === 'all' ? allTimings : (metrics?.cameras?.[cameraSel]?.timings_ms as Record<string, Percentiles>) ?? {};

  const fpsRows = camEntries.map(([id, m]) => ({ id, name: nameOf(id), input: m.fps_in, processed: m.fps_processed }));

  const stageRows = (['p50', 'p95'] as const).map((p) => {
    const row: Record<string, number | string> = { name: p };
    for (const s of STAGES) row[s.key] = selected[s.key]?.[p] ?? 0;
    return row;
  });
  const stageSum = (p: 'p50' | 'p95') => STAGES.reduce((s, st) => s + (selected[st.key]?.[p] ?? 0), 0);

  const historyRows = useMemo(
    () =>
      history.map((pt) => ({
        t: new Date(pt.at).toLocaleTimeString('ru-RU', { minute: '2-digit', second: '2-digit' }),
        ...pt.fps,
      })),
    [history],
  );
  const historyIds = useMemo(() => [...new Set(history.flatMap((p) => Object.keys(p.fps)))], [history]);

  const fpsTooltip = useMemo(() => makeTooltip({ format: (v) => `${formatNumber(v, 1)} к/с` }), []);
  const msTooltip = useMemo(() => makeTooltip({ format: (v) => formatMs(v), hideZero: true, title: (l) => `Перцентиль ${l}` }), []);
  const lineTooltip = useMemo(() => makeTooltip({ format: (v) => `${formatNumber(v, 1)} к/с`, title: (l) => `мм:сс ${l}` }), []);

  const fresh = metricsAt != null && Date.now() - metricsAt < 10_000;

  if (!metrics) {
    return (
      <div className="page">
        <div className="panel empty" style={{ minHeight: 300 }}>
          <span className="spinner" />
          <span className="empty__title">Ждём метрики конвейера</span>
          <span>Источник: WS-сообщения metrics (раз в 2 с) или GET /api/metrics</span>
        </div>
      </div>
    );
  }

  return (
    <div className="page perf">
      <div className="perf-kpis reveal">
        <div className="perf-backend">
          <span className="perf-backend__icon">
            <Cpu size={26} />
          </span>
          <div>
            <span className="stat__label">Активный бэкенд инференса</span>
            <div className="perf-backend__name display">{backendLabel(metrics.backend)}</div>
            <span className="stat__note">
              {String(metrics.inference?.device ?? metrics.inference?.backend ?? metrics.backend)} ·{' '}
              <span className={`live-dot${fresh ? '' : ' live-dot--warn'}`} /> {fresh ? 'метрики обновляются' : 'метрики устарели'}
            </span>
          </div>
        </div>
        <div className="stat">
          <span className="stat__label">Загрузка ускорителя{load.estimated ? ' (оценка)' : ''}</span>
          <span className="stat__value">{load.value != null ? formatPercent(load.value) : '—'}</span>
          <div className="meter meter--blue" aria-hidden="true">
            <span style={{ width: `${(load.value ?? 0) * 100}%` }} />
          </div>
          <span className="stat__note">{load.estimated ? 'Σ fps × p50 инференса по моделям' : 'по данным бэкенда'}</span>
        </div>
        <div className="stat">
          <span className="stat__label">Обработка, кадров/с</span>
          <span className="stat__value">{formatNumber(fpsTotal, 1)}</span>
          <span className="stat__note">
            {camEntries.length} {camEntries.length === 1 ? 'поток' : camEntries.length < 5 ? 'потока' : 'потоков'} · на входе {formatNumber(fpsInTotal, 0)} к/с
          </span>
        </div>
        <div className="stat">
          <span className="stat__label">Сквозная обработка кадра</span>
          <span className="stat__value">
            {formatMs(allTimings.total?.p50)}
            <small> / {formatMs(allTimings.total?.p95)}</small>
          </span>
          <span className="stat__note">p50 / p95, среднее по камерам</span>
        </div>
        <div className={`stat${maxLag > 2 ? ' stat--bad' : ''}`}>
          <span className="stat__label">Макс. отставание</span>
          <span className="stat__value">{formatNumber(maxLag, 1)} с</span>
          <span className="stat__note">ошибок: {errors}</span>
        </div>
      </div>

      <div className="perf-grid">
        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 1 } as React.CSSProperties}
          title="FPS по камерам"
          sub="Обработано против входного потока (анализ идёт с частотой выборки sample_fps)"
          table={{
            columns: ['Камера', 'Вход, к/с', 'Обработано, к/с'],
            rows: fpsRows.map((r) => [r.name, formatNumber(r.input, 1), formatNumber(r.processed, 1)]),
          }}
        >
          <Legend
            items={[
              { label: 'Обработано', color: 'var(--series-1)' },
              { label: 'Входной поток', color: INPUT_GRAY },
            ]}
          />
          <ResponsiveContainer width="100%" height={Math.max(160, fpsRows.length * 44 + 20)}>
            <BarChart data={fpsRows} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }} barGap={2} barCategoryGap={10}>
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" {...axisProps} />
              <YAxis type="category" dataKey="name" {...axisProps} axisLine={false} width={170} tick={{ fill: '#a9a7a0', fontSize: 12 }} />
              <Tooltip content={fpsTooltip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Bar dataKey="processed" name="Обработано" fill="var(--series-1)" radius={[0, 4, 4, 0]} barSize={12} isAnimationActive={false} />
              <Bar dataKey="input" name="Входной поток" fill={INPUT_GRAY} radius={[0, 4, 4, 0]} barSize={6} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          className="reveal"
          style={{ ['--i' as string]: 2 } as React.CSSProperties}
          title="FPS обработки во времени"
          sub="Последние 3 минуты, обновление раз в 2 с"
        >
          <Legend shape="line" items={historyIds.map((id) => ({ label: nameOf(id), color: colorOf(id) }))} />
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={historyRows} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
              <CartesianGrid {...gridProps} vertical={false} />
              <XAxis dataKey="t" {...axisProps} minTickGap={40} />
              <YAxis {...axisProps} axisLine={false} domain={[0, 'auto']} />
              <Tooltip content={lineTooltip} cursor={{ stroke: 'rgba(255,255,255,0.25)' }} />
              {historyIds.map((id) => (
                <Line
                  key={id}
                  type="monotone"
                  dataKey={id}
                  name={nameOf(id)}
                  stroke={colorOf(id)}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: SURFACE, strokeWidth: 2 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          className="perf-wide reveal"
          style={{ ['--i' as string]: 3 } as React.CSSProperties}
          title="Задержка по этапам конвейера"
          sub={
            <>
              Сумма этапов: p50 <b className="num">{formatMs(stageSum('p50'))}</b> · p95 <b className="num">{formatMs(stageSum('p95'))}</b>{' '}
              (сумма p95 по этапам — верхняя оценка; измеренный сквозной p95 — {formatMs(selected.total?.p95)})
            </>
          }
          actions={
            <select className="select select--sm" value={cameraSel} onChange={(e) => setCameraSel(e.target.value)} aria-label="Камера">
              <option value="all">Все камеры (среднее)</option>
              {camEntries.map(([id]) => (
                <option key={id} value={id}>
                  {nameOf(id)}
                </option>
              ))}
            </select>
          }
          table={{
            columns: ['Этап', 'Где', 'p50', 'p95'],
            rows: STAGES.map((s) => [s.label, STAGE_PHASE_LABELS[s.phase], formatMs(selected[s.key]?.p50), formatMs(selected[s.key]?.p95)]),
          }}
        >
          <div className="stage-legend">
            {(['host-pre', 'link', 'accel', 'host-post'] as const).map((phase) => (
              <div key={phase} className="stage-legend__group">
                <span className="stage-legend__phase">{STAGE_PHASE_LABELS[phase]}</span>
                <ul>
                  {STAGES.filter((s) => s.phase === phase).map((s) => (
                    <li key={s.key}>
                      <i style={{ background: s.color }} />
                      {s.label}
                      <b className="num">{formatMs(selected[s.key]?.p50)}</b>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={130}>
            <BarChart data={stageRows} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 0 }} barCategoryGap={12}>
              <CartesianGrid {...gridProps} horizontal={false} />
              <XAxis type="number" {...axisProps} tickFormatter={(v: number) => `${v} мс`} />
              <YAxis type="category" dataKey="name" {...axisProps} axisLine={false} width={40} tick={{ fill: '#edebe6', fontSize: 12, fontWeight: 600 }} />
              <Tooltip content={msTooltip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              {STAGES.map((s, i) => (
                <Bar
                  key={s.key}
                  dataKey={s.key}
                  name={s.label}
                  stackId="lat"
                  fill={s.color}
                  stroke={SURFACE}
                  strokeWidth={2}
                  barSize={26}
                  radius={i === STAGES.length - 1 ? [0, 4, 4, 0] : 0}
                  isAnimationActive={false}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard className="perf-wide reveal" style={{ ['--i' as string]: 4 } as React.CSSProperties} title="Потоки" sub="Состояние каждой камеры">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Камера</th>
                  <th>Источник</th>
                  <th className="right">Выборка</th>
                  <th className="right">Вход</th>
                  <th className="right">Обработано</th>
                  <th className="right">Отставание</th>
                  <th className="right">Кадр p50</th>
                  <th className="right">Кадр p95</th>
                  <th className="right">Инференс p95</th>
                  <th className="right">Ошибки</th>
                </tr>
              </thead>
              <tbody>
                {camEntries.map(([id, m]) => {
                  const cam = cameras.find((c) => c.camera_id === id);
                  return (
                    <tr key={id}>
                      <td>
                        <span className="perf-cam">
                          <i style={{ background: colorOf(id) }} />
                          {nameOf(id)} <span className="mono muted">{id}</span>
                        </span>
                      </td>
                      <td className="dim">{cam ? SOURCE_TYPE_LABELS[cam.source.type] : '—'}</td>
                      <td className="right">{cam ? `${formatNumber(cam.sample_fps, 1)} к/с` : '—'}</td>
                      <td className="right">{formatNumber(m.fps_in, 1)}</td>
                      <td className="right">
                        <b>{formatNumber(m.fps_processed, 1)}</b>
                      </td>
                      <td className={`right${m.lag_s > 2 ? ' perf-bad' : ''}`}>{formatNumber(m.lag_s)} с</td>
                      <td className="right">{formatMs(m.timings_ms?.total?.p50)}</td>
                      <td className="right">{formatMs(m.timings_ms?.total?.p95)}</td>
                      <td className="right">{formatMs(m.timings_ms?.infer?.p95)}</td>
                      <td className="right">{m.errors ?? 0}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </ChartCard>

        {models.length > 0 && (
          <ChartCard className="perf-wide reveal" style={{ ['--i' as string]: 5 } as React.CSSProperties} title="Модели на ускорителе" sub="backend.stats(): вызовы, батчи, передача и инференс">
            <div className="table-wrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>Модель</th>
                    <th className="right">Кадров/с</th>
                    <th className="right">Вызовов</th>
                    <th className="right">Макс. батч</th>
                    <th className="right">Передача p50 / p95</th>
                    <th className="right">Инференс p50 / p95</th>
                    <th className="right">Ошибки</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map(([id, s]) => (
                    <tr key={id}>
                      <td className="mono">{id}</td>
                      <td className="right">{formatNumber(s.fps ?? 0, 1)}</td>
                      <td className="right">{s.calls ?? '—'}</td>
                      <td className="right">{s.max_batch_seen ?? '—'}</td>
                      <td className="right">
                        {formatMs(s.timings_ms?.transfer?.p50)} / {formatMs(s.timings_ms?.transfer?.p95)}
                      </td>
                      <td className="right">
                        <b>{formatMs(s.timings_ms?.infer?.p50)}</b> / {formatMs(s.timings_ms?.infer?.p95)}
                      </td>
                      <td className={`right${(s.errors ?? 0) > 0 ? ' perf-bad' : ''}`} title={s.last_error ?? undefined}>
                        {s.errors ?? 0}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        )}
      </div>
    </div>
  );
}
