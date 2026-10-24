import { Boxes, ExternalLink, GraduationCap, Lock, Rocket, Scale, ShieldCheck, ShieldQuestion, ShieldX } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { isForbidden, useStartJob } from '../../api/flywheelQueries';
import { toast } from '../../components/ui/toast';
import {
  datasetVersionOf,
  eventRates,
  jobKindLabel,
  jobLogTail,
  safeUrl,
  toEpoch,
  type DatasetVersion,
  type FlywheelJob,
  type FlywheelSummary,
  type JobKind,
  type ModelCard,
} from '../../domain/flywheel';
import { formatNumber, formatSeconds, formatSmartDateTime, nowSec, formatRelative } from '../../domain/format';
import { JobPhaseIcon, ModelStatusBadge, trainGuard, useTrainForbidden } from './parts';

type Better = 'higher' | 'lower';

interface MetricRow {
  label: ReactNode;
  value: number | null;
  base: number | null;
  better: Better;
  format: (v: number) => string;
  /** подпись под значением, например «4/4» */
  detail?: string;
  baseDetail?: string;
}

const fmt2 = (v: number) => v.toFixed(2);
const fmtInt = (v: number) => formatNumber(v, 0);

function verdict(row: MetricRow): 'up' | 'down' | 'same' | null {
  if (row.value == null || row.base == null) return null;
  const diff = row.value - row.base;
  if (Math.abs(diff) < 1e-9) return 'same';
  const better = row.better === 'higher' ? diff > 0 : diff < 0;
  return better ? 'up' : 'down';
}

function MetricsTable({ card }: { card: ModelCard }) {
  const ev = card.holdout?.events ?? null;
  const baseEv = card.baseline?.holdout?.events ?? null;
  const test = card.holdout?.dataset_test ?? null;
  const baseTest = card.baseline?.holdout?.dataset_test ?? null;
  const rates = eventRates(ev);
  const baseRates = eventRates(baseEv);
  const frac = (a?: number, b?: number) => (a != null && b != null ? `${a}/${a + b}` : undefined);

  const candidates: MetricRow[] = [
    { label: 'Recall по событиям', value: rates.recall, base: baseRates.recall, better: 'higher', format: fmt2, detail: frac(ev?.tp, ev?.fn), baseDetail: frac(baseEv?.tp, baseEv?.fn) },
    { label: 'Precision по событиям', value: rates.precision, base: baseRates.precision, better: 'higher', format: fmt2, detail: frac(ev?.tp, ev?.fp), baseDetail: frac(baseEv?.tp, baseEv?.fp) },
    ev?.fp_per_hour != null || baseEv?.fp_per_hour != null
      ? { label: 'Ложных тревог в час', value: ev?.fp_per_hour ?? null, base: baseEv?.fp_per_hour ?? null, better: 'lower', format: (v: number) => formatNumber(v, 1) }
      : { label: 'Ложные тревоги', value: ev?.fp ?? null, base: baseEv?.fp ?? null, better: 'lower', format: fmtInt },
    { label: 'Дубли', value: ev?.duplicates ?? null, base: baseEv?.duplicates ?? null, better: 'lower', format: fmtInt },
    { label: 'Задержка p50', value: ev?.latency_p50_s ?? null, base: baseEv?.latency_p50_s ?? null, better: 'lower', format: (v: number) => formatSeconds(v) },
    {
      label: (
        <>
          mAP50 на тесте набора
          {test?.name && <small className="fw-metrics__sub">{test.name}{test.images ? `, ${test.images} кадров` : ''}</small>}
        </>
      ),
      value: test?.map50 ?? null,
      base: baseTest?.map50 ?? null,
      better: 'higher',
      format: fmt2,
    },
  ];
  const rows = candidates.filter((r) => r.value != null || r.base != null);

  if (rows.length === 0) return <p className="fw-model__empty">Метрик на отложенных данных в карточке нет.</p>;

  return (
    <table className="fw-metrics">
      <thead>
        <tr>
          <th>На отложенных данных</th>
          <th className="right">Эта версия</th>
          <th className="right" title={card.baseline?.version ?? undefined}>
            База
          </th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => {
          const v = verdict(r);
          return (
            <tr key={i}>
              <td>{r.label}</td>
              <td className={`right num fw-metrics__val${v ? ` fw-metrics__val--${v}` : ''}`}>
                {r.value != null ? r.format(r.value) : '—'}
                <span className="fw-metrics__mark" aria-label={v === 'up' ? 'лучше базы' : v === 'down' ? 'хуже базы' : v === 'same' ? 'как у базы' : undefined}>
                  {v === 'up' ? '▲' : v === 'down' ? '▼' : v === 'same' ? '=' : ''}
                </span>
                {r.detail && <small>{r.detail}</small>}
              </td>
              <td className="right num muted">
                {r.base != null ? r.format(r.base) : '—'}
                {r.baseDetail && <small>{r.baseDetail}</small>}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function GateBlock({ card }: { card: ModelCard }) {
  const gate = card.gate;
  if (!gate) {
    return (
      <div className="fw-gate fw-gate--none">
        <ShieldQuestion size={18} aria-hidden="true" />
        <div>
          <strong>Гейт ещё не запускали</strong>
          <p>Решение появится после прогона отложенных роликов и теста наборов</p>
        </div>
      </div>
    );
  }
  return (
    <div className={`fw-gate ${gate.promote ? 'fw-gate--pass' : 'fw-gate--fail'}`}>
      {gate.promote ? <ShieldCheck size={18} aria-hidden="true" /> : <ShieldX size={18} aria-hidden="true" />}
      <div>
        <strong>{gate.promote ? 'Гейт пройден: не хуже базы' : 'Гейт не пройден: хуже базы'}</strong>
        {gate.reasons && gate.reasons.length > 0 && (
          <ul>
            {gate.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function ModelCardView({
  card,
  forbidden,
  onDeploy,
  deploying,
}: {
  card: ModelCard;
  forbidden: string | null;
  onDeploy: (card: ModelCard) => void;
  deploying: boolean;
}) {
  const created = toEpoch(card.created_at);
  const train = card.train;
  const tpu = card.tpu_check;
  const canDeploy = card.status === 'candidate' && card.gate?.promote === true;
  return (
    <article className={`panel fw-model fw-model--${card.status}`}>
      <header className="fw-model__head">
        <div>
          <span className="eyebrow">{card.model_id}</span>
          <h3 className="display">{card.version}</h3>
          <p>
            {card.parent ? <>от {card.parent}</> : 'с нуля'}
            {created ? ` · ${formatSmartDateTime(created)}` : ''}
            {card.classes?.length ? ` · классы: ${card.classes.join(', ')}` : ''}
          </p>
        </div>
        <ModelStatusBadge status={card.status} />
      </header>

      <GateBlock card={card} />

      <MetricsTable card={card} />

      <div className="fw-model__facts">
        {tpu && (tpu.recall != null || tpu.precision != null) && (
          <div>
            <span className="fw-model__key">TPU int8 против CPU fp32</span>
            <span className="num">
              recall {tpu.recall != null ? fmt2(tpu.recall) : '—'} · precision {tpu.precision != null ? fmt2(tpu.precision) : '—'}
              {tpu.frames ? <span className="muted"> · {tpu.frames} кадров</span> : null}
            </span>
          </div>
        )}
        {train && (
          <div>
            <span className="fw-model__key">Обучение</span>
            <span>
              {[
                train.epochs ? `${train.epochs} эпох` : null,
                train.imgsz ? `вход ${train.imgsz}` : null,
                train.device ?? null,
                train.minutes ? `${train.minutes} мин` : null,
              ]
                .filter(Boolean)
                .join(' · ') || '—'}
            </span>
          </div>
        )}
        {train?.datasets && train.datasets.length > 0 && (
          <div>
            <span className="fw-model__key">Данные</span>
            <ul className="fw-mix">
              {train.datasets.map((d) => {
                const url = safeUrl(d.url);
                return (
                  <li key={d.name}>
                    {url ? (
                      <a href={url} target="_blank" rel="noreferrer noopener">
                        {d.name} <ExternalLink size={11} aria-hidden="true" />
                      </a>
                    ) : (
                      <span className="mono">{d.name}</span>
                    )}
                    {d.images != null && <b className="num">{formatNumber(d.images, 0)}</b>}
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        <div>
          <span className="fw-model__key">Программа TPU</span>
          <span className={`mono fw-model__path${card.tpu_program ? '' : ' muted'}`}>{card.tpu_program ?? 'не собрана'}</span>
        </div>
      </div>

      {canDeploy && (
        <footer className="fw-model__foot">
          <span className="fw-model__hint">{forbidden ?? 'Гейт пройден — можно включить в configs/models*.yaml'}</span>
          <button type="button" className="btn btn--sm" onClick={() => onDeploy(card)} disabled={Boolean(forbidden) || deploying} title={forbidden ?? 'Выкатить версию'}>
            {forbidden ? <Lock size={14} /> : <Rocket size={14} />} Выкатить
          </button>
        </footer>
      )}
    </article>
  );
}

function JobsList({ jobs }: { jobs: FlywheelJob[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const now = nowSec();
  if (jobs.length === 0) return <p className="fw-model__empty">Задач ещё не было.</p>;
  return (
    <ul className="fw-jobs">
      {jobs.slice(0, 8).map((job) => {
        const tail = jobLogTail(job);
        const at = toEpoch(job.finished_at ?? job.started_at ?? job.created_at);
        const expanded = open === job.id;
        return (
          <li key={job.id} className="fw-job">
            <button type="button" className="fw-job__row" onClick={() => setOpen(expanded ? null : job.id)} aria-expanded={expanded}>
              <JobPhaseIcon status={job.status} />
              <span className="fw-job__kind">{jobKindLabel(job.kind)}</span>
              <span className="mono muted fw-job__id">{job.id}</span>
              <span className="fw-job__status">{job.status}</span>
              <span className="muted num fw-job__at">{at ? formatRelative(at, now) : ''}</span>
            </button>
            {(expanded || tail.length > 0) && (
              <pre className={`fw-job__log${expanded ? ' is-open' : ''}`}>{(expanded ? tail : tail.slice(-2)).join('\n') || 'лог пуст'}</pre>
            )}
            {expanded && job.log_url && (
              <a className="fw-link fw-job__more" href={job.log_url} target="_blank" rel="noreferrer noopener">
                весь лог <ExternalLink size={11} aria-hidden="true" />
              </a>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function ModelsPanel({
  summary,
  models,
  datasets,
  jobs,
  isLoading,
  error,
}: {
  summary: FlywheelSummary | undefined;
  models: ModelCard[];
  datasets: DatasetVersion[];
  jobs: FlywheelJob[];
  isLoading: boolean;
  error: Error | null;
}) {
  const startJob = useStartJob();
  const forbidden = useTrainForbidden(summary?.allow_train);
  const modelIds = [...new Set(models.map((m) => m.model_id))];
  const [modelChoice, setModelChoice] = useState('');
  const modelId = modelChoice || modelIds[0] || '';
  const dataset = datasets[0]?.version ?? datasetVersionOf(summary?.last_dataset);

  const launch = (kind: JobKind, args: Record<string, unknown>, ok: string) => {
    startJob.mutate(
      { kind, args },
      {
        onSuccess: () => toast(ok),
        onError: (err) => {
          if (isForbidden(err)) {
            // сервер без GUARD_FLYWHEEL_ALLOW_TRAIN=1: кнопки гаснут и объясняют почему
            trainGuard.forbid((err as Error).message || 'Задачи train и deploy выключены на сервере');
            toast(`Запрещено сервером: ${(err as Error).message}`, 'bad');
          } else {
            toast(`Не запущено: ${(err as Error).message}`, 'bad');
          }
        },
      },
    );
  };

  return (
    <div className="fw-models">
      <section className="panel fw-train reveal">
        <div className="fw-train__main">
          <span className="fw-train__icon">
            <GraduationCap size={22} />
          </span>
          <div>
            <h3 className="panel__title">Дообучение и сборка под LinQ</h3>
            <p className="panel__sub">
              ultralytics на MPS → ONNX → компиляция TPU → сверка TPU с CPU → гейт на отложенных данных → реестр моделей
            </p>
          </div>
        </div>
        <div className="fw-train__controls">
          <label className="field">
            <span className="field__label">Модель</span>
            <select className="select" value={modelId} onChange={(e) => setModelChoice(e.target.value)} disabled={modelIds.length === 0}>
              {modelIds.length === 0 && <option value="">нет карточек</option>}
              {modelIds.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </label>
          <div className="field">
            <span className="field__label">Датасет</span>
            <span className="fw-train__ds mono">{dataset ?? '—'}</span>
          </div>
          <span className="fw-train__btn" title={forbidden ?? undefined}>
            <button
              type="button"
              className="btn btn--primary"
              disabled={Boolean(forbidden) || !modelId || startJob.isPending}
              onClick={() => launch('train', { model_id: modelId, dataset }, `Дообучение ${modelId} поставлено в очередь`)}
            >
              {forbidden ? <Lock size={15} /> : <GraduationCap size={15} />} Дообучить
            </button>
          </span>
          <button
            type="button"
            className="btn"
            disabled={startJob.isPending || models.length === 0}
            onClick={() => launch('gate', { version: models[0]?.version }, `Гейт для ${models[0]?.version} запущен`)}
            title="Сравнить новую версию с базовой на отложенных роликах и тесте наборов"
          >
            <Scale size={15} /> Запустить гейт
          </button>
        </div>
        {forbidden && (
          <p className="fw-note fw-note--lock">
            <Lock size={14} aria-hidden="true" /> {forbidden}
          </p>
        )}
      </section>

      {error ? (
        <div className="error-box">Карточки моделей не загрузились: {error.message}</div>
      ) : isLoading ? (
        <div className="panel empty">
          <span className="spinner" />
        </div>
      ) : models.length === 0 ? (
        <div className="panel empty">
          <Boxes size={30} />
          <span className="empty__title">Карточек моделей ещё нет</span>
          <span>Любое дообучение заканчивается карточкой ml/model_cards/&lt;version&gt;.json</span>
        </div>
      ) : (
        <div className="fw-models__grid">
          {models.map((card, i) => (
            <div key={card.version} className="reveal" style={{ ['--i' as string]: i + 1 }}>
              <ModelCardView
                card={card}
                forbidden={forbidden}
                deploying={startJob.isPending}
                onDeploy={(c) => launch('deploy', { version: c.version }, `Выкатка ${c.version} запущена`)}
              />
            </div>
          ))}
        </div>
      )}

      <section className="panel fw-jobs-panel">
        <div className="panel__head">
          <div>
            <h3 className="panel__title">Задачи</h3>
            <p className="panel__sub">CLI ml/flywheel/* подпроцессом; лог — data/flywheel/jobs/&lt;id&gt;.log</p>
          </div>
        </div>
        <div className="panel__body">
          <JobsList jobs={jobs} />
        </div>
      </section>
    </div>
  );
}
