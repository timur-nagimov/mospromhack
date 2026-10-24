// Верх страницы «Данные»: полоса цикла из шести шагов и счётчики

import { ChevronRight } from 'lucide-react';
import type { ReactNode } from 'react';
import {
  datasetVersionOf,
  jobPhase,
  modelVersions,
  SOURCE_COLORS,
  SOURCE_LABELS,
  SOURCE_ORDER,
  toEpoch,
  type DatasetVersion,
  type FlywheelJob,
  type FlywheelSummary,
  type ModelCard,
} from '../../domain/flywheel';
import { formatNumber, formatPercent, formatRelative, nowSec } from '../../domain/format';
import { useNow } from '../../hooks/useNow';
import { useLive } from '../../state/live';

export type DataTab = 'review' | 'datasets' | 'models';

type StepState = 'active' | 'ok' | 'warn' | 'bad' | 'idle';

interface Step {
  n: number;
  title: string;
  value: ReactNode;
  note: ReactNode;
  state: StepState;
  tab: DataTab;
}

const STATE_TEXT: Record<StepState, string> = {
  active: 'идёт',
  ok: 'готово',
  warn: 'ждёт',
  bad: 'ошибка',
  idle: 'нет данных',
};

function lastJob(jobs: FlywheelJob[], kinds: string[]): FlywheelJob | undefined {
  return jobs.find((j) => kinds.includes(j.kind));
}

function jobState(job: FlywheelJob | undefined, fallback: StepState): StepState {
  if (!job) return fallback;
  const phase = jobPhase(job.status);
  return phase === 'active' ? 'active' : phase === 'failed' ? 'bad' : phase === 'done' ? 'ok' : fallback;
}

function jobWhen(job: FlywheelJob | undefined, now: number): string | null {
  const at = toEpoch(job?.finished_at ?? job?.started_at ?? job?.created_at);
  return at ? formatRelative(at, now) : null;
}

export function CycleStrip({
  summary,
  datasets,
  models,
  jobs,
  trainForbidden,
  onOpen,
}: {
  summary: FlywheelSummary | undefined;
  datasets: DatasetVersion[];
  models: ModelCard[];
  jobs: FlywheelJob[];
  trainForbidden: string | null;
  onOpen: (tab: DataTab) => void;
}) {
  const now = useNow();
  const lastAt = useLive((s) => s.flywheelLastAt);
  const byStatus = summary?.by_status ?? {};
  const queue = (byStatus.new ?? 0) + (byStatus.auto ?? 0);
  const autolabel = lastJob(jobs, ['autolabel']);
  const train = lastJob(jobs, ['train']);
  const gate = lastJob(jobs, ['gate', 'deploy']);
  const lastDataset = datasets[0] ?? (typeof summary?.last_dataset === 'object' ? summary?.last_dataset : null) ?? null;
  const lastVersion = lastDataset?.version ?? datasetVersionOf(summary?.last_dataset);
  const newest = models[0];
  const deployed = modelVersions(summary?.models?.deployed);
  const deployedList = deployed.length ? deployed : models.filter((m) => m.status === 'deployed').map((m) => m.version);
  const lastGated = models.find((m) => m.gate);

  const steps: Step[] = [
    {
      n: 1,
      title: 'Сбор',
      value: <>{formatNumber(summary?.today ?? 0, 0)} <small>сегодня</small></>,
      note: (
        <>
          всего {formatNumber(summary?.total ?? 0, 0)}
          {lastAt ? ` · последний ${formatRelative(lastAt / 1000, now)}` : ''}
        </>
      ),
      state: summary ? (summary.collecting ? 'active' : 'idle') : 'idle',
      tab: 'review',
    },
    {
      n: 2,
      title: 'Авторазметка',
      value: <>{formatNumber(byStatus.auto ?? 0, 0)} <small>размечено</small></>,
      note: (
        <>
          ждут учителя: {formatNumber(byStatus.new ?? 0, 0)}
          {autolabel ? ` · запуск ${jobWhen(autolabel, now) ?? ''}` : ''}
        </>
      ),
      state: jobState(autolabel, (byStatus.new ?? 0) > 0 ? 'warn' : 'idle'),
      tab: 'review',
    },
    {
      n: 3,
      title: 'Проверка',
      value: <>{formatNumber(summary?.reviewed ?? 0, 0)} <small>проверено</small></>,
      note: <>в очереди {formatNumber(queue, 0)}</>,
      state: !summary ? 'idle' : queue > 0 ? 'warn' : 'ok',
      tab: 'review',
    },
    {
      n: 4,
      title: 'Датасет',
      value: lastVersion ? <span className="mono fw-step__version">{lastVersion}</span> : '—',
      note: lastDataset
        ? `train ${formatNumber(lastDataset.n_train ?? 0, 0)} · val ${formatNumber(lastDataset.n_val ?? 0, 0)}`
        : 'версий ещё нет',
      state: jobState(lastJob(jobs, ['export']), lastVersion ? 'ok' : 'idle'),
      tab: 'datasets',
    },
    {
      n: 5,
      title: 'Дообучение и сборка',
      value: newest ? <span className="mono fw-step__version">{newest.version}</span> : '—',
      note: trainForbidden
        ? 'обучение из UI выключено'
        : train
          ? `обучение ${jobWhen(train, now) ?? ''}`
          : newest
            ? newest.tpu_program
              ? 'под LinQ: программа TPU собрана'
              : 'под LinQ: программы TPU нет'
            : 'карточек нет',
      state: jobState(train, newest ? (newest.tpu_program ? 'ok' : 'warn') : 'idle'),
      tab: 'models',
    },
    {
      n: 6,
      title: 'Гейт и выкатка',
      value: deployedList.length ? <span className="mono fw-step__version">{deployedList.join(', ')}</span> : '—',
      note: lastGated
        ? `гейт: ${lastGated.version} — ${lastGated.gate?.promote ? 'пройден' : 'не пройден'}`
        : 'гейт не запускали',
      state: jobState(gate, lastGated ? (lastGated.gate?.promote ? 'ok' : 'bad') : 'idle'),
      tab: 'models',
    },
  ];

  return (
    <ol className="fw-cycle reveal" aria-label="Цикл маховика данных">
      {steps.map((step) => (
        <li key={step.n} className={`fw-step fw-step--${step.state}`}>
          <button type="button" className="fw-step__btn" onClick={() => onOpen(step.tab)} title={`${step.title}: ${STATE_TEXT[step.state]}`}>
            <span className="fw-step__head">
              <span className="fw-step__n mono">{step.n}</span>
              <span className="fw-step__title">{step.title}</span>
              <span className="fw-step__dot" aria-label={STATE_TEXT[step.state]} />
            </span>
            <span className="fw-step__value">{step.value}</span>
            <span className="fw-step__note">{step.note}</span>
          </button>
          {step.n < 6 && <ChevronRight className="fw-step__arrow" size={16} aria-hidden="true" />}
        </li>
      ))}
    </ol>
  );
}

export function Counters({ summary, datasets }: { summary: FlywheelSummary | undefined; datasets: DatasetVersion[] }) {
  const total = summary?.total ?? 0;
  const bySource = summary?.by_source ?? {};
  const sources = SOURCE_ORDER.filter((s) => s !== 'import' || (bySource[s] ?? 0) > 0);
  const sourceTotal = sources.reduce((sum, s) => sum + (bySource[s] ?? 0), 0);
  const lastDataset = datasets[0] ?? (typeof summary?.last_dataset === 'object' ? summary?.last_dataset : null) ?? null;
  const inDataset = lastDataset ? (lastDataset.n_train ?? 0) + (lastDataset.n_val ?? 0) : null;
  const reviewedShare = total > 0 ? (summary?.reviewed ?? 0) / total : 0;
  const created = toEpoch(lastDataset?.created_at);
  const dropped = summary?.dropped ?? 0;

  return (
    <div className="fw-counters reveal" style={{ ['--i' as string]: 1 }}>
      <div className="stat">
        <span className="stat__label">Собрано сегодня</span>
        <span className="stat__value num">{formatNumber(summary?.today ?? 0, 0)}</span>
        <span className="stat__note">
          всего {formatNumber(total, 0)} · отложенная выборка {formatNumber(summary?.holdout ?? 0, 0)}
        </span>
      </div>

      <div className="stat fw-sources">
        <span className="stat__label">По источникам</span>
        <div className="fw-sources__bar" aria-hidden="true">
          {sources.map((s) =>
            (bySource[s] ?? 0) > 0 ? (
              <span key={s} style={{ flexGrow: bySource[s] ?? 0, background: SOURCE_COLORS[s] }} title={`${SOURCE_LABELS[s]}: ${bySource[s]}`} />
            ) : null,
          )}
          {sourceTotal === 0 && <span className="fw-sources__empty" />}
        </div>
        <ul className="fw-sources__legend">
          {sources.map((s) => (
            <li key={s}>
              <i style={{ background: SOURCE_COLORS[s] }} />
              <span>{SOURCE_LABELS[s]}</span>
              <b className="num">{formatNumber(bySource[s] ?? 0, 0)}</b>
            </li>
          ))}
        </ul>
      </div>

      <div className="stat">
        <span className="stat__label">Проверено человеком</span>
        <span className="stat__value num">{formatNumber(summary?.reviewed ?? 0, 0)}</span>
        <div className="meter" aria-hidden="true">
          <span style={{ width: `${Math.min(100, reviewedShare * 100)}%` }} />
        </div>
        <span className="stat__note">{formatPercent(reviewedShare)} собранного</span>
      </div>

      <div className="stat">
        <span className="stat__label">В последней версии датасета</span>
        <span className="stat__value num">{inDataset != null ? formatNumber(inDataset, 0) : '—'}</span>
        <span className="stat__note">
          {lastDataset ? (
            <>
              <span className="mono">{lastDataset.version}</span>
              {created ? ` · ${formatRelative(created, nowSec())}` : ''}
            </>
          ) : (
            'версий ещё нет'
          )}
        </span>
      </div>

      <div className={`stat${dropped > 0 ? ' fw-stat--warn' : ''}`}>
        <span className="stat__label">Отброшено</span>
        <span className="stat__value num">{formatNumber(dropped, 0)}</span>
        <span className="stat__note">очередь записи переполнена — оповещения не ждут диск</span>
      </div>
    </div>
  );
}
