import { Boxes, Database, ListChecks, PlugZap } from 'lucide-react';
import { useSearchParams } from 'react-router-dom';
import { isNotFound, useDatasets, useFlywheelSummary, useJobs, useModelCards } from '../../api/flywheelQueries';
import { formatNumber } from '../../domain/format';
import { DatasetsPanel } from './DatasetsPanel';
import { ModelsPanel } from './ModelsPanel';
import { Counters, CycleStrip, type DataTab } from './Overview';
import { useTrainForbidden } from './parts';
import { ReviewQueue } from './ReviewQueue';
import './flywheel.css';

const TABS: DataTab[] = ['review', 'datasets', 'models'];

/**
 * «Данные» — маховик данных (docs/FLYWHEEL.md, раздел UI): цикл сбор → авторазметка →
 * проверка → датасет → дообучение и сборка под LinQ → гейт и выкатка.
 */
export function DataPage() {
  const [params, setParams] = useSearchParams();
  const tabParam = params.get('tab') as DataTab | null;
  const tab: DataTab = tabParam && TABS.includes(tabParam) ? tabParam : 'review';

  const summaryQuery = useFlywheelSummary();
  const datasetsQuery = useDatasets();
  const modelsQuery = useModelCards();
  const jobsQuery = useJobs();
  const summary = summaryQuery.data;
  const datasets = datasetsQuery.data ?? [];
  const models = modelsQuery.data ?? [];
  const jobs = jobsQuery.data ?? [];
  const trainForbidden = useTrainForbidden(summary?.allow_train);
  const apiMissing = isNotFound(summaryQuery.error);
  const queue = (summary?.by_status?.new ?? 0) + (summary?.by_status?.auto ?? 0);

  const openTab = (next: DataTab) => {
    setParams(
      (prev) => {
        const p = new URLSearchParams(prev);
        if (next === 'review') p.delete('tab');
        else p.set('tab', next);
        return p;
      },
      { replace: true },
    );
  };

  return (
    <div className="page fw-page">
      {apiMissing ? (
        <div className="fw-missing reveal" role="status">
          <PlugZap size={20} aria-hidden="true" />
          <div>
            <strong>API маховика на сервере нет</strong>
            <p>
              GET /api/flywheel/summary ответил 404: бэкенд собран без guard/api/flywheel.py. Страница заработает после обновления
              шлюза; посмотреть её на демо-данных можно по адресу <span className="mono">?mock=1</span>.
            </p>
          </div>
        </div>
      ) : summaryQuery.isError ? (
        <div className="error-box reveal">Сводка маховика не загрузилась: {(summaryQuery.error as Error)?.message}</div>
      ) : null}

      <CycleStrip summary={summary} datasets={datasets} models={models} jobs={jobs} trainForbidden={trainForbidden} onOpen={openTab} />
      <Counters summary={summary} datasets={datasets} />

      <div className="tabs fw-tabs" role="tablist" aria-label="Разделы маховика">
        <button type="button" role="tab" aria-selected={tab === 'review'} onClick={() => openTab('review')}>
          <ListChecks size={15} /> Очередь проверки <span className="count">{formatNumber(queue, 0)}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === 'datasets'} onClick={() => openTab('datasets')}>
          <Database size={15} /> Датасеты <span className="count">{datasets.length}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === 'models'} onClick={() => openTab('models')}>
          <Boxes size={15} /> Модели <span className="count">{models.length}</span>
        </button>
      </div>

      {tab === 'review' && <ReviewQueue summary={summary} models={models} jobs={jobs} />}
      {tab === 'datasets' && (
        <DatasetsPanel
          summary={summary}
          datasets={datasets}
          isLoading={datasetsQuery.isLoading}
          error={datasetsQuery.error && !isNotFound(datasetsQuery.error) ? (datasetsQuery.error as Error) : null}
        />
      )}
      {tab === 'models' && (
        <ModelsPanel
          summary={summary}
          models={models}
          datasets={datasets}
          jobs={jobs}
          isLoading={modelsQuery.isLoading}
          error={modelsQuery.error && !isNotFound(modelsQuery.error) ? (modelsQuery.error as Error) : null}
        />
      )}
    </div>
  );
}
