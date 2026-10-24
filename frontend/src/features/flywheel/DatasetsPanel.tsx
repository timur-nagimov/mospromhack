import { Database, ExternalLink, PackagePlus, TriangleAlert } from 'lucide-react';
import { useState } from 'react';
import { useBuildDataset } from '../../api/flywheelQueries';
import { toast } from '../../components/ui/toast';
import { classLabel, manifestSources, toEpoch, type DatasetVersion, type FlywheelSummary } from '../../domain/flywheel';
import { formatNumber, formatSmartDateTime } from '../../domain/format';

export function DatasetsPanel({
  summary,
  datasets,
  isLoading,
  error,
}: {
  summary: FlywheelSummary | undefined;
  datasets: DatasetVersion[];
  isLoading: boolean;
  error: Error | null;
}) {
  const build = useBuildDataset();
  const [includeHoldout, setIncludeHoldout] = useState(false);
  const accepted = summary?.by_status?.accepted ?? 0;
  const holdout = summary?.holdout ?? 0;

  const run = () => {
    build.mutate(
      { include_holdout: includeHoldout },
      {
        onSuccess: (ds) =>
          toast(`Версия ${ds.version}: train ${formatNumber(ds.n_train ?? 0, 0)}, val ${formatNumber(ds.n_val ?? 0, 0)}`),
        onError: (err) => toast(`Версия не собрана: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  return (
    <div className="fw-datasets">
      <section className="panel fw-build reveal">
        <div className="panel__head">
          <div>
            <h3 className="panel__title">Новая версия датасета</h3>
            <p className="panel__sub">YOLO-формат, разбиение по камере и времени, манифест с источниками и sha256</p>
          </div>
        </div>
        <div className="panel__body fw-build__body">
          <ul className="fw-build__facts">
            <li>
              <b className="num">{formatNumber(accepted, 0)}</b> принятых сэмплов: только они и открытые наборы с готовой разметкой
            </li>
            <li>
              <b className="num">{formatNumber(holdout, 0)}</b> в отложенной выборке: {includeHoldout ? 'войдут в датасет' : 'не войдут'}
            </li>
          </ul>
          <label className="switch">
            <input type="checkbox" checked={includeHoldout} onChange={(e) => setIncludeHoldout(e.target.checked)} />
            <span className="switch__track" />
            Включить кадры отложенных роликов
          </label>
          {includeHoldout && (
            <p className="fw-note fw-note--warn">
              <TriangleAlert size={14} aria-hidden="true" /> Оценка на этих роликах станет загрязнённой — отчёт гейта это пометит.
            </p>
          )}
          <button type="button" className="btn btn--primary" onClick={run} disabled={build.isPending}>
            {build.isPending ? <span className="spinner fw-spinner-sm fw-spinner-dark" /> : <PackagePlus size={16} />}
            {build.isPending ? 'Собираем…' : 'Собрать версию'}
          </button>
        </div>
      </section>

      <section className="panel fw-versions reveal" style={{ ['--i' as string]: 1 }}>
        <div className="panel__head">
          <div>
            <h3 className="panel__title">Версии</h3>
            <p className="panel__sub">Новые сверху; по версии видно, на чём учили каждую модель</p>
          </div>
        </div>
        {error ? (
          <div className="panel__body">
            <div className="error-box">Список версий не загрузился: {error.message}</div>
          </div>
        ) : isLoading ? (
          <div className="empty">
            <span className="spinner" />
          </div>
        ) : datasets.length === 0 ? (
          <div className="empty">
            <Database size={30} />
            <span className="empty__title">Версий ещё нет</span>
            <span>Примите сэмплы в очереди и соберите первую версию</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table fw-versions__table">
              <thead>
                <tr>
                  <th>Версия</th>
                  <th>Создана</th>
                  <th className="right">Train</th>
                  <th className="right">Val</th>
                  <th className="right">Hard negative</th>
                  <th>Классы</th>
                  <th>Состав</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds, i) => {
                  const created = toEpoch(ds.created_at);
                  const sources = manifestSources(ds.manifest);
                  const includes = Boolean(ds.manifest && (ds.manifest as Record<string, unknown>).include_holdout);
                  return (
                    <tr key={ds.version} className={i === 0 ? 'fw-versions__last' : undefined}>
                      <td>
                        <span className="mono fw-versions__ver">{ds.version}</span>
                        {i === 0 && <span className="badge badge--plain">последняя</span>}
                        {includes && (
                          <span className="badge fw-holdout" title="В версию вошли кадры отложенных роликов">
                            с отложенными
                          </span>
                        )}
                        {ds.path && <div className="mono muted fw-versions__path">{ds.path}</div>}
                      </td>
                      <td className="num">{created ? formatSmartDateTime(created) : '—'}</td>
                      <td className="right num">{formatNumber(ds.n_train ?? 0, 0)}</td>
                      <td className="right num">{formatNumber(ds.n_val ?? 0, 0)}</td>
                      <td className="right num">{formatNumber(ds.n_hard_negative ?? 0, 0)}</td>
                      <td>
                        <div className="fw-chips">
                          {(ds.classes ?? []).map((c) => (
                            <span key={c} className="fw-chip" title={c}>
                              {classLabel(c)}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td>
                        {sources.length === 0 ? (
                          <span className="muted">—</span>
                        ) : (
                          <ul className="fw-mix">
                            {sources.map((src) => (
                              <li key={src.name}>
                                {src.url ? (
                                  <a href={src.url} target="_blank" rel="noreferrer noopener">
                                    {src.name} <ExternalLink size={11} aria-hidden="true" />
                                  </a>
                                ) : (
                                  <span>{src.name}</span>
                                )}
                                {src.count != null && <b className="num">{formatNumber(src.count, 0)}</b>}
                              </li>
                            ))}
                          </ul>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
