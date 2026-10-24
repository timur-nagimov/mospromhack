import { Camera, ChevronLeft, ChevronRight, FilterX, Inbox, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../../api';
import { isForbidden, useCaptureFrame, useFlywheelSample, useFlywheelSamples, useStartJob } from '../../api/flywheelQueries';
import { useCameraMap, useCameras } from '../../api/queries';
import { toast } from '../../components/ui/toast';
import {
  classLabel,
  DEFAULT_CLASSES,
  isHoldout,
  jobPhase,
  needsReview,
  SAMPLE_STATUS_LABELS,
  SAMPLE_STATUS_ORDER,
  sourceLabel,
  SOURCE_ORDER,
  type FlywheelJob,
  type FlywheelSample,
  type FlywheelSummary,
  type ModelCard,
  type SampleQuery,
} from '../../domain/flywheel';
import { formatNumber, formatSmartDateTime, plural } from '../../domain/format';
import { useLive } from '../../state/live';
import { LazyFrame } from './FrameBoxes';
import { HoldoutBadge, SampleStatusBadge, SourceTag } from './parts';
import { SampleCard } from './SampleCard';

const PAGE_SIZE = 50;

function QueueItem({
  sample,
  active,
  fresh,
  cameraName,
  onSelect,
}: {
  sample: FlywheelSample;
  active: boolean;
  fresh: boolean;
  cameraName: string;
  onSelect: () => void;
}) {
  const ref = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: 'nearest' });
  }, [active]);
  const boxes = (sample.labels?.length ? sample.labels : sample.detections ?? []).map((b, i) => ({
    key: `t${i}`,
    cls: b.cls,
    bbox: b.bbox,
    kind: sample.labels?.length ? ('label' as const) : ('detection' as const),
  }));
  return (
    <li>
      <button
        ref={ref}
        type="button"
        className={`fw-qitem${active ? ' is-active' : ''}${fresh ? ' is-fresh' : ''}${needsReview(sample) ? '' : ' is-done'}`}
        onClick={onSelect}
        aria-current={active ? 'true' : undefined}
      >
        <LazyFrame
          getSrc={() => api().flywheel.frameUrl(sample, 'thumb')}
          width={sample.width}
          height={sample.height}
          boxes={boxes}
          captions={false}
          alt=""
        />
        <span className="fw-qitem__body">
          <span className="fw-qitem__top">
            <SourceTag source={sample.source} />
            {fresh && <span className="fw-qitem__new">новый</span>}
          </span>
          <span className="fw-qitem__cam">
            {cameraName} · <span className="num">{formatSmartDateTime(sample.created_at)}</span>
          </span>
          <span className="fw-qitem__reason mono">{sample.reason}</span>
          <span className="fw-qitem__foot">
            <SampleStatusBadge status={sample.status} />
            {isHoldout(sample) && <HoldoutBadge />}
          </span>
        </span>
      </button>
    </li>
  );
}

export function ReviewQueue({
  summary,
  models,
  jobs,
}: {
  summary: FlywheelSummary | undefined;
  models: ModelCard[];
  jobs: FlywheelJob[];
}) {
  const [params, setParams] = useSearchParams();
  const { data: cameras = [] } = useCameras();
  const cameraMap = useCameraMap();
  const fresh = useLive((s) => s.flywheelFresh);
  const startJob = useStartJob();
  const capture = useCaptureFrame();

  const filters = {
    source: params.get('source') ?? '',
    status: params.get('status') ?? '',
    camera_id: params.get('camera') ?? '',
    cls: params.get('cls') ?? '',
    holdout: params.get('holdout') ?? '',
  };
  const pageRaw = Number(params.get('page'));
  const page = Number.isFinite(pageRaw) && pageRaw > 0 ? Math.floor(pageRaw) : 0;
  const query: SampleQuery = {
    source: filters.source || undefined,
    status: filters.status || undefined,
    camera_id: filters.camera_id || undefined,
    cls: filters.cls || undefined,
    holdout: filters.holdout === '1' ? 1 : filters.holdout === '0' ? 0 : undefined,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
    order: '-created_at',
  };
  const { data, isLoading, isFetching, isError, error } = useFlywheelSamples(query);
  const items = useMemo(() => data?.items ?? [], [data]);
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const selectedParam = params.get('sample');
  const selectedId = selectedParam ?? items[0]?.id ?? null;
  const listItem = items.find((s) => s.id === selectedId);
  const single = useFlywheelSample(listItem ? null : selectedId);
  const sample = listItem ?? single.data ?? null;
  const index = sample ? items.findIndex((s) => s.id === sample.id) : -1;

  const update = (patch: Record<string, string | number | undefined | null>, resetPage = true) => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        for (const [k, v] of Object.entries(patch)) {
          if (v === undefined || v === null || v === '') next.delete(k);
          else next.set(k, String(v));
        }
        if (resetPage && !('page' in patch)) next.delete('page');
        return next;
      },
      { replace: true },
    );
  };
  const select = (id: string) => update({ sample: id }, false);
  const goPage = (p: number) => update({ page: p > 0 ? p : undefined, sample: undefined }, false);

  const hasNext = index >= 0 ? index < items.length - 1 || page < pages - 1 : items.length > 0;
  const hasPrev = index > 0 || page > 0;
  const next = () => {
    if (index >= 0 && index < items.length - 1) select(items[index + 1].id);
    else if (index < 0 && items.length > 0) select(items[0].id);
    else if (page < pages - 1) goPage(page + 1);
  };
  const prev = () => {
    if (index > 0) select(items[index - 1].id);
    else if (page > 0) goPage(page - 1);
  };
  /** После решения — к следующему непроверенному ниже по списку, иначе просто к следующему */
  const advance = () => {
    const after = index >= 0 ? items.slice(index + 1) : items;
    const target = after.find((s) => needsReview(s));
    if (target) select(target.id);
    else next();
  };

  const classOptions = useMemo(() => {
    const set = new Set<string>(DEFAULT_CLASSES);
    for (const m of models) for (const c of m.classes ?? []) set.add(c);
    for (const c of Object.keys(summary?.by_class ?? {})) set.add(c);
    return [...set];
  }, [models, summary]);
  const filterClasses = useMemo(() => Object.keys(summary?.by_class ?? {}).sort(), [summary]);

  const cameraOptions = useMemo(() => {
    const ids = new Set(cameras.map((c) => c.camera_id));
    for (const s of items) ids.add(s.camera_id);
    return [...ids].map((id) => ({ id, name: cameraMap.get(id)?.name ?? id }));
  }, [cameras, cameraMap, items]);

  const activeFilters = Object.values(filters).filter(Boolean).length;
  const newCount = summary?.by_status?.new ?? 0;
  const autolabelRunning = jobs.some((j) => j.kind === 'autolabel' && jobPhase(j.status) === 'active');
  const [captureChoice, setCaptureChoice] = useState('');
  const captureCamera = captureChoice || cameras.find((c) => c.enabled)?.camera_id || '';

  const runAutolabel = () => {
    startJob.mutate(
      { kind: 'autolabel', args: {} },
      {
        onSuccess: () => toast('Авторазметка запущена: учитель размечает новые сэмплы'),
        onError: (err) =>
          toast(isForbidden(err) ? `Авторазметка запрещена: ${(err as Error).message}` : `Не запущено: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  const captureFrame = () => {
    if (!captureCamera) return;
    capture.mutate(captureCamera, {
      onSuccess: (s) => {
        toast(`Кадр ${cameraMap.get(captureCamera)?.name ?? captureCamera} добавлен в выборку`);
        select(s.id);
      },
      onError: (err) => toast(`Кадр не снят: ${(err as Error).message}`, 'bad'),
    });
  };

  return (
    <div className="fw-review">
      <div className="filters fw-filters" role="search" aria-label="Фильтры очереди">
        <label className="field">
          <span className="field__label">Источник</span>
          <select className="select" value={filters.source} onChange={(e) => update({ source: e.target.value, sample: undefined })}>
            <option value="">Все источники</option>
            {SOURCE_ORDER.map((s) => (
              <option key={s} value={s}>
                {sourceLabel(s)}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Статус</span>
          <select className="select" value={filters.status} onChange={(e) => update({ status: e.target.value, sample: undefined })}>
            <option value="">Все статусы</option>
            {SAMPLE_STATUS_ORDER.map((s) => (
              <option key={s} value={s}>
                {SAMPLE_STATUS_LABELS[s]}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Камера</span>
          <select className="select" value={filters.camera_id} onChange={(e) => update({ camera: e.target.value, sample: undefined })}>
            <option value="">Все камеры</option>
            {cameraOptions.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name === c.id ? c.id : `${c.name} · ${c.id}`}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Класс</span>
          <select className="select" value={filters.cls} onChange={(e) => update({ cls: e.target.value, sample: undefined })}>
            <option value="">Все классы</option>
            {filterClasses.map((c) => (
              <option key={c} value={c}>
                {classLabel(c) === c ? c : `${classLabel(c)} · ${c}`}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Выборка</span>
          <select className="select" value={filters.holdout} onChange={(e) => update({ holdout: e.target.value, sample: undefined })}>
            <option value="">Вся</option>
            <option value="0">Обучающая</option>
            <option value="1">Отложенная</option>
          </select>
        </label>
        {activeFilters > 0 && (
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => update({ source: undefined, status: undefined, camera: undefined, cls: undefined, holdout: undefined, sample: undefined })}
          >
            <FilterX size={15} /> Сбросить
          </button>
        )}
        <div className="filters__actions fw-filters__actions">
          <button type="button" className="btn" onClick={runAutolabel} disabled={startJob.isPending || autolabelRunning} title="Учитель (YOLOv8x и YOLO-World) размечает новые сэмплы">
            {autolabelRunning ? <span className="spinner fw-spinner-sm" /> : <Sparkles size={15} />}
            {autolabelRunning ? 'Идёт авторазметка' : `Авторазметка${newCount ? ` (${formatNumber(newCount, 0)})` : ''}`}
          </button>
          <div className="fw-capture">
            <select
              className="select"
              value={captureCamera}
              onChange={(e) => setCaptureChoice(e.target.value)}
              aria-label="Камера для кадра в выборку"
            >
              {cameras.map((c) => (
                <option key={c.camera_id} value={c.camera_id} disabled={!c.enabled}>
                  {c.name}
                </option>
              ))}
            </select>
            <button type="button" className="btn" onClick={captureFrame} disabled={!captureCamera || capture.isPending} title="Текущий кадр камеры в выборку (source = operator)">
              <Camera size={15} /> Кадр в выборку
            </button>
          </div>
        </div>
      </div>

      <div className="fw-review__body">
        <section className="panel fw-queue" aria-label="Очередь проверки">
          <div className="fw-queue__summary">
            <span>
              <b className="num">{formatNumber(total, 0)}</b> {plural(total, ['сэмпл', 'сэмпла', 'сэмплов'])}
            </span>
            {isFetching && <span className="spinner fw-spinner-sm" aria-label="Обновление" />}
            <span className="fw-queue__keys">
              <span className="kbd">1</span> принять <span className="kbd">2</span> отклонить <span className="kbd">←</span>
              <span className="kbd">→</span>
            </span>
          </div>
          {isError ? (
            <div className="error-box fw-queue__error">Очередь не загрузилась: {(error as Error)?.message}</div>
          ) : isLoading ? (
            <div className="empty">
              <span className="spinner" />
            </div>
          ) : items.length === 0 ? (
            <div className="empty">
              <Inbox size={30} />
              <span className="empty__title">{activeFilters ? 'Под фильтр ничего не попало' : 'Выборка пока пуста'}</span>
              <span>{activeFilters ? 'Сбросьте фильтры или выберите другой статус' : 'Сэмплы появятся с первыми событиями и неуверенными кадрами'}</span>
            </div>
          ) : (
            <ul className="fw-queue__list loading-veil" data-loading={isFetching && !isLoading}>
              {items.map((s) => (
                <QueueItem
                  key={s.id}
                  sample={s}
                  active={s.id === sample?.id}
                  fresh={fresh.has(s.id)}
                  cameraName={cameraMap.get(s.camera_id)?.name ?? s.camera_id}
                  onSelect={() => select(s.id)}
                />
              ))}
            </ul>
          )}
          <div className="pager fw-queue__pager">
            <span className="pager__info num">
              {total > 0 ? `${page * PAGE_SIZE + 1}–${Math.min(total, (page + 1) * PAGE_SIZE)} из ${total}` : '0 из 0'}
            </span>
            <div className="pager__nav">
              <button type="button" className="btn btn--sm btn--icon" onClick={() => goPage(page - 1)} disabled={page === 0} aria-label="Предыдущая страница">
                <ChevronLeft size={16} />
              </button>
              <span className="num">
                {page + 1} / {pages}
              </span>
              <button type="button" className="btn btn--sm btn--icon" onClick={() => goPage(page + 1)} disabled={page >= pages - 1} aria-label="Следующая страница">
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </section>

        {sample ? (
          <SampleCard
            key={sample.id}
            sample={sample}
            classes={classOptions}
            hasPrev={hasPrev}
            hasNext={hasNext}
            onPrev={prev}
            onNext={next}
            onDecided={advance}
          />
        ) : (
          <section className="panel fw-card fw-card--empty">
            {single.isError ? (
              <div className="error-box">Сэмпл не загрузился: {(single.error as Error)?.message}</div>
            ) : single.isLoading ? (
              <span className="spinner" />
            ) : (
              <div className="empty">
                <Inbox size={30} />
                <span className="empty__title">Выберите сэмпл слева</span>
                <span>
                  Клавиши: <span className="kbd">1</span> принять, <span className="kbd">2</span> отклонить, <span className="kbd">←</span>
                  <span className="kbd">→</span> соседний сэмпл
                </span>
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
