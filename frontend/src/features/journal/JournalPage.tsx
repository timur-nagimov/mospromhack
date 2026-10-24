import { ChevronLeft, ChevronRight, Download, FilterX, MessageSquare, SearchX } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../../api';
import { useCameraMap, useCameras, useEvents, useRuleTypes, zoneName } from '../../api/queries';
import { EventThumb } from '../../components/media/EventThumb';
import { SeverityBadge, StatusBadge, ViolationIcon } from '../../components/ui/badges';
import { toast } from '../../components/ui/toast';
import { formatDate, formatSeconds, formatTime, fromLocalInputValue, plural, toLocalInputValue } from '../../domain/format';
import { SEVERITY_LABELS, SEVERITY_ORDER, STATUS_LABELS, STATUS_ORDER } from '../../domain/labels';
import { PERIOD_LABELS, periodRange, type PeriodPreset } from '../../domain/period';
import type { EventQuery, EventStatus, GuardEvent, Severity } from '../../domain/types';
import { getViolationMeta, VIOLATION_TYPES } from '../../domain/violations';
import { useNow } from '../../hooks/useNow';
import { useEventModal } from '../event/useEventModal';
import { downloadText, eventsToCsv } from './csv';
import './journal.css';

const PAGE_SIZES = [25, 50, 100];
const EXPORT_PAGE = 500;
const EXPORT_MAX = 20000;
const PRESETS: PeriodPreset[] = ['today', '24h', '7d', '30d', 'all', 'custom'];

export function JournalPage() {
  const [params, setParams] = useSearchParams();
  const now = useNow();
  const { data: cameras = [] } = useCameras();
  const cameraMap = useCameraMap();
  const { data: ruleTypes = [] } = useRuleTypes();
  const { openEvent } = useEventModal();
  const [exporting, setExporting] = useState<string | null>(null);

  const preset = (params.get('period') as PeriodPreset) || '7d';
  const filters = {
    type: params.get('type') ?? '',
    camera_id: params.get('camera') ?? '',
    zone_id: params.get('zone') ?? '',
    status: (params.get('status') ?? '') as EventStatus | '',
    severity: (params.get('severity') ?? '') as Severity | '',
  };
  // Параметры адреса могут быть отредактированы руками — читаем безопасно
  const numParam = (key: string): number | undefined => {
    const raw = params.get(key);
    const n = raw == null || raw === '' ? NaN : Number(raw);
    return Number.isFinite(n) ? n : undefined;
  };
  const customFrom = numParam('from');
  const customTo = numParam('to');
  const pageParam = Math.max(0, Math.floor(numParam('page') ?? 0));
  const pageSize = PAGE_SIZES.includes(Number(params.get('size'))) ? Number(params.get('size')) : 25;

  const minute = Math.floor(now / 60);
  const range = useMemo(
    () => periodRange(preset, minute * 60, { from: customFrom, to: customTo }),
    [preset, minute, customFrom, customTo],
  );

  const baseQuery: EventQuery = {
    type: filters.type || undefined,
    camera_id: filters.camera_id || undefined,
    zone_id: filters.zone_id || undefined,
    status: filters.status || undefined,
    severity: filters.severity || undefined,
    from: range.from,
    to: range.to,
  };
  // Страница за пределами выдачи (например, после смены фильтра) прижимается к последней
  const [knownTotal, setKnownTotal] = useState<number | null>(null);
  const lastPage = knownTotal != null ? Math.max(0, Math.ceil(knownTotal / pageSize) - 1) : Infinity;
  const page = Math.min(pageParam, lastPage);
  const query: EventQuery = { ...baseQuery, limit: pageSize, offset: page * pageSize };
  const { data, isLoading, isFetching, isError, error } = useEvents(query);
  useEffect(() => {
    if (data) setKnownTotal(data.total);
  }, [data]);

  // Бэкенд может не знать параметр severity — дофильтровываем на клиенте
  const items = useMemo(
    () => (data?.items ?? []).filter((e) => !filters.severity || e.severity === filters.severity),
    [data, filters.severity],
  );
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / pageSize));

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

  const typeOptions = useMemo(() => {
    const known = new Set<string>(VIOLATION_TYPES);
    const extra = ruleTypes.map((r) => r.type).filter((t) => !known.has(t));
    return [...VIOLATION_TYPES, ...extra];
  }, [ruleTypes]);

  const zoneOptions = useMemo(() => {
    const list = filters.camera_id ? cameras.filter((c) => c.camera_id === filters.camera_id) : cameras;
    return list.flatMap((c) => c.zones.map((z) => ({ id: z.zone_id, label: z.name, camera: c.name })));
  }, [cameras, filters.camera_id]);

  const activeFilters = Object.values(filters).filter(Boolean).length + (preset !== '7d' ? 1 : 0);

  const exportCsv = async () => {
    setExporting('0');
    try {
      // Фиксируем верхнюю границу: новые события во время выгрузки не сдвигают offset
      const pinnedTo = Math.min(baseQuery.to ?? Infinity, Math.floor(Date.now() / 1000));
      const byId = new Map<string, GuardEvent>();
      let offset = 0;
      let totalCount = Infinity;
      while (offset < totalCount && offset < EXPORT_MAX) {
        const pageData = await api().listEvents({ ...baseQuery, to: pinnedTo, limit: EXPORT_PAGE, offset });
        totalCount = pageData.total;
        for (const e of pageData.items) byId.set(e.id, e);
        offset += EXPORT_PAGE;
        setExporting(`${Math.min(offset, totalCount)} из ${totalCount}`);
        if (pageData.items.length === 0) break;
      }
      const rows = [...byId.values()].filter((e) => !filters.severity || e.severity === filters.severity);
      const stamp = toLocalInputValue(Date.now() / 1000).replace(/[:T]/g, '-');
      downloadText(`zhurnal-sobytiy-${stamp}.csv`, eventsToCsv(rows, cameraMap, window.location.origin));
      if (totalCount > EXPORT_MAX) {
        toast(`Выгружены первые ${rows.length} из ${totalCount} — сузьте период для полной выгрузки`, 'bad');
      } else {
        toast(`Выгружено ${rows.length} ${plural(rows.length, ['событие', 'события', 'событий'])}`);
      }
    } catch (err) {
      toast(`Экспорт не удался: ${(err as Error).message}`, 'bad');
    } finally {
      setExporting(null);
    }
  };

  return (
    <div className="page journal">
      <div className="filters reveal" role="search" aria-label="Фильтры журнала">
        <div className="field">
          <span className="field__label">Период</span>
          <div className="segmented">
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                aria-pressed={preset === p}
                onClick={() =>
                  update(
                    p === 'custom'
                      ? { period: p, from: range.from ?? Math.floor(now - 86400), to: range.to ?? Math.floor(now) }
                      : { period: p, from: undefined, to: undefined },
                  )
                }
              >
                {PERIOD_LABELS[p]}
              </button>
            ))}
          </div>
        </div>
        {preset === 'custom' && (
          <div className="filters__range">
            <label className="field">
              <span className="field__label">С</span>
              <input
                className="input"
                type="datetime-local"
                value={customFrom ? toLocalInputValue(customFrom) : ''}
                onChange={(e) => update({ from: fromLocalInputValue(e.target.value) })}
              />
            </label>
            <label className="field">
              <span className="field__label">По</span>
              <input
                className="input"
                type="datetime-local"
                value={customTo ? toLocalInputValue(customTo) : ''}
                onChange={(e) => update({ to: fromLocalInputValue(e.target.value) })}
              />
            </label>
          </div>
        )}
        <label className="field">
          <span className="field__label">Тип нарушения</span>
          <select className="select" value={filters.type} onChange={(e) => update({ type: e.target.value })}>
            <option value="">Все типы</option>
            {typeOptions.map((t) => (
              <option key={t} value={t}>
                {getViolationMeta(t).title}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Камера</span>
          <select className="select" value={filters.camera_id} onChange={(e) => update({ camera: e.target.value, zone: undefined })}>
            <option value="">Все камеры</option>
            {cameras.map((c) => (
              <option key={c.camera_id} value={c.camera_id}>
                {c.name} · {c.site}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Зона</span>
          <select className="select" value={filters.zone_id} onChange={(e) => update({ zone: e.target.value })}>
            <option value="">Все зоны</option>
            {zoneOptions.map((z) => (
              <option key={`${z.camera}-${z.id}`} value={z.id}>
                {filters.camera_id ? z.label : `${z.label} (${z.camera})`}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Статус</span>
          <select className="select" value={filters.status} onChange={(e) => update({ status: e.target.value })}>
            <option value="">Все статусы</option>
            {STATUS_ORDER.map((s) => (
              <option key={s} value={s}>
                {STATUS_LABELS[s]}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Важность</span>
          <select className="select" value={filters.severity} onChange={(e) => update({ severity: e.target.value })}>
            <option value="">Любая</option>
            {SEVERITY_ORDER.map((s) => (
              <option key={s} value={s}>
                {SEVERITY_LABELS[s]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <section className="panel journal__table reveal" style={{ ['--i' as string]: 1 }}>
        <div className="journal__summary">
          <span>
            Найдено <b className="num">{total}</b> {plural(total, ['событие', 'события', 'событий'])}
          </span>
          {isFetching && <span className="spinner" aria-label="Загрузка" />}
          <div className="filters__actions">
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              disabled={activeFilters === 0}
              onClick={() => setParams(new URLSearchParams(), { replace: true })}
            >
              <FilterX size={15} /> Сбросить фильтры
            </button>
            <button
              type="button"
              className="btn btn--primary btn--sm"
              onClick={exportCsv}
              disabled={exporting != null || total === 0}
              title="Все события по текущим фильтрам, CSV для Excel"
            >
              <Download size={15} />
              {exporting != null ? `Экспорт… ${exporting}` : 'Экспорт CSV'}
            </button>
          </div>
        </div>
        {isError ? (
          <div className="panel__body">
            <div className="error-box">Журнал не загрузился: {(error as Error).message}</div>
          </div>
        ) : !isLoading && items.length === 0 ? (
          <div className="empty">
            <SearchX size={28} />
            <span className="empty__title">Ничего не найдено</span>
            <span>Измените фильтры или период</span>
          </div>
        ) : (
          <div className="table-wrap loading-veil" data-loading={isFetching && !isLoading}>
            <table className="table">
              <thead>
                <tr>
                  <th>Кадр</th>
                  <th>Время</th>
                  <th>Нарушение</th>
                  <th>Важность</th>
                  <th>Камера · участок</th>
                  <th>Зона</th>
                  <th>Статус</th>
                  <th className="right">Длит.</th>
                  <th aria-label="Комментарий" />
                </tr>
              </thead>
              <tbody>
                {items.map((e) => {
                  const meta = getViolationMeta(e.type);
                  const camera = cameraMap.get(e.camera_id);
                  return (
                    <tr
                      key={e.id}
                      className="is-clickable"
                      tabIndex={0}
                      onClick={() => openEvent(e.id)}
                      onKeyDown={(ev) => {
                        if (ev.key === 'Enter') openEvent(e.id);
                      }}
                    >
                      <td className="journal__thumb">
                        <EventThumb event={e} width={84} />
                      </td>
                      <td>
                        <div className="journal__time">
                          <span className="mono">{formatTime(e.confirmed_at)}</span>
                          <span className="muted">{formatDate(e.confirmed_at)}</span>
                        </div>
                      </td>
                      <td>
                        <div className="journal__type">
                          <ViolationIcon type={e.type} severity={e.severity} size="sm" />
                          <span>{meta.title}</span>
                        </div>
                      </td>
                      <td>
                        <SeverityBadge severity={e.severity} />
                      </td>
                      <td>
                        <div className="journal__cam">
                          <span>{camera?.name ?? e.camera_id}</span>
                          <span className="muted">{e.site}</span>
                        </div>
                      </td>
                      <td className="dim">{zoneName(cameraMap, e.camera_id, e.zone_id) ?? <span className="muted">весь кадр</span>}</td>
                      <td>
                        <StatusBadge status={e.status} />
                      </td>
                      <td className="right dim">{e.ended_at ? formatSeconds(e.ended_at - e.started_at) : 'идёт'}</td>
                      <td className="journal__note">
                        {e.operator_note && (
                          <span title={e.operator_note}>
                            <MessageSquare size={15} aria-label={`Комментарий: ${e.operator_note}`} />
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <footer className="pager">
          <label className="pager__size">
            <span className="muted">Строк</span>
            <select className="select select--sm" value={pageSize} onChange={(e) => update({ size: e.target.value })}>
              {PAGE_SIZES.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <span className="pager__info num">
            {total === 0 ? '0' : `${page * pageSize + 1}–${Math.min(total, (page + 1) * pageSize)}`} из {total}
          </span>
          <div className="pager__nav">
            <button
              type="button"
              className="btn btn--sm btn--icon"
              disabled={page === 0}
              onClick={() => update({ page: page - 1 }, false)}
              aria-label="Предыдущая страница"
            >
              <ChevronLeft size={16} />
            </button>
            <span className="num">
              {page + 1} / {pages}
            </span>
            <button
              type="button"
              className="btn btn--sm btn--icon"
              disabled={page + 1 >= pages}
              onClick={() => update({ page: page + 1 }, false)}
              aria-label="Следующая страница"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </footer>
      </section>
    </div>
  );
}
