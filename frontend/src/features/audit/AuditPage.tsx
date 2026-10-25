import { Cctv, ChevronLeft, ChevronRight, DatabaseZap, FilterX, KeyRound, ScrollText, SearchX, UserCog, type LucideIcon } from 'lucide-react';
import { useMemo } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useAudit, useUsers } from '../../api/authQueries';
import {
  AUDIT_ACTIONS,
  AUDIT_GROUP_TITLES,
  auditActionMeta,
  roleTitle,
  type AuditEntry,
  type AuditGroup,
  type AuditQuery,
} from '../../domain/auth';
import { formatDate, formatTime, fromLocalInputValue, plural, toLocalInputValue } from '../../domain/format';
import { STATUS_LABELS } from '../../domain/labels';
import { PERIOD_LABELS, periodRange, type PeriodPreset } from '../../domain/period';
import type { EventStatus } from '../../domain/types';
import { useNow } from '../../hooks/useNow';
import { useAuth, useCan } from '../../state/auth';
import { AccessLevel } from '../auth/parts';
import { useEventModal } from '../event/useEventModal';
import '../journal/journal.css';
import './audit.css';

const PRESETS: PeriodPreset[] = ['24h', '7d', '30d', 'all', 'custom'];
const PAGE_SIZES = [25, 50, 100];
const DEFAULT_PRESET: PeriodPreset = '7d';

const GROUP_ICON: Record<AuditGroup, LucideIcon> = {
  auth: KeyRound,
  cameras: Cctv,
  events: ScrollText,
  users: UserCog,
  flywheel: DatabaseZap,
};

const DETAIL_LABELS: Record<string, string> = {
  status: 'статус',
  operator_note: 'заметка',
  role: 'роль',
  active: 'активна',
  display_name: 'имя',
  password: 'пароль',
  zones: 'зоны',
  rules: 'правила',
  rule_id: 'правило',
  name: 'название',
  source: 'источник',
  kind: 'задача',
  model_id: 'модель',
  version: 'версия',
  min_duration_s: 'мин. длительность, с',
  active_hours: 'расписание',
  fields: 'поля',
};

const SAMPLE_STATUS: Record<string, string> = { accepted: 'принят', rejected: 'отклонён', auto: 'авторазметка', new: 'новый' };

function detailValue(key: string, value: unknown, action: string): string {
  if (value == null) return '—';
  if (key === 'status' && typeof value === 'string') {
    if (action.startsWith('flywheel')) return SAMPLE_STATUS[value] ?? value;
    return STATUS_LABELS[value as EventStatus] ?? value;
  }
  if (key === 'role' && typeof value === 'string') return roleTitle(value);
  if (typeof value === 'boolean') return value ? 'да' : 'нет';
  if (Array.isArray(value)) return value.map((v) => (typeof v === 'object' ? JSON.stringify(v) : String(v))).join(', ');
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

/** Подробности записи: {status: "confirmed"} → «статус: Подтверждено» */
export function describeAuditDetails(entry: AuditEntry): string {
  const { details } = entry;
  if (details == null || details === '') return '';
  if (typeof details !== 'object') return String(details);
  const text = Object.entries(details as Record<string, unknown>)
    .map(([k, v]) => `${DETAIL_LABELS[k] ?? k}: ${detailValue(k, v, entry.action)}`)
    .join(' · ');
  return text.length > 180 ? `${text.slice(0, 177)}…` : text;
}

function Target({ entry }: { entry: AuditEntry }) {
  const { openEvent } = useEventModal();
  const can = useCan();
  const target = entry.target;
  if (!target) return <span className="muted">—</span>;
  if (entry.action === 'event.update' && can('events.view')) {
    return (
      <button type="button" className="audit-link mono" onClick={() => openEvent(target)} title="Открыть карточку события">
        {target}
      </button>
    );
  }
  if (/^(camera|zones|rules)\./.test(entry.action) && entry.action !== 'camera.delete' && can('cameras.view')) {
    return (
      <Link className="audit-link mono" to={`/cameras/${encodeURIComponent(target)}`}>
        {target}
      </Link>
    );
  }
  return <span className="mono">{target}</span>;
}

/** «Журнал действий» (audit.view): кто, когда и что изменил, с какого адреса */
export function AuditPage() {
  const [params, setParams] = useSearchParams();
  const now = useNow();
  const can = useCan();
  const demoUsers = useAuth((s) => s.config?.demo_users ?? []);
  // имена — из списка учётных записей, если он доступен роли (users.manage)
  const { data: users = [] } = useUsers();
  const names = useMemo(() => new Map(users.map((u) => [u.username, u.display_name])), [users]);

  const preset = (params.get('period') as PeriodPreset | null) ?? DEFAULT_PRESET;
  const customFrom = params.get('from') ? Number(params.get('from')) : undefined;
  const customTo = params.get('to') ? Number(params.get('to')) : undefined;
  const range = periodRange(preset, now, { from: customFrom, to: customTo });
  const username = params.get('user') ?? '';
  const action = params.get('action') ?? '';
  const pageSize = PAGE_SIZES.includes(Number(params.get('size'))) ? Number(params.get('size')) : 50;
  const page = Math.max(0, Number(params.get('page') ?? 0) || 0);

  const query: AuditQuery = {
    username: username || undefined,
    action: action || undefined,
    from: range.from,
    to: range.to,
    limit: pageSize,
    offset: page * pageSize,
  };
  const { data, isLoading, isFetching, isError, error } = useAudit(query);
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const pages = Math.max(1, Math.ceil(total / pageSize));

  const knownUsers = useMemo(() => {
    const set = new Set<string>([...demoUsers.map((u) => u.username), ...users.map((u) => u.username)]);
    for (const e of items) if (e.username) set.add(e.username);
    return [...set].sort();
  }, [demoUsers, users, items]);

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

  const actionGroups = useMemo(() => {
    const groups = new Map<AuditGroup, string[]>();
    for (const [code, meta] of Object.entries(AUDIT_ACTIONS)) groups.set(meta.group, [...(groups.get(meta.group) ?? []), code]);
    return [...groups.entries()];
  }, []);

  const activeFilters = [username, action].filter(Boolean).length + (preset !== DEFAULT_PRESET ? 1 : 0);

  return (
    <div className="page journal audit">
      <div className="filters reveal" role="search" aria-label="Фильтры журнала действий">
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
                      : { period: p === DEFAULT_PRESET ? undefined : p, from: undefined, to: undefined },
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
          <span className="field__label">Пользователь</span>
          <select className="select" value={username} onChange={(e) => update({ user: e.target.value })}>
            <option value="">Все</option>
            {knownUsers.map((u) => (
              <option key={u} value={u}>
                {names.get(u) ? `${names.get(u)} · ${u}` : u}
              </option>
            ))}
          </select>
        </label>
        <label className="field audit__action-filter">
          <span className="field__label">Действие</span>
          <select className="select" value={action} onChange={(e) => update({ action: e.target.value })}>
            <option value="">Все действия</option>
            {actionGroups.map(([group, codes]) => (
              <optgroup key={group} label={AUDIT_GROUP_TITLES[group]}>
                {codes.map((code) => (
                  <option key={code} value={code}>
                    {AUDIT_ACTIONS[code].title}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </label>
      </div>

      <section className="panel journal__table reveal" style={{ ['--i' as string]: 1 }}>
        <div className="journal__summary">
          <span>
            Найдено <b className="num">{total}</b> {plural(total, ['запись', 'записи', 'записей'])}
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
          </div>
        </div>
        {isError ? (
          <div className="panel__body">
            <div className="error-box">Журнал действий не загрузился: {(error as Error).message}</div>
          </div>
        ) : !isLoading && items.length === 0 ? (
          <div className="empty">
            <SearchX size={28} />
            <span className="empty__title">Записей нет</span>
            <span>Измените фильтры или период</span>
          </div>
        ) : (
          <div className="table-wrap loading-veil" data-loading={isFetching && !isLoading}>
            <table className="table audit__table">
              <thead>
                <tr>
                  <th>Время</th>
                  <th>Пользователь</th>
                  <th>Действие</th>
                  <th>Объект</th>
                  <th>Подробности</th>
                  <th className="right">Адрес</th>
                </tr>
              </thead>
              <tbody>
                {items.map((e, i) => {
                  const meta = auditActionMeta(e.action);
                  const Icon = GROUP_ICON[meta.group];
                  const details = describeAuditDetails(e);
                  return (
                    <tr key={`${e.ts}-${e.action}-${i}`}>
                      <td>
                        <div className="journal__time">
                          <span className="mono">{formatTime(e.ts)}</span>
                          <span className="muted">{formatDate(e.ts)}</span>
                        </div>
                      </td>
                      <td>
                        {e.username ? (
                          <div className="audit-user">
                            <span>{names.get(e.username) ?? e.username}</span>
                            <span className="audit-user__role muted">
                              {e.role && <AccessLevel role={e.role} />}
                              {e.role ? roleTitle(e.role) : 'неизвестный логин'}
                              {names.has(e.username) && <span className="mono"> · {e.username}</span>}
                            </span>
                          </div>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                      <td>
                        <div className={`audit-action${meta.tone === 'bad' ? ' audit-action--bad' : ''}`}>
                          <span className="audit-action__icon" aria-hidden="true">
                            <Icon size={14} />
                          </span>
                          <span className="audit-action__text">
                            <span>{meta.title}</span>
                            <code className="mono">{e.action}</code>
                          </span>
                        </div>
                      </td>
                      <td>
                        <Target entry={e} />
                      </td>
                      <td className="audit__details" title={details || undefined}>
                        {details || <span className="muted">—</span>}
                      </td>
                      <td className="right mono dim">{e.ip ?? '—'}</td>
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
          {!can('users.manage') && <span className="muted audit__hint">Имена пользователей видит администратор; здесь — логины</span>}
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
