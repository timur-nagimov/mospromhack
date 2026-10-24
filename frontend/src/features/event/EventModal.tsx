import { Ban, Check, ChevronRight, DatabaseZap, Film, Forward, Image as ImageIcon, ImageOff, X } from 'lucide-react';
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../api';
import { useSampleForEvent } from '../../api/flywheelQueries';
import { useCameraMap, usePatchEvent, useEvent } from '../../api/queries';
import { StageBar } from '../../components/charts/StageBar';
import { SeverityBadge, StatusBadge, ViolationIcon } from '../../components/ui/badges';
import { toast } from '../../components/ui/toast';
import { describeDetails } from '../../domain/details';
import { formatDateTime, formatMs, formatSeconds, formatTime, shortId } from '../../domain/format';
import { OPERATOR_DECISIONS, STATUS_LABELS, ZONE_KIND_LABELS } from '../../domain/labels';
import type { EventStatus, GuardEvent } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';
import { useNow } from '../../hooks/useNow';
import { liveActions, liveStore, useLive } from '../../state/live';
import { useEventModal } from './useEventModal';
import './event.css';
import { IncidentEmployees } from '../employees/IncidentEmployees';

const MockStream = lazy(() => import('../../components/media/MockStream'));

const DECISION_ICON = { confirmed: Check, false: Ban, forwarded: Forward } as const;

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable;
}

function EventMedia({ event }: { event: GuardEvent }) {
  const [tab, setTab] = useState<'frame' | 'clip'>('frame');
  const [frameFailed, setFrameFailed] = useState(false);
  const [clipFailed, setClipFailed] = useState(false);
  const cameras = useCameraMap();
  const camera = cameras.get(event.camera_id);
  const snapshot = useMemo(() => api().media.snapshotUrl(event), [event]);
  const clip = api().media.clipUrl(event);

  return (
    <div className="ev-media">
      <div className="tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === 'frame'} onClick={() => setTab('frame')}>
          <ImageIcon size={15} /> Кадр с рамкой
        </button>
        <button type="button" role="tab" aria-selected={tab === 'clip'} onClick={() => setTab('clip')}>
          <Film size={15} /> Клип ±5 с
        </button>
      </div>
      <div className="snapshot ev-media__view">
        {tab === 'frame' &&
          (frameFailed ? (
            <div className="snapshot__missing">
              <ImageOff size={26} />
              Снапшот недоступен
            </div>
          ) : (
            <img src={snapshot} alt={`Кадр нарушения: ${getViolationMeta(event.type).title}`} onError={() => setFrameFailed(true)} />
          ))}
        {tab === 'clip' &&
          (clip && !clipFailed ? (
            <video key={clip} src={clip} poster={snapshot} controls autoPlay muted loop playsInline onError={() => setClipFailed(true)} />
          ) : clip == null && camera ? (
            <Suspense fallback={null}>
              <MockStream camera={camera} alert={event} replayFrom={event.started_at - 5} replayLength={10} />
              <span className="ev-media__note">демо: реконструкция клипа</span>
            </Suspense>
          ) : (
            <div className="snapshot__missing">
              <Film size={26} />
              Клип ещё формируется
              <small className="muted">ролик ±5 с сохраняется после окончания нарушения</small>
            </div>
          ))}
      </div>
    </div>
  );
}

/**
 * Кадр события попал в выборку маховика данных — ссылка на сэмпл на странице «Данные».
 * Без API маховика (404) или без сэмпла бейджа просто нет.
 */
function FlywheelBadge({ eventId }: { eventId: string }) {
  const { data: sample } = useSampleForEvent(eventId);
  if (!sample) return null;
  return (
    <Link
      className="badge ev-flywheel"
      to={{ pathname: '/data', search: `?sample=${encodeURIComponent(sample.id)}` }}
      title="Кадр события сохранён в обучающую выборку: открыть сэмпл на странице «Данные»"
    >
      <DatabaseZap size={12} aria-hidden="true" />в выборке маховика
    </Link>
  );
}

function EventCard({ event, onClose, onNext, hasNext }: { event: GuardEvent; onClose: () => void; onNext: () => void; hasNext: boolean }) {
  const now = useNow();
  const cameras = useCameraMap();
  const patch = usePatchEvent();
  const [note, setNote] = useState(event.operator_note ?? '');
  const noteRef = useRef<HTMLTextAreaElement>(null);
  const meta = getViolationMeta(event.type);
  const camera = cameras.get(event.camera_id);
  const zone = camera?.zones.find((z) => z.zone_id === event.zone_id);
  const confirmWindow = event.confirmed_at - event.started_at;
  const processing = event.latency_ms?.total ?? Object.values(event.latency_ms ?? {}).reduce((s, v) => s + v, 0);
  const duration = (event.ended_at ?? now) - event.started_at;
  const details = describeDetails(event.details);

  // Черновик комментария живёт, пока открыта эта карточка (key={event.id} у родителя):
  // event_update из WS не затирает то, что оператор печатает

  const decide = (status: Exclude<EventStatus, 'new'>) => {
    if (patch.isPending) return;
    patch.mutate(
      { id: event.id, patch: { status, operator_note: note.trim() ? note.trim() : null } },
      {
        onSuccess: () => toast(`Статус: ${STATUS_LABELS[status]}`),
        onError: (err) => toast(`Не сохранено: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  const saveNote = () => {
    patch.mutate(
      { id: event.id, patch: { operator_note: note.trim() ? note.trim() : null } },
      {
        onSuccess: () => toast('Комментарий сохранён'),
        onError: (err) => toast(`Не сохранено: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  // Хоткеи: 1/2/3 — решение, N — следующее неразобранное, Esc — закрыть
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isTypingTarget(e.target)) (e.target as HTMLElement).blur();
        else onClose();
        return;
      }
      if (isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const decision = OPERATOR_DECISIONS.find((d) => d.hotkey === e.key);
      if (decision) {
        e.preventDefault();
        decide(decision.status);
      } else if ((e.key === 'n' || e.key === 'т') && hasNext) {
        onNext();
      } else if (e.key === 'c' || e.key === 'с') {
        e.preventDefault();
        noteRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  return (
    <>
      <header className="ev-head">
        <ViolationIcon type={event.type} severity={event.severity} size="lg" />
        <div className="ev-head__title">
          <div className="ev-head__kicker">
            <span className="eyebrow">Событие</span>
            <span className="mono muted" title={event.id}>
              #{shortId(event.id)}
            </span>
            {meta.scenario && <span className="badge badge--plain">{meta.scenario}</span>}
          </div>
          <h2 id="ev-title">{meta.title}</h2>
          <p>{meta.description}</p>
        </div>
        <div className="ev-head__badges">
          <FlywheelBadge eventId={event.id} />
          <SeverityBadge severity={event.severity} />
          <StatusBadge status={event.status} />
        </div>
        <button type="button" className="btn btn--ghost btn--icon" onClick={onClose} aria-label="Закрыть (Esc)">
          <X size={20} />
        </button>
      </header>

      <div className="ev-body">
        <EventMedia event={event} />

        <aside className="ev-side">
          <IncidentEmployees key={event.id} eventId={event.id} />
          <dl className="ev-facts">
            <div>
              <dt>Камера</dt>
              <dd>
                {camera?.name ?? event.camera_id} <span className="mono muted">{event.camera_id}</span>
              </dd>
            </div>
            <div>
              <dt>Участок</dt>
              <dd>{event.site}</dd>
            </div>
            <div>
              <dt>Зона</dt>
              <dd>
                {zone
                  ? `${zone.name} · ${ZONE_KIND_LABELS[zone.kind]}`
                  : typeof event.details?.zone_name === 'string'
                    ? event.details.zone_name
                    : event.zone_id ?? 'Весь кадр'}
              </dd>
            </div>
            <div>
              <dt>Начало нарушения</dt>
              <dd className="num">{formatDateTime(event.started_at)}</dd>
            </div>
            <div>
              <dt>Оповещение</dt>
              <dd className="num">{formatDateTime(event.confirmed_at)}</dd>
            </div>
            <div>
              <dt>Окончание</dt>
              <dd className="num">
                {event.ended_at ? (
                  formatTime(event.ended_at)
                ) : (
                  <span className="ev-ongoing">
                    <span className="live-dot live-dot--bad" /> продолжается
                  </span>
                )}
                <span className="muted"> · длится {formatSeconds(duration)}</span>
              </dd>
            </div>
            {event.track_id != null && (
              <div>
                <dt>Трек</dt>
                <dd className="mono">
                  #{event.track_id} <span className="muted">· правило {event.rule_id}</span>
                </dd>
              </div>
            )}
            {details.map((d) => (
              <div key={d.label}>
                <dt>{d.label}</dt>
                <dd>{d.value}</dd>
              </div>
            ))}
          </dl>

          <section className="ev-latency" aria-label="Задержка оповещения">
            <div className="ev-latency__head">
              <span className="eyebrow">Задержка оповещения</span>
              <strong className="num">{formatSeconds(confirmWindow + processing / 1000)}</strong>
            </div>
            <p className="ev-latency__formula">
              окно подтверждения <b className="num">{formatSeconds(confirmWindow)}</b> + обработка <b className="num">{formatMs(processing)}</b>
            </p>
            <StageBar values={event.latency_ms ?? {}} />
          </section>
        </aside>
      </div>

      <footer className="ev-actions">
        <div className="ev-decisions" role="group" aria-label="Решение оператора">
          {OPERATOR_DECISIONS.map((d) => {
            const Icon = DECISION_ICON[d.status];
            return (
              <button
                key={d.status}
                type="button"
                className={`decision decision--${d.status}${event.status === d.status ? ' is-active' : ''}`}
                onClick={() => decide(d.status)}
                disabled={patch.isPending}
                aria-pressed={event.status === d.status}
              >
                <Icon size={18} strokeWidth={2.4} aria-hidden="true" />
                <span>{d.label}</span>
                <span className="kbd">{d.hotkey}</span>
              </button>
            );
          })}
        </div>
        <div className="ev-note">
          <textarea
            ref={noteRef}
            className="textarea"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Комментарий оператора (C): кто уведомлён, какие меры приняты…"
            maxLength={1000}
            rows={2}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) saveNote();
            }}
          />
          <div className="ev-note__actions">
            <button type="button" className="btn btn--sm" onClick={saveNote} disabled={patch.isPending || note === (event.operator_note ?? '')}>
              Сохранить комментарий
            </button>
            <button type="button" className="btn btn--sm btn--ghost" onClick={onNext} disabled={!hasNext}>
              Следующее неразобранное <span className="kbd">N</span>
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </footer>
    </>
  );
}

export function EventModalHost() {
  const { eventId, openEvent, closeEvent } = useEventModal();
  const feedEvent = useLive((s) => (eventId ? s.feed.find((e) => e.id === eventId) : undefined));
  const { data: event, isError, error } = useEvent(eventId, feedEvent);

  const nextUnreviewed = useLive((s) => s.feed.find((e) => e.status === 'new' && e.id !== eventId) ?? null);

  // Открытие карточки гасит баннер критической тревоги по этому событию
  useEffect(() => {
    const alarm = liveStore.get().alarm;
    if (eventId && alarm?.id === eventId) liveActions.dismissAlarm();
  }, [eventId]);

  if (!eventId) return null;

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) closeEvent();
      }}
    >
      <div className="modal ev-modal" role="dialog" aria-modal="true" aria-labelledby="ev-title">
        {event ? (
          <EventCard
            key={event.id}
            event={event}
            onClose={closeEvent}
            hasNext={nextUnreviewed != null}
            onNext={() => nextUnreviewed && openEvent(nextUnreviewed.id)}
          />
        ) : isError ? (
          <div className="ev-loading">
            <div className="error-box">Не удалось загрузить событие: {(error as Error)?.message}</div>
            <button type="button" className="btn" onClick={closeEvent}>
              Закрыть
            </button>
          </div>
        ) : (
          <div className="ev-loading">
            <span className="spinner" />
          </div>
        )}
      </div>
    </div>
  );
}
