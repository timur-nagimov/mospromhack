import { BellRing, Inbox } from 'lucide-react';
import { useMemo, useState } from 'react';
import { zoneName } from '../../api/queries';
import { EventThumb } from '../../components/media/EventThumb';
import { SeverityBadge, StatusBadge, ViolationIcon } from '../../components/ui/badges';
import { formatRelative, formatTime } from '../../domain/format';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import { getViolationMeta, isCriticalEvent } from '../../domain/violations';

type Filter = 'all' | 'new' | 'critical';

interface Props {
  feed: GuardEvent[];
  freshIds: ReadonlySet<string>;
  cameras: Map<string, CameraConfig>;
  now: number;
  unreviewed: number | null;
  onOpen: (id: string) => void;
}

function FeedItem({
  event,
  cameras,
  now,
  fresh,
  onOpen,
}: {
  event: GuardEvent;
  cameras: Map<string, CameraConfig>;
  now: number;
  fresh: boolean;
  onOpen: (id: string) => void;
}) {
  const meta = getViolationMeta(event.type);
  const critical = isCriticalEvent(event);
  const camera = cameras.get(event.camera_id);
  const zone = zoneName(cameras, event.camera_id, event.zone_id);
  const cls = ['feed-item', critical ? 'feed-item--critical' : '', event.status === 'new' ? 'is-new' : 'is-done', fresh ? 'is-fresh' : '']
    .filter(Boolean)
    .join(' ');
  return (
    <li>
      <button type="button" className={cls} onClick={() => onOpen(event.id)} style={{ ['--sev' as string]: `var(--sev-${event.severity})` }}>
        <span className="feed-item__media">
          <EventThumb event={event} width={92} />
          <span className="feed-item__icon">
            <ViolationIcon type={event.type} severity={event.severity} size="sm" />
          </span>
        </span>
        <span className="feed-item__body">
          <span className="feed-item__row">
            <strong className="feed-item__title">{meta.title}</strong>
            <span className="feed-item__time mono">{formatTime(event.confirmed_at)}</span>
          </span>
          <span className="feed-item__where">
            {camera?.name ?? event.camera_id}
            {zone ? ` · ${zone}` : ''}
          </span>
          <span className="feed-item__row feed-item__meta">
            <span className="feed-item__badges">
              {event.status === 'new' ? critical ? <SeverityBadge severity={event.severity} solid /> : <StatusBadge status="new" /> : <StatusBadge status={event.status} />}
            </span>
            <span className="feed-item__ago">{formatRelative(event.confirmed_at, now)}</span>
          </span>
        </span>
      </button>
    </li>
  );
}

export function AlertFeed({ feed, freshIds, cameras, now, unreviewed, onOpen }: Props) {
  const [filter, setFilter] = useState<Filter>('all');

  const { pinned, rest } = useMemo(() => {
    const pinnedList = feed.filter((e) => e.status === 'new' && isCriticalEvent(e));
    const pinnedIds = new Set(pinnedList.map((e) => e.id));
    let others = feed.filter((e) => !pinnedIds.has(e.id));
    if (filter === 'new') others = others.filter((e) => e.status === 'new');
    if (filter === 'critical') others = others.filter((e) => isCriticalEvent(e));
    return { pinned: pinnedList, rest: others.slice(0, 80) };
  }, [feed, filter]);

  const criticalCount = pinned.length;

  return (
    <section className="panel feed" aria-label="Лента тревог">
      <header className="feed__head">
        <div className="feed__title">
          <span className="eyebrow">Лента тревог</span>
          <div className="feed__counter">
            <span className="feed__counter-num num">{unreviewed ?? '—'}</span>
            <span className="feed__counter-label">
              неразобранных
              {criticalCount > 0 && <em>{criticalCount} крит.</em>}
            </span>
          </div>
        </div>
        <div className="segmented" role="group" aria-label="Фильтр ленты">
          {(
            [
              ['all', 'Все'],
              ['new', 'Новые'],
              ['critical', 'Критичные'],
            ] as const
          ).map(([key, label]) => (
            <button key={key} type="button" aria-pressed={filter === key} onClick={() => setFilter(key)}>
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className="feed__scroll">
        {pinned.length > 0 && (
          <div className="feed__group feed__group--pinned">
            <div className="feed__group-label">
              <BellRing size={13} aria-hidden="true" /> Требуют реакции
            </div>
            <ul className="feed__list">
              {pinned.map((e) => (
                <FeedItem key={e.id} event={e} cameras={cameras} now={now} fresh={freshIds.has(e.id)} onOpen={onOpen} />
              ))}
            </ul>
          </div>
        )}
        <div className="feed__group">
          {pinned.length > 0 && <div className="feed__group-label">Остальные события</div>}
          {rest.length === 0 ? (
            <div className="empty">
              <Inbox size={26} />
              <span className="empty__title">Событий нет</span>
              <span>Новые тревоги появятся здесь автоматически</span>
            </div>
          ) : (
            <ul className="feed__list">
              {rest.map((e) => (
                <FeedItem key={e.id} event={e} cameras={cameras} now={now} fresh={freshIds.has(e.id)} onOpen={onOpen} />
              ))}
            </ul>
          )}
        </div>
      </div>
      <footer className="feed__foot">
        <span className="kbd">1</span>
        <span className="kbd">2</span>
        <span className="kbd">3</span>
        <span>решение оператора в карточке события</span>
      </footer>
    </section>
  );
}
