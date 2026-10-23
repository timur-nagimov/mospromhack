import { useMemo } from 'react';
import { formatHM, formatTime } from '../../domain/format';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import { getViolationMeta, isCriticalEvent } from '../../domain/violations';

interface Props {
  cameras: CameraConfig[];
  events: GuardEvent[];
  from: number;
  now: number;
  label: string;
  onOpen: (id: string) => void;
}

/** Хроника смены: события по камерам на оси времени — видно серии и «горячие» участки */
export function ShiftTimeline({ cameras, events, from, now, label, onOpen }: Props) {
  const span = Math.max(600, now - from);
  const ticks = useMemo(() => {
    const out: number[] = [];
    const d = new Date(from * 1000);
    d.setMinutes(0, 0, 0);
    let t = d.getTime() / 1000 + 3600;
    const step = span > 8 * 3600 ? 7200 : 3600;
    while (t < now) {
      out.push(t);
      t += step;
    }
    return out;
  }, [from, now, span]);

  const byCamera = useMemo(() => {
    const map = new Map<string, GuardEvent[]>();
    for (const e of events) {
      if (e.confirmed_at < from) continue;
      const list = map.get(e.camera_id) ?? [];
      list.push(e);
      map.set(e.camera_id, list);
    }
    return map;
  }, [events, from]);

  const pos = (ts: number) => `${Math.min(100, Math.max(0, ((ts - from) / span) * 100))}%`;

  return (
    <section className="panel timeline" aria-label="Хроника смены">
      <header className="timeline__head">
        <span className="eyebrow">Хроника смены</span>
        <span className="muted">{label} · события по камерам</span>
        <span className="timeline__legend">
          <i className="timeline__dot timeline__dot--critical" /> критические
          <i className="timeline__dot" /> остальные
          <i className="timeline__dot timeline__dot--done" /> разобранные
        </span>
      </header>
      <div className="timeline__body">
        <div className="timeline__axis">
          <span />
          <div className="timeline__track">
            {ticks.map((t) => (
              <span key={t} className="timeline__tick" style={{ left: pos(t) }}>
                {formatHM(t)}
              </span>
            ))}
            <span className="timeline__tick timeline__tick--now" style={{ left: '100%' }}>
              сейчас
            </span>
          </div>
        </div>
        {cameras.map((c) => (
          <div key={c.camera_id} className="timeline__row">
            <span className="timeline__cam" title={c.name}>
              {c.name}
            </span>
            <div className="timeline__track">
              {ticks.map((t) => (
                <span key={t} className="timeline__grid" style={{ left: pos(t) }} />
              ))}
              {(byCamera.get(c.camera_id) ?? []).map((e) => {
                const critical = isCriticalEvent(e);
                const cls = ['timeline__dot', critical ? 'timeline__dot--critical' : '', e.status !== 'new' ? 'timeline__dot--done' : '']
                  .filter(Boolean)
                  .join(' ');
                return (
                  <button
                    key={e.id}
                    type="button"
                    className={cls}
                    style={{ left: pos(e.confirmed_at), ['--sev' as string]: `var(--sev-${e.severity})` }}
                    title={`${formatTime(e.confirmed_at)} · ${getViolationMeta(e.type).title}`}
                    aria-label={`${formatTime(e.confirmed_at)}, ${getViolationMeta(e.type).title}`}
                    onClick={() => onOpen(e.id)}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
