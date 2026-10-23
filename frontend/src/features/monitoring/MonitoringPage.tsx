import { Cctv } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useCameraMap, useCameras, useEvents } from '../../api/queries';
import { formatMs, formatSeconds } from '../../domain/format';
import { isCriticalEvent } from '../../domain/violations';
import { useElementSize } from '../../hooks/useElementSize';
import { useNow } from '../../hooks/useNow';
import { activeEventFor, useLive } from '../../state/live';
import { MAX_MJPEG } from '../../components/media/CameraStream';
import { useEventModal } from '../event/useEventModal';
import { AlertFeed } from './AlertFeed';
import { CameraTile } from './CameraTile';
import { currentShift } from './shift';
import { ShiftTimeline } from './ShiftTimeline';
import './monitoring.css';

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const TILE_HEAD_PX = 46;
const GRID_GAP_PX = 10;
const TIMELINE_MIN_PX = 168;

/**
 * Раскладка плиток: кадр всегда 16:9 без обрезки. Перебираем число колонок
 * и берём то, при котором кадр крупнее всего и сетка помещается по высоте.
 */
function gridLayout(n: number, width: number, height: number): { cols: number; tileW: number | null } {
  if (n === 0 || !width || !height) return { cols: n <= 4 ? 2 : 3, tileW: null };
  let best = { cols: 1, tileW: 0 };
  for (let cols = 1; cols <= Math.min(n, 5); cols += 1) {
    const rows = Math.ceil(n / cols);
    const cellW = (width - GRID_GAP_PX * (cols - 1)) / cols;
    const cellH = (height - GRID_GAP_PX * (rows - 1)) / rows - TILE_HEAD_PX;
    const videoW = Math.floor(Math.min(cellW, (cellH * 16) / 9));
    if (videoW > best.tileW + 4) best = { cols, tileW: videoW };
  }
  return best;
}

export function MonitoringPage() {
  const now = useNow();
  const { data: cameras = [], isLoading } = useCameras();
  const cameraMap = useCameraMap();
  const feed = useLive((s) => s.feed);
  const freshIds = useLive((s) => s.freshIds);
  const unreviewed = useLive((s) => s.unreviewed);
  const metrics = useLive((s) => s.metrics);
  const { openEvent } = useEventModal();
  const [focusedId, setFocusedId] = useState<string | null>(null);
  const gridRef = useRef<HTMLElement>(null);
  const gridSize = useElementSize(gridRef);

  const shift = currentShift(now);
  const { data: shiftEvents } = useEvents({ from: Math.floor(shift.start), limit: 500 });

  const kpi = useMemo(() => {
    const items = shiftEvents?.items ?? [];
    const alertDelays = items.map((e) => e.confirmed_at - e.started_at + (e.latency_ms?.total ?? 0) / 1000);
    return {
      total: shiftEvents?.total ?? null,
      critical: items.filter((e) => isCriticalEvent(e)).length,
      alertDelay: median(alertDelays),
    };
  }, [shiftEvents]);

  const visibleCameras = focusedId ? cameras.filter((c) => c.camera_id === focusedId) : cameras;
  const online = cameras.filter((c) => c.enabled && (metrics?.cameras?.[c.camera_id]?.fps_processed ?? 0) > 0).length;
  const worstLatency = Math.max(0, ...Object.values(metrics?.cameras ?? {}).map((m) => m.timings_ms?.total?.p95 ?? 0));
  // Живой MJPEG — сначала камерам с активным нарушением, потом по порядку (лимит пула соединений)
  const alertsByCamera = new Map(visibleCameras.map((c) => [c.camera_id, activeEventFor(feed, c.camera_id, now)]));
  const liveIds = new Set(
    [...visibleCameras]
      .filter((c) => c.enabled)
      .sort((a, b) => Number(Boolean(alertsByCamera.get(b.camera_id))) - Number(Boolean(alertsByCamera.get(a.camera_id))))
      .slice(0, MAX_MJPEG)
      .map((c) => c.camera_id),
  );

  const layout = gridLayout(visibleCameras.length, gridSize.width, gridSize.height - TIMELINE_MIN_PX - GRID_GAP_PX);

  // Хроника: события смены из журнала + свежие из WS (у них актуальный статус)
  const timelineEvents = useMemo(() => {
    const byId = new Map((shiftEvents?.items ?? []).map((e) => [e.id, e]));
    for (const e of feed) if (e.confirmed_at >= shift.start) byId.set(e.id, e);
    return [...byId.values()];
  }, [shiftEvents, feed, shift.start]);

  return (
    <div className="page page--fill monitoring">
      <div className="kpis reveal">
        <div className="kpi kpi--accent">
          <span className="kpi__label">Неразобрано</span>
          <span className="kpi__value">{unreviewed ?? '—'}</span>
          <span className="kpi__note">ждут решения оператора</span>
        </div>
        <div className="kpi">
          <span className="kpi__label">Событий за смену</span>
          <span className="kpi__value">{kpi.total ?? '—'}</span>
          <span className="kpi__note">{shift.label}</span>
        </div>
        <div className={`kpi${kpi.critical > 0 ? ' kpi--critical' : ''}`}>
          <span className="kpi__label">Критических за смену</span>
          <span className="kpi__value">{kpi.critical}</span>
          <span className="kpi__note">требуют немедленной реакции</span>
        </div>
        <div className="kpi">
          <span className="kpi__label">До оповещения, медиана</span>
          <span className="kpi__value">{kpi.alertDelay != null ? formatSeconds(kpi.alertDelay) : '—'}</span>
          <span className="kpi__note">окно подтверждения + обработка</span>
        </div>
        <div className="kpi">
          <span className="kpi__label">Камеры в работе</span>
          <span className="kpi__value">
            {online}
            <small>/{cameras.length}</small>
          </span>
          <span className="kpi__note">обработка кадра p95 {worstLatency ? formatMs(worstLatency) : '—'}</span>
        </div>
      </div>

      <div className="monitoring__body">
        <section className="cam-grid-wrap" aria-label="Камеры" ref={gridRef}>
          {cameras.length === 0 && !isLoading ? (
            <div className="panel empty">
              <Cctv size={30} />
              <span className="empty__title">Камеры не настроены</span>
              <Link className="btn btn--primary" to="/cameras">
                Добавить камеру
              </Link>
            </div>
          ) : (
            <div
              className="cam-grid"
              style={{
                gridTemplateColumns: layout.tileW ? `repeat(${layout.cols}, ${layout.tileW}px)` : `repeat(${layout.cols}, minmax(0, 1fr))`,
              }}
            >
              {visibleCameras.map((camera, i) => (
                <div key={camera.camera_id} className="reveal" style={{ ['--i' as string]: i + 1 }}>
                  <CameraTile
                    camera={camera}
                    metrics={metrics?.cameras?.[camera.camera_id]}
                    alert={alertsByCamera.get(camera.camera_id) ?? null}
                    now={now}
                    focused={focusedId === camera.camera_id}
                    live={liveIds.has(camera.camera_id)}
                    onToggleFocus={() => setFocusedId((id) => (id === camera.camera_id ? null : camera.camera_id))}
                    onOpenEvent={openEvent}
                  />
                </div>
              ))}
            </div>
          )}
          {cameras.length > 0 && (
            <ShiftTimeline
              cameras={cameras}
              events={timelineEvents}
              from={shift.start}
              now={now}
              label={shift.label}
              onOpen={openEvent}
            />
          )}
        </section>
        <AlertFeed feed={feed} freshIds={freshIds} cameras={cameraMap} now={now} unreviewed={unreviewed} onOpen={openEvent} />
      </div>
    </div>
  );
}
