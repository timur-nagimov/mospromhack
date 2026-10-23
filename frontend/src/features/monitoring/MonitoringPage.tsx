import { Cctv } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useCameraMap, useCameras, useEvents } from '../../api/queries';
import { formatMs, formatSeconds } from '../../domain/format';
import { isCriticalEvent } from '../../domain/violations';
import { useElementSize } from '../../hooks/useElementSize';
import { useNow } from '../../hooks/useNow';
import { activeEventFor, useLive } from '../../state/live';
import { useEventModal } from '../event/useEventModal';
import { AlertFeed } from './AlertFeed';
import { CameraTile } from './CameraTile';
import { currentShift } from './shift';
import './monitoring.css';

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

const TILE_HEAD_PX = 45;
const GRID_GAP_PX = 10;

/** Число колонок, при котором кадр 16:9 в плитке получается крупнее всего */
function gridColumns(n: number, width: number, height: number): number {
  if (n <= 1) return 1;
  if (!width || !height) return n <= 4 ? 2 : 3;
  let best = 1;
  let bestW = 0;
  for (let cols = 1; cols <= Math.min(n, 5); cols += 1) {
    const rows = Math.ceil(n / cols);
    const cellW = (width - GRID_GAP_PX * (cols - 1)) / cols;
    const cellH = (height - GRID_GAP_PX * (rows - 1)) / rows - TILE_HEAD_PX;
    const videoW = Math.min(cellW, (cellH * 16) / 9);
    if (videoW > bestW + 4) {
      bestW = videoW;
      best = cols;
    }
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
  const cols = gridColumns(visibleCameras.length, gridSize.width, gridSize.height);
  const rows = Math.max(1, Math.ceil(visibleCameras.length / cols));

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
          <span className="kpi__note">падения и сближения с техникой</span>
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
              style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))` }}
            >
              {visibleCameras.map((camera, i) => (
                <div key={camera.camera_id} className="reveal" style={{ ['--i' as string]: i + 1, minHeight: 0 }}>
                  <CameraTile
                    camera={camera}
                    metrics={metrics?.cameras?.[camera.camera_id]}
                    alert={activeEventFor(feed, camera.camera_id, now)}
                    now={now}
                    focused={focusedId === camera.camera_id}
                    onToggleFocus={() => setFocusedId((id) => (id === camera.camera_id ? null : camera.camera_id))}
                    onOpenEvent={openEvent}
                  />
                </div>
              ))}
            </div>
          )}
        </section>
        <AlertFeed feed={feed} freshIds={freshIds} cameras={cameraMap} now={now} unreviewed={unreviewed} onOpen={openEvent} />
      </div>
    </div>
  );
}
