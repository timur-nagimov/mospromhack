import { Maximize2, Minimize2 } from 'lucide-react';
import { memo } from 'react';
import { CameraStream } from '../../components/media/CameraStream';
import { formatClock, formatMs, formatNumber } from '../../domain/format';
import type { CameraConfig, CameraMetrics, GuardEvent } from '../../domain/types';
import { getViolationMeta, isCriticalEvent } from '../../domain/violations';
import { ViolationIcon } from '../../components/ui/badges';
import { SourceStateChip, SourceStateDot, sourceView } from '../../components/ui/sourceState';

interface Props {
  camera: CameraConfig;
  metrics: CameraMetrics | undefined;
  alert: GuardEvent | null;
  now: number;
  focused: boolean;
  /** живой MJPEG или опрос кадров (см. MAX_MJPEG) */
  live: boolean;
  onToggleFocus: () => void;
  onOpenEvent: (id: string) => void;
}

function CameraTileImpl({ camera, metrics, alert, now, focused, live, onToggleFocus, onOpenEvent }: Props) {
  const fps = metrics?.fps_processed;
  const latency = metrics?.timings_ms?.total?.p95;
  const online = camera.enabled && fps != null && fps > 0;
  const source = sourceView(camera.enabled, metrics?.source, online);
  const noSignal = source.state === 'reconnecting' || source.state === 'offline' || source.state === 'error';
  const critical = alert ? isCriticalEvent(alert) : false;
  const state = alert ? (critical ? 'critical' : 'alert') : 'ok';

  return (
    <article className={`tile tile--${state}`} aria-label={`Камера ${camera.name}`}>
      <header className="tile__head">
        <SourceStateDot view={source} />
        <div className="tile__name">
          <strong>{camera.name}</strong>
          <span>
            {camera.site} · <span className="mono">{camera.camera_id}</span>
          </span>
        </div>
        <div className="tile__badges">
          <SourceStateChip view={source} />
          <span className="metric-chip" title="Обработано кадров в секунду">
            <b className="num">{fps != null ? formatNumber(fps, 1) : '—'}</b> к/с
          </span>
          <span className="metric-chip" title="Сквозная задержка обработки кадра, p95">
            <b className="num">{formatMs(latency)}</b>
          </span>
          <button
            type="button"
            className="btn btn--ghost btn--sm btn--icon"
            onClick={onToggleFocus}
            aria-label={focused ? 'Показать все камеры' : 'Развернуть камеру'}
            title={focused ? 'Все камеры' : 'Развернуть'}
          >
            {focused ? <Minimize2 size={15} /> : <Maximize2 size={15} />}
          </button>
        </div>
      </header>
      <div className="tile__video">
        <CameraStream camera={camera} alert={alert} live={live} />
        {noSignal && (
          <div className={`tile__nosignal tile__nosignal--${source.tone}`} title={source.title}>
            <strong>{source.state === 'reconnecting' ? 'Связь с камерой потеряна' : source.label}</strong>
            <span>{source.state === 'error' ? 'Проверьте адрес источника' : 'Переподключение идёт автоматически'}</span>
          </div>
        )}
        {alert && (
          <button type="button" className="tile__alert" onClick={() => onOpenEvent(alert.id)}>
            <span className="tile__alert-stripe hazard--red" aria-hidden="true" />
            <ViolationIcon type={alert.type} severity={alert.severity} size="sm" />
            <span className="tile__alert-text">
              <small>Нарушение</small>
              <strong>{getViolationMeta(alert.type).title}</strong>
            </span>
            <span className="tile__alert-timer mono">{formatClock(now - alert.started_at)}</span>
          </button>
        )}
      </div>
    </article>
  );
}

export const CameraTile = memo(CameraTileImpl);
