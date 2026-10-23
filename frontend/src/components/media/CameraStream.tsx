import { VideoOff } from 'lucide-react';
import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from '../../api';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import './media.css';

const MockStream = lazy(() => import('./MockStream'));

const RETRY_MS = 5000;

/** MJPEG с оверлеями бэкенда; при обрыве — заглушка и переподключение */
function MjpegStream({ camera }: { camera: CameraConfig }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!failed) return;
    const t = window.setTimeout(() => {
      setFailed(false);
      setAttempt((a) => a + 1);
    }, RETRY_MS);
    return () => window.clearTimeout(t);
  }, [failed]);

  const src = `${api().media.streamUrl(camera.camera_id)}${attempt ? `?r=${attempt}` : ''}`;
  return (
    <>
      {!failed && (
        <img
          key={attempt}
          className="stream__media"
          src={src}
          alt={`Видеопоток: ${camera.name}`}
          onError={() => setFailed(true)}
          decoding="async"
        />
      )}
      {failed && (
        <div className="stream__nosignal">
          <VideoOff size={22} aria-hidden="true" />
          <span>Нет сигнала</span>
          <small>переподключение…</small>
        </div>
      )}
    </>
  );
}

export function CameraStream({ camera, alert }: { camera: CameraConfig; alert?: GuardEvent | null }) {
  if (!camera.enabled) {
    return (
      <div className="stream__nosignal">
        <VideoOff size={22} aria-hidden="true" />
        <span>Камера отключена</span>
      </div>
    );
  }
  if (api().mode === 'mock') {
    return (
      <Suspense fallback={<div className="stream__nosignal" />}>
        <MockStream camera={camera} alert={alert ?? null} />
      </Suspense>
    );
  }
  return <MjpegStream camera={camera} />;
}
