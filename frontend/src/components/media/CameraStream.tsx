import { VideoOff } from 'lucide-react';
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import './media.css';

const MockStream = lazy(() => import('./MockStream'));

const RETRY_MS = 5000;
const POLL_MS = 1000;

/**
 * Браузер держит не больше 6 HTTP/1.1-соединений на origin, а каждый MJPEG
 * занимает одно навсегда. Поэтому живых MJPEG не больше MAX_MJPEG (по умолчанию 4),
 * остальные плитки опрашивают frame.jpg раз в секунду.
 */
export const MAX_MJPEG = Math.max(1, Number(import.meta.env.VITE_MAX_MJPEG ?? 4) || 4);

/** MJPEG с оверлеями бэкенда; при обрыве — заглушка и переподключение */
function MjpegStream({ camera }: { camera: CameraConfig }) {
  const [attempt, setAttempt] = useState(0);
  const [failed, setFailed] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!failed) return;
    const t = window.setTimeout(() => {
      setFailed(false);
      setAttempt((a) => a + 1);
    }, RETRY_MS);
    return () => window.clearTimeout(t);
  }, [failed]);

  // Явно рвём соединение при размонтировании, чтобы не держать слот пула
  useEffect(() => {
    const img = imgRef.current;
    return () => {
      if (img) img.src = 'data:,';
    };
  }, [attempt]);

  const src = `${api().media.streamUrl(camera.camera_id)}${attempt ? `?r=${attempt}` : ''}`;
  return (
    <>
      {!failed && (
        <img
          ref={imgRef}
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

/** Стоп-кадры раз в секунду: короткие запросы не занимают пул соединений */
function PolledFrame({ camera }: { camera: CameraConfig }) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  // Перезапуск цикла только при смене камеры, а не при каждом перезапросе списка
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const load = () => {
      const url = api().media.frameUrl(cameraRef.current);
      const img = new Image();
      img.onload = () => {
        if (!alive) return;
        setSrc(url);
        setFailed(false);
        timer = window.setTimeout(load, POLL_MS);
      };
      img.onerror = () => {
        if (!alive) return;
        setFailed(true);
        timer = window.setTimeout(load, RETRY_MS);
      };
      img.src = url;
    };
    load();
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [camera.camera_id]);

  if (failed && !src) {
    return (
      <div className="stream__nosignal">
        <VideoOff size={22} aria-hidden="true" />
        <span>Нет сигнала</span>
      </div>
    );
  }
  return (
    <>
      {src && <img className="stream__media" src={src} alt={`Кадр камеры: ${camera.name}`} />}
      <span className="stream__mode" title="Живой MJPEG показывается для приоритетных камер; эта обновляется раз в секунду">
        кадр / 1 с
      </span>
    </>
  );
}

export function CameraStream({ camera, alert, live = true }: { camera: CameraConfig; alert?: GuardEvent | null; live?: boolean }) {
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
  return live ? <MjpegStream camera={camera} /> : <PolledFrame camera={camera} />;
}
