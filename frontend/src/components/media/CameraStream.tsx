import { VideoOff } from 'lucide-react';
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { API_BASE } from '../../api/http';
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

/**
 * Последний кадр каждой камеры (blob: URL). Плитка меняет режим (MJPEG ↔ кадр / 1 с) или переподключается —
 * и сразу показывает его, а не тёмный фон до первого кадра.
 */
const lastFrames = new Map<string, string>();

function rememberFrame(cameraId: string, url: string) {
  const prev = lastFrames.get(cameraId);
  lastFrames.set(cameraId, url);
  // старый кадр ещё может быть на экране в соседней плитке — освобождаем с запасом
  if (prev && prev !== url) window.setTimeout(() => URL.revokeObjectURL(prev), 5000);
}

/** Снимок текущего кадра MJPEG перед закрытием соединения */
function captureFrame(cameraId: string, img: HTMLImageElement | null) {
  if (!img || img.naturalWidth === 0) return;
  try {
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d')?.drawImage(img, 0, 0);
    canvas.toBlob((blob) => blob && rememberFrame(cameraId, URL.createObjectURL(blob)), 'image/jpeg', 0.8);
  } catch {
    // поток с чужого origin без CORS: снимка не будет, плитка дождётся нового кадра
  }
}

/** MJPEG с оверлеями бэкенда; при обрыве — заглушка и переподключение */
function MjpegStream({ camera }: { camera: CameraConfig }) {
  const cameraId = camera.camera_id;
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

  // Явно рвём соединение при размонтировании, чтобы не держать слот пула; кадр — в запас
  useEffect(() => {
    const img = imgRef.current;
    return () => {
      captureFrame(cameraId, img);
      if (img) img.src = 'data:,';
    };
  }, [attempt, cameraId]);

  const src = `${api().media.streamUrl(cameraId)}${attempt ? `?r=${attempt}` : ''}`;
  const poster = lastFrames.get(cameraId);
  return (
    <>
      {!failed && poster && <img className="stream__media" src={poster} alt="" aria-hidden="true" />}
      {!failed && (
        <img
          ref={imgRef}
          key={attempt}
          className="stream__media stream__media--over"
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
  const cameraId = camera.camera_id;
  const [src, setSrc] = useState<string | null>(() => lastFrames.get(cameraId) ?? null);
  const [failed, setFailed] = useState(false);
  const cameraRef = useRef(camera);
  cameraRef.current = camera;

  // Перезапуск цикла только при смене камеры, а не при каждом перезапросе списка.
  // Кадр забираем сами: один запрос вместо двух, а при ошибке на экране остаётся прежний кадр, а не «битая» картинка.
  useEffect(() => {
    let alive = true;
    let timer = 0;
    const abort = new AbortController();
    setSrc(lastFrames.get(cameraId) ?? null);
    const load = async () => {
      try {
        const res = await fetch(api().media.frameUrl(cameraRef.current), {
          cache: 'no-store',
          credentials: API_BASE ? 'include' : 'same-origin',
          signal: abort.signal,
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const url = URL.createObjectURL(await res.blob());
        if (!alive) {
          URL.revokeObjectURL(url);
          return;
        }
        rememberFrame(cameraId, url);
        setSrc(url);
        setFailed(false);
        timer = window.setTimeout(load, POLL_MS);
      } catch {
        if (!alive) return;
        setFailed(true);
        timer = window.setTimeout(load, RETRY_MS);
      }
    };
    void load();
    return () => {
      alive = false;
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [cameraId]);

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
