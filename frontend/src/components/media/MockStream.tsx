import { useEffect, useRef } from 'react';
import { drawScene } from '../../api/mock/scene';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';

const FRAME_INTERVAL_MS = 125; // ~8 кадров/с хватает для «живой» картинки

interface Props {
  camera: CameraConfig;
  alert: GuardEvent | null;
  /** Воспроизведение клипа: время сцены = start + (прошло с монтирования) */
  replayFrom?: number;
  replayLength?: number;
}

/** Анимированная синтетическая камера (только мок-режим) */
export default function MockStream({ camera, alert, replayFrom, replayLength = 10 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef({ camera, alert, replayFrom, replayLength });
  propsRef.current = { camera, alert, replayFrom, replayLength };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let last = 0;
    let visible = true;
    const mountedAt = performance.now();

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
    });
    io.observe(canvas);

    const frame = (ts: number) => {
      raf = requestAnimationFrame(frame);
      if (!visible || document.hidden || ts - last < FRAME_INTERVAL_MS) return;
      last = ts;
      const p = propsRef.current;
      const elapsed = (ts - mountedAt) / 1000;
      const t = p.replayFrom != null ? p.replayFrom + (elapsed % p.replayLength) : Date.now() / 1000;
      const a = p.alert;
      drawScene(ctx, canvas.width, canvas.height, {
        camera: p.camera,
        t,
        overlays: true,
        highlight: a ? { bbox: a.bbox, label: `НАРУШЕНИЕ · ${getViolationMeta(a.type).short}`, type: a.type } : null,
      });
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="stream__media" role="img" aria-label={`Видеопоток (демо): ${camera.name}`} />;
}
