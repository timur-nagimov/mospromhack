// Синтетическая «камера наблюдения» для мок-режима: стилизованная сцена цеха,
// люди и техника в движении, оверлеи как у MJPEG бэкенда (рамки, треки, зоны).
// Рисуем в виртуальном кадре FRAME_W×FRAME_H, масштабируя под canvas.

import type { BBox, CameraConfig } from '../../domain/types';
import { ZONE_KIND_COLORS } from '../../domain/labels';
import { FRAME_H, FRAME_W } from './data';

type SceneKind = 'racks' | 'line' | 'weld' | 'dock' | 'scaffold' | 'perimeter';

const KIND_BY_CAMERA: Record<string, SceneKind> = {
  'cam-01': 'racks',
  'cam-02': 'line',
  'cam-03': 'weld',
  'cam-04': 'dock',
  'cam-05': 'scaffold',
  'cam-06': 'perimeter',
};

const KINDS: SceneKind[] = ['racks', 'line', 'weld', 'dock', 'scaffold', 'perimeter'];

function sceneKind(cameraId: string): SceneKind {
  const known = KIND_BY_CAMERA[cameraId];
  if (known) return known;
  let h = 0;
  for (const ch of cameraId) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return KINDS[h % KINDS.length];
}

export interface SceneHighlight {
  bbox: BBox;
  label: string;
  type: string;
}

export interface SceneOptions {
  camera: CameraConfig;
  /** epoch-секунды: определяет позиции людей и техники */
  t: number;
  /** рамки детекций и зоны — как в stream.mjpg; для frame.jpg выключено */
  overlays: boolean;
  highlight?: SceneHighlight | null;
  /** подписи HUD (имя камеры, время) */
  hud?: boolean;
}

const HORIZON = 300;
const VX = 640;

let noiseCanvas: HTMLCanvasElement | null = null;

function noise(): HTMLCanvasElement {
  if (noiseCanvas) return noiseCanvas;
  const c = document.createElement('canvas');
  c.width = 192;
  c.height = 192;
  const ctx = c.getContext('2d');
  if (ctx) {
    const img = ctx.createImageData(c.width, c.height);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.random() * 255;
      img.data[i] = v;
      img.data[i + 1] = v;
      img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  noiseCanvas = c;
  return c;
}

/** Рост человека в пикселях по высоте точки опоры (перспектива) */
const personHeight = (footY: number): number => 70 + (footY - HORIZON) * 0.62;

function floor(ctx: CanvasRenderingContext2D, top: string, bottom: string): void {
  const wall = ctx.createLinearGradient(0, 0, 0, HORIZON);
  wall.addColorStop(0, '#15191c');
  wall.addColorStop(1, '#262c30');
  ctx.fillStyle = wall;
  ctx.fillRect(0, 0, FRAME_W, HORIZON);
  const g = ctx.createLinearGradient(0, HORIZON, 0, FRAME_H);
  g.addColorStop(0, top);
  g.addColorStop(1, bottom);
  ctx.fillStyle = g;
  ctx.fillRect(0, HORIZON, FRAME_W, FRAME_H - HORIZON);
  ctx.strokeStyle = 'rgba(255,255,255,0.045)';
  ctx.lineWidth = 1.5;
  for (let i = -14; i <= 14; i += 1) {
    ctx.beginPath();
    ctx.moveTo(VX + i * 14, HORIZON);
    ctx.lineTo(VX + i * 150, FRAME_H);
    ctx.stroke();
  }
  for (let k = 1; k < 11; k += 1) {
    const y = HORIZON + (FRAME_H - HORIZON) * (k / 11) ** 1.7;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(FRAME_W, y);
    ctx.stroke();
  }
}

function box3d(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, depth: number, face: string): void {
  ctx.fillStyle = face;
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = 'rgba(255,255,255,0.06)';
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + depth, y - depth * 0.5);
  ctx.lineTo(x + w + depth, y - depth * 0.5);
  ctx.lineTo(x + w, y);
  ctx.closePath();
  ctx.fill();
  ctx.fillStyle = 'rgba(0,0,0,0.25)';
  ctx.beginPath();
  ctx.moveTo(x + w, y);
  ctx.lineTo(x + w + depth, y - depth * 0.5);
  ctx.lineTo(x + w + depth, y + h - depth * 0.5);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
  ctx.fill();
}

function hazardStripe(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = 'rgba(230,180,30,0.85)';
  for (let i = -h; i < w + h; i += 28) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + h);
    ctx.lineTo(x + i + 14, y + h);
    ctx.lineTo(x + i + 14 + h, y);
    ctx.lineTo(x + i + h, y);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
}

function drawBackdrop(ctx: CanvasRenderingContext2D, kind: SceneKind, t: number): void {
  switch (kind) {
    case 'racks': {
      floor(ctx, '#353b3e', '#474d4f');
      // стеллажи слева и справа, уходящие к точке схода
      for (let side = -1; side <= 1; side += 2) {
        for (let i = 0; i < 6; i += 1) {
          const k = i / 6;
          const x = side < 0 ? 40 + k * 330 : 1240 - k * 330;
          const top = 40 + k * 150;
          const bottom = 700 - k * 330;
          const w = 70 - k * 40;
          ctx.fillStyle = '#2b4a63';
          ctx.fillRect(x - (side < 0 ? 0 : w), top, 7 - k * 3, bottom - top);
          for (let s = 0; s < 4; s += 1) {
            const sy = top + ((bottom - top) * (s + 0.6)) / 4.2;
            ctx.fillStyle = '#c66a2a';
            ctx.fillRect(x - (side < 0 ? 0 : w), sy, w, 5 - k * 2);
            ctx.fillStyle = s % 2 ? '#6b5a44' : '#5a5f63';
            ctx.fillRect(x - (side < 0 ? -6 : w - 6), sy - (34 - k * 18), w - 12, 32 - k * 17);
          }
        }
      }
      ctx.strokeStyle = 'rgba(230,185,40,0.7)';
      ctx.lineWidth = 6;
      ctx.beginPath();
      ctx.moveTo(420, FRAME_H);
      ctx.lineTo(600, HORIZON + 40);
      ctx.moveTo(860, FRAME_H);
      ctx.lineTo(690, HORIZON + 40);
      ctx.stroke();
      break;
    }
    case 'line': {
      floor(ctx, '#33393b', '#43494b');
      for (let i = 0; i < 5; i += 1) box3d(ctx, 90 + i * 240, 170, 150, 120, 26, i % 2 ? '#3d5566' : '#4b5357');
      ctx.fillStyle = '#25292b';
      ctx.beginPath();
      ctx.moveTo(60, 430);
      ctx.lineTo(1220, 395);
      ctx.lineTo(1232, 470);
      ctx.lineTo(48, 522);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 2;
      const shift = (t * 40) % 40;
      for (let x = 60 + shift; x < 1220; x += 40) {
        const y1 = 430 - ((x - 60) / 1160) * 35;
        ctx.beginPath();
        ctx.moveTo(x, y1);
        ctx.lineTo(x - 10, y1 + 88);
        ctx.stroke();
      }
      for (let i = 0; i < 4; i += 1) {
        const x = ((t * 40 + i * 300) % 1160) + 60;
        const y = 430 - ((x - 60) / 1160) * 35;
        box3d(ctx, x, y - 34, 70, 38, 12, '#8a6a46');
      }
      break;
    }
    case 'weld': {
      floor(ctx, '#2f3436', '#3e4345');
      for (let i = 0; i < 3; i += 1) {
        ctx.fillStyle = 'rgba(120,40,30,0.55)';
        ctx.fillRect(110 + i * 300, 150, 220, 260);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.strokeRect(110 + i * 300, 150, 220, 260);
      }
      box3d(ctx, 380, 420, 330, 60, 30, '#545b5f');
      // искры сварки
      const flicker = Math.sin(t * 23) * 0.5 + 0.5;
      const glow = ctx.createRadialGradient(560, 410, 4, 560, 410, 120);
      glow.addColorStop(0, `rgba(170,220,255,${0.55 + flicker * 0.35})`);
      glow.addColorStop(1, 'rgba(170,220,255,0)');
      ctx.fillStyle = glow;
      ctx.fillRect(420, 280, 280, 260);
      ctx.fillStyle = 'rgba(255,220,150,0.9)';
      for (let i = 0; i < 14; i += 1) {
        const a = (i / 14) * Math.PI * 2 + t * 3;
        const r = 10 + ((t * 90 + i * 17) % 60);
        ctx.fillRect(560 + Math.cos(a) * r, 410 + Math.sin(a) * r * 0.6, 3, 3);
      }
      box3d(ctx, 1000, 360, 140, 150, 20, '#3a4a40');
      break;
    }
    case 'dock': {
      floor(ctx, '#3a3f41', '#4b5052');
      ctx.fillStyle = '#0c0e10';
      ctx.fillRect(300, 70, 680, 250);
      ctx.fillStyle = '#5c6265';
      ctx.fillRect(330, 100, 620, 200);
      ctx.fillStyle = '#7a6750';
      for (let i = 0; i < 4; i += 1) ctx.fillRect(360 + i * 140, 210, 110, 90);
      hazardStripe(ctx, 260, 318, 760, 18);
      break;
    }
    case 'scaffold': {
      floor(ctx, '#34393b', '#45494a');
      ctx.fillStyle = '#2a3034';
      ctx.fillRect(120, 40, 1040, 300);
      ctx.strokeStyle = '#8f9599';
      ctx.lineWidth = 6;
      for (let x = 150; x <= 1130; x += 140) {
        ctx.beginPath();
        ctx.moveTo(x, 40);
        ctx.lineTo(x, 520);
        ctx.stroke();
      }
      for (const y of [110, 220, 330]) {
        ctx.beginPath();
        ctx.moveTo(140, y);
        ctx.lineTo(1140, y);
        ctx.stroke();
        ctx.fillStyle = '#8a6d45';
        ctx.fillRect(140, y - 10, 1000, 10);
      }
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(160,165,170,0.6)';
      for (let x = 150; x < 1130; x += 140) {
        ctx.beginPath();
        ctx.moveTo(x, 330);
        ctx.lineTo(x + 140, 220);
        ctx.stroke();
      }
      break;
    }
    case 'perimeter': {
      const sky = ctx.createLinearGradient(0, 0, 0, 330);
      sky.addColorStop(0, '#0d1418');
      sky.addColorStop(1, '#1f2a2e');
      ctx.fillStyle = sky;
      ctx.fillRect(0, 0, FRAME_W, 330);
      const g = ctx.createLinearGradient(0, 300, 0, FRAME_H);
      g.addColorStop(0, '#2b3330');
      g.addColorStop(1, '#3c4440');
      ctx.fillStyle = g;
      ctx.fillRect(0, 300, FRAME_W, FRAME_H - 300);
      const lamp = ctx.createRadialGradient(980, 120, 6, 980, 300, 380);
      lamp.addColorStop(0, 'rgba(255,230,170,0.5)');
      lamp.addColorStop(1, 'rgba(255,230,170,0)');
      ctx.fillStyle = lamp;
      ctx.fillRect(560, 60, 720, 660);
      // сетка ограждения
      ctx.strokeStyle = 'rgba(190,200,205,0.35)';
      ctx.lineWidth = 1.4;
      for (let x = -40; x < 1320; x += 22) {
        ctx.beginPath();
        ctx.moveTo(x, 300 - (x / 1280) * 40);
        ctx.lineTo(x + 60, 445 - ((x + 60) / 1280) * 40);
        ctx.moveTo(x + 60, 300 - ((x + 60) / 1280) * 40);
        ctx.lineTo(x, 445 - (x / 1280) * 40);
        ctx.stroke();
      }
      ctx.fillStyle = '#5d6669';
      for (let x = 0; x <= 1280; x += 160) ctx.fillRect(x, 286 - (x / 1280) * 40, 9, 165);
      break;
    }
  }
}

interface PersonStyle {
  helmet: boolean;
  vest: boolean;
  lying?: boolean;
  phone?: boolean;
  smoking?: boolean;
  carry?: boolean;
}

function drawPerson(ctx: CanvasRenderingContext2D, fx: number, fy: number, h: number, phase: number, s: PersonStyle): void {
  const cloth = '#4d565c';
  if (s.lying) {
    const w = h * 2.3;
    const y = fy - h * 0.35;
    ctx.fillStyle = cloth;
    ctx.beginPath();
    ctx.roundRect(fx - w / 2 + h * 0.5, y, w - h * 0.7, h * 0.34, h * 0.12);
    ctx.fill();
    if (s.vest) {
      ctx.fillStyle = '#d9752c';
      ctx.fillRect(fx - w / 2 + h * 0.7, y + 2, w * 0.3, h * 0.3);
    }
    ctx.fillStyle = '#c9a58a';
    ctx.beginPath();
    ctx.arc(fx - w / 2 + h * 0.3, y + h * 0.17, h * 0.17, 0, Math.PI * 2);
    ctx.fill();
    return;
  }
  const head = h * 0.075;
  const swing = Math.sin(phase) * h * 0.09;
  ctx.strokeStyle = '#3b4247';
  ctx.lineCap = 'round';
  ctx.lineWidth = h * 0.085;
  ctx.beginPath();
  ctx.moveTo(fx - h * 0.04, fy - h * 0.46);
  ctx.lineTo(fx - h * 0.04 + swing, fy);
  ctx.moveTo(fx + h * 0.04, fy - h * 0.46);
  ctx.lineTo(fx + h * 0.04 - swing, fy);
  ctx.stroke();
  ctx.fillStyle = cloth;
  ctx.beginPath();
  ctx.roundRect(fx - h * 0.13, fy - h * 0.82, h * 0.26, h * 0.4, h * 0.06);
  ctx.fill();
  if (s.vest) {
    ctx.fillStyle = '#e07a2e';
    ctx.fillRect(fx - h * 0.12, fy - h * 0.79, h * 0.24, h * 0.3);
    ctx.fillStyle = 'rgba(230,235,235,0.85)';
    ctx.fillRect(fx - h * 0.12, fy - h * 0.62, h * 0.24, h * 0.03);
  }
  ctx.strokeStyle = cloth;
  ctx.lineWidth = h * 0.06;
  ctx.beginPath();
  if (s.phone) {
    ctx.moveTo(fx + h * 0.12, fy - h * 0.78);
    ctx.lineTo(fx + h * 0.16, fy - h * 0.62);
    ctx.lineTo(fx + h * 0.07, fy - h * 0.86);
  } else if (s.carry) {
    ctx.moveTo(fx - h * 0.12, fy - h * 0.76);
    ctx.lineTo(fx - h * 0.2, fy - h * 0.6);
    ctx.moveTo(fx + h * 0.12, fy - h * 0.76);
    ctx.lineTo(fx + h * 0.2, fy - h * 0.6);
  } else {
    ctx.moveTo(fx - h * 0.13, fy - h * 0.78);
    ctx.lineTo(fx - h * 0.15 - swing * 0.6, fy - h * 0.47);
    ctx.moveTo(fx + h * 0.13, fy - h * 0.78);
    ctx.lineTo(fx + h * 0.15 + swing * 0.6, fy - h * 0.47);
  }
  ctx.stroke();
  if (s.carry) {
    ctx.fillStyle = '#8a6a46';
    ctx.fillRect(fx - h * 0.3, fy - h * 0.98, h * 0.6, h * 0.4);
  }
  ctx.fillStyle = '#c9a58a';
  ctx.beginPath();
  ctx.arc(fx, fy - h * 0.9, head, 0, Math.PI * 2);
  ctx.fill();
  if (s.helmet) {
    ctx.fillStyle = '#f0c63a';
    ctx.beginPath();
    ctx.arc(fx, fy - h * 0.91, head * 1.15, Math.PI, 0);
    ctx.fill();
    ctx.fillRect(fx - head * 1.35, fy - h * 0.915, head * 2.7, head * 0.3);
  }
  if (s.phone) {
    ctx.fillStyle = '#111';
    ctx.fillRect(fx + head * 0.7, fy - h * 0.93, head * 0.5, head * 0.9);
  }
  if (s.smoking) {
    ctx.fillStyle = '#ff8a3a';
    ctx.beginPath();
    ctx.arc(fx + head * 1.4, fy - h * 0.87, head * 0.22, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = 'rgba(210,210,210,0.35)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(fx + head * 1.4, fy - h * 0.9);
    ctx.bezierCurveTo(fx + head * 2, fy - h * 1.0, fx + head * 0.8, fy - h * 1.08, fx + head * 1.6, fy - h * 1.18);
    ctx.stroke();
  }
}

function drawForklift(ctx: CanvasRenderingContext2D, x: number, y: number, scale: number, facing: 1 | -1): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(scale * facing, scale);
  ctx.fillStyle = '#d9a426';
  ctx.beginPath();
  ctx.roundRect(-60, -70, 110, 55, 6);
  ctx.fill();
  ctx.strokeStyle = '#2a2d2f';
  ctx.lineWidth = 5;
  ctx.strokeRect(-40, -130, 60, 62);
  ctx.fillStyle = '#2a2d2f';
  ctx.fillRect(55, -125, 8, 115);
  ctx.fillRect(55, -18, 55, 7);
  ctx.beginPath();
  ctx.arc(-35, -12, 15, 0, Math.PI * 2);
  ctx.arc(30, -12, 15, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

interface Actor {
  x: number;
  y: number;
  h: number;
  phase: number;
  style: PersonStyle;
  track: number;
}

/** Позиции людей на кадре в момент t: петли вдоль заданных траекторий */
function actors(kind: SceneKind, t: number): Actor[] {
  const loop = (period: number, offset = 0) => (((t + offset) % period) + period) % period / period;
  const pingpong = (period: number, offset = 0) => {
    const p = loop(period, offset);
    return p < 0.5 ? p * 2 : 2 - p * 2;
  };
  const mk = (x: number, y: number, track: number, style: PersonStyle, speed = 7): Actor => ({
    x,
    y,
    h: personHeight(y),
    phase: t * speed + track,
    style,
    track,
  });
  switch (kind) {
    case 'racks':
      return [
        mk(520 + pingpong(18) * 160, 560 + pingpong(18) * 60, 12, { helmet: true, vest: true }),
        mk(700 - pingpong(26, 5) * 90, 420 + pingpong(26, 5) * 30, 27, { helmet: true, vest: true }),
      ];
    case 'line':
      return [
        mk(300 + pingpong(30) * 120, 600, 41, { helmet: true, vest: true }, 4),
        mk(760, 580, 44, { helmet: true, vest: true }, 0.6),
        mk(1020 - pingpong(22, 3) * 300, 650, 47, { helmet: true, vest: true }),
      ];
    case 'weld':
      return [mk(600, 520, 63, { helmet: true, vest: false }, 0.5), mk(880 + pingpong(20) * 120, 600, 66, { helmet: true, vest: true })];
    case 'dock':
      return [mk(380 + pingpong(16) * 520, 620, 81, { helmet: true, vest: true }), mk(860, 470, 85, { helmet: true, vest: true }, 0.4)];
    case 'scaffold':
      return [
        { ...mk(380 + pingpong(24) * 380, 330, 92, { helmet: true, vest: true }, 5), h: 150 },
        mk(820, 600, 95, { helmet: true, vest: true }, 0.5),
      ];
    case 'perimeter':
      return [mk(200 + pingpong(40) * 800, 600, 104, { helmet: false, vest: true }, 5)];
  }
}

function drawLabel(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, bg: string, fg = '#fff', size = 15): void {
  ctx.font = `600 ${size}px "JetBrains Mono Variable", ui-monospace, monospace`;
  const w = ctx.measureText(text).width + 12;
  const h = size + 9;
  const yy = Math.max(0, y - h);
  ctx.fillStyle = bg;
  ctx.fillRect(x, yy, w, h);
  ctx.fillStyle = fg;
  ctx.fillText(text, x + 6, yy + size + 2);
}

function drawZones(ctx: CanvasRenderingContext2D, camera: CameraConfig): void {
  for (const zone of camera.zones) {
    if (zone.polygon.length < 3) continue;
    const color = ZONE_KIND_COLORS[zone.kind] ?? '#e5484d';
    ctx.beginPath();
    zone.polygon.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x * FRAME_W, y * FRAME_H) : ctx.lineTo(x * FRAME_W, y * FRAME_H)));
    ctx.closePath();
    ctx.globalAlpha = 0.13;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 0.85;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
    const [lx, ly] = zone.polygon[0];
    drawLabel(ctx, zone.name, lx * FRAME_W + 4, ly * FRAME_H + 26, 'rgba(10,12,14,0.72)', color, 13);
  }
}

export function drawScene(ctx: CanvasRenderingContext2D, width: number, height: number, o: SceneOptions): void {
  const kind = sceneKind(o.camera.camera_id);
  ctx.save();
  ctx.setTransform(width / FRAME_W, 0, 0, height / FRAME_H, 0, 0);
  drawBackdrop(ctx, kind, o.t);

  if (kind === 'dock' || kind === 'racks') {
    const p = ((o.t % 22) + 22) % 22 / 22;
    const x = kind === 'dock' ? 200 + Math.sin(p * Math.PI * 2) * 360 + 360 : 980 + Math.sin(p * Math.PI * 2) * 140;
    const y = kind === 'dock' ? 520 : 640;
    drawForklift(ctx, x, y, kind === 'dock' ? 1.15 : 1.35, Math.cos(p * Math.PI * 2) > 0 ? 1 : -1);
  }

  if (o.overlays) drawZones(ctx, o.camera);

  for (const a of actors(kind, o.t)) {
    drawPerson(ctx, a.x, a.y, a.h, a.phase, a.style);
    if (o.overlays) {
      const x1 = a.x - a.h * 0.24;
      const y1 = a.y - a.h * 1.0;
      ctx.strokeStyle = 'rgba(61,214,140,0.9)';
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1, y1, a.h * 0.48, a.h * 1.02);
      drawLabel(ctx, `#${a.track} person ${(0.86 + (a.track % 10) / 100).toFixed(2)}`, x1, y1, 'rgba(22,120,74,0.85)', '#fff', 12);
    }
  }

  if (o.highlight) {
    const [x1, y1, x2, y2] = o.highlight.bbox;
    const w = x2 - x1;
    const h = y2 - y1;
    const type = o.highlight.type;
    if (type === 'vehicle_proximity') drawForklift(ctx, x2 + 120, y2, 1.25, -1);
    if (type === 'smoking_fire' && o.highlight.label.includes('огонь')) {
      const g = ctx.createRadialGradient(x2 + 40, y2 - 30, 4, x2 + 40, y2 - 30, 60);
      g.addColorStop(0, 'rgba(255,200,90,0.95)');
      g.addColorStop(1, 'rgba(255,90,20,0)');
      ctx.fillStyle = g;
      ctx.fillRect(x2 - 20, y2 - 100, 120, 110);
    }
    drawPerson(ctx, x1 + w / 2, y2, type === 'fall_immobile' ? w / 2.3 : h, 0.6, {
      helmet: type !== 'ppe_missing',
      vest: type !== 'ppe_missing',
      lying: type === 'fall_immobile',
      phone: type === 'phone_usage',
      smoking: type === 'smoking_fire',
      carry: type === 'unsafe_carry',
    });
    ctx.strokeStyle = '#ff3b3b';
    ctx.lineWidth = 5;
    ctx.strokeRect(x1, y1, w, h);
    drawLabel(ctx, o.highlight.label, x1 - 2.5, y1 - 2, '#e5484d', '#fff', 17);
  }

  // зерно, сканлайны и виньетка — «картинка с камеры»
  const pattern = ctx.createPattern(noise(), 'repeat');
  if (pattern) {
    ctx.save();
    ctx.globalAlpha = 0.07;
    ctx.translate(Math.random() * 192, Math.random() * 192);
    ctx.fillStyle = pattern;
    ctx.fillRect(-192, -192, FRAME_W + 384, FRAME_H + 384);
    ctx.restore();
  }
  ctx.fillStyle = 'rgba(0,0,0,0.1)';
  for (let y = 0; y < FRAME_H; y += 4) ctx.fillRect(0, y, FRAME_W, 1);
  const vignette = ctx.createRadialGradient(FRAME_W / 2, FRAME_H / 2, FRAME_H * 0.35, FRAME_W / 2, FRAME_H / 2, FRAME_W * 0.72);
  vignette.addColorStop(0, 'rgba(0,0,0,0)');
  vignette.addColorStop(1, 'rgba(0,0,0,0.6)');
  ctx.fillStyle = vignette;
  ctx.fillRect(0, 0, FRAME_W, FRAME_H);

  if (o.hud !== false) {
    const d = new Date(o.t * 1000);
    const pad = (n: number) => String(n).padStart(2, '0');
    const stamp = `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    ctx.font = '600 22px "JetBrains Mono Variable", ui-monospace, monospace';
    ctx.fillStyle = 'rgba(240,240,235,0.88)';
    ctx.fillText(stamp, FRAME_W - ctx.measureText(stamp).width - 26, 44);
    ctx.fillText(o.camera.camera_id.toUpperCase(), 26, 44);
    if (Math.floor(o.t * 2) % 2 === 0) {
      ctx.fillStyle = '#e5484d';
      ctx.beginPath();
      ctx.arc(34, FRAME_H - 34, 8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = 'rgba(240,240,235,0.8)';
    ctx.font = '600 18px "JetBrains Mono Variable", ui-monospace, monospace';
    ctx.fillText('REC', 50, FRAME_H - 28);
  }
  ctx.restore();
}

/** Статичный кадр в data:URL: стоп-кадр для редактора зон и снапшоты событий */
export function renderToDataUrl(options: SceneOptions, width = 960, height = 540): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  if (!ctx) return '';
  drawScene(ctx, width, height, options);
  return canvas.toDataURL('image/jpeg', 0.84);
}
