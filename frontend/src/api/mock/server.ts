// Мок-реализация GuardApi: данные в памяти, фейковый WS (события раз в 10–20 с,
// метрики раз в 2 с), задержки ответа как у сети. Включается VITE_MOCK=1 или ?mock=1.

import type {
  CameraConfig,
  CameraMetrics,
  EventQuery,
  GuardEvent,
  MetricsSnapshot,
  Percentiles,
  StatsItem,
  StageKey,
  WsMessage,
} from '../../domain/types';
import { STAGE_KEYS } from '../../domain/labels';
import { getViolationMeta } from '../../domain/violations';
import { nowSec } from '../../domain/format';
import { moscowDayKey, moscowHour, moscowWeekdayHour } from '../../domain/moscow';
import { ApiError, type GuardApi, type StatsQuery, type Subscription } from '../types';
import { buildEvent, pickDraft, RULE_TYPES, seedCameras, seedHistory } from './data';
import { createFlywheelMock } from './flywheel';
import { createEmployeesMock } from './employees';
import { between, mulberry32 } from './random';
import { renderToDataUrl } from './scene';

const clone = <T>(value: T): T => structuredClone(value);

/** Базовые задержки этапов (p50, мс) для бэкенда linq-tpu */
const STAGE_BASE: Record<StageKey, number> = {
  decode: 3.4,
  preprocess: 2.6,
  transfer: 17.5,
  queue: 4.2,
  infer: 9.6,
  postprocess: 1.9,
  tracking: 1.2,
  rules: 0.6,
  alert: 1.1,
};

const STAGE_TAIL: Record<StageKey, number> = {
  decode: 1.6,
  preprocess: 1.5,
  transfer: 1.9,
  queue: 3.1,
  infer: 1.25,
  postprocess: 1.5,
  tracking: 1.6,
  rules: 2.0,
  alert: 2.4,
};

export function createMockApi(): GuardApi {
  const rng = mulberry32((Date.now() % 100000) + 7);
  const startedAt = nowSec();
  let cameras: CameraConfig[] = seedCameras();
  let events: GuardEvent[] = seedHistory(cameras, startedAt);

  // Пара событий «прямо сейчас», чтобы мониторинг сразу показывал активное нарушение
  const activeDraft = cameras.find((c) => c.camera_id === 'cam-02');
  if (activeDraft) {
    const ppeRule = activeDraft.rules.find((r) => r.type === 'ppe_missing');
    if (ppeRule) {
      events.unshift(buildEvent(rng, { camera: activeDraft, rule: ppeRule, confirmedAt: startedAt - 26 }, 'new', false));
    }
  }

  const subscribers = new Set<Subscription>();
  const timers: number[] = [];
  let liveTimer: number | null = null;
  let liveCount = 0;

  const frameCache = new Map<string, string>();
  const snapshotCache = new Map<string, string>();

  function respond<T>(value: T, min = 90, max = 260): Promise<T> {
    return new Promise((resolve) => window.setTimeout(() => resolve(clone(value)), between(rng, min, max)));
  }

  function fail(status: number, message: string): Promise<never> {
    return new Promise((_, reject) => window.setTimeout(() => reject(new ApiError(status, message)), 120));
  }

  function emit(message: WsMessage): void {
    for (const s of subscribers) s.onMessage(clone(message));
  }

  // Маховик данных: сэмплы событий и живой сбор по фейковому WS
  const flywheel = createFlywheelMock({ cameras: () => cameras, events: () => events, emit, respond, fail });
  const employees = createEmployeesMock(() => events);

  function findCamera(id: string): CameraConfig | undefined {
    return cameras.find((c) => c.camera_id === id);
  }

  function metricsSnapshot(): MetricsSnapshot {
    const t = nowSec();
    const camMetrics: Record<string, CameraMetrics> = {};
    let fpsTotal = 0;
    let inferP50 = 0;
    cameras.forEach((cam, idx) => {
      if (!cam.enabled) return;
      const wobble = Math.sin(t / 7 + idx) * 0.5 + 0.5;
      const fps = Math.max(0, cam.sample_fps - 0.05 - rng() * 0.25 - (idx === 3 ? wobble * 0.4 : 0));
      fpsTotal += fps;
      const timings: Partial<Record<string, Percentiles>> = {};
      let p50Sum = 0;
      let p95Sum = 0;
      for (const key of STAGE_KEYS) {
        const base = STAGE_BASE[key] * (1 + idx * 0.03) * (0.92 + rng() * 0.16);
        const p50 = Math.round(base * 10) / 10;
        const p95 = Math.round(base * STAGE_TAIL[key] * (0.95 + rng() * 0.1) * 10) / 10;
        timings[key] = { p50, p95 };
        p50Sum += p50;
        p95Sum += p95;
      }
      timings.total = { p50: Math.round(p50Sum * 0.98 * 10) / 10, p95: Math.round(p95Sum * 0.78 * 10) / 10 };
      inferP50 = Math.max(inferP50, timings.infer?.p50 ?? 0);
      camMetrics[cam.camera_id] = {
        fps_in: cam.source.type === 'images' ? cam.sample_fps : 25,
        fps_processed: Math.round(fps * 100) / 100,
        lag_s: Math.round((0.08 + rng() * 0.18 + (idx === 3 ? wobble * 0.25 : 0)) * 100) / 100,
        timings_ms: timings,
      };
    });
    const utilization = Math.min(0.95, (fpsTotal * inferP50 * 1.6) / 1000 + 0.05 + rng() * 0.03);
    const r1 = (v: number) => Math.round(v * 10) / 10;
    const model = (share: number, infer: number, transfer: number, calls: number) => ({
      timings_ms: {
        transfer: { p50: r1(transfer), p95: r1(transfer * 1.8) },
        infer: { p50: r1(infer), p95: r1(infer * 1.25) },
      },
      fps: r1(fpsTotal * share),
      calls,
      items: calls * 3,
      max_batch_seen: 4,
      errors: 0,
      last_error: null,
    });
    const tick = Math.floor(t - startedAt);
    return {
      backend: 'linq-tpu',
      cameras: camMetrics,
      inference: {
        backend: 'linq-tpu',
        device: 'LinQ TPU (стенд хакатона)',
        utilization: Math.round(utilization * 1000) / 1000,
        models: {
          det: model(1, inferP50, 16.8, 4200 + tick * 8),
          ppe: model(0.55, inferP50 * 0.62, 9.4, 2300 + tick * 4),
          pose: model(0.3, inferP50 * 0.85, 12.1, 1250 + tick * 2),
        },
      },
    };
  }

  /** Новое подтверждённое событие «с камеры» + его закрытие через 20–45 с */
  function spawnLive(): void {
    const t = nowSec();
    let draft = pickDraft(rng, cameras, t);
    if (liveCount === 0) {
      // Первое живое событие — критическое падение (сквозной сценарий кейса)
      const cam = cameras.find((c) => c.enabled && c.rules.some((r) => r.type === 'fall_immobile' && r.enabled));
      const fallRule = cam?.rules.find((r) => r.type === 'fall_immobile');
      if (cam && fallRule) draft = { camera: cam, rule: fallRule, confirmedAt: t };
    }
    if (!draft) return;
    liveCount += 1;
    const event = buildEvent(rng, draft, 'new', false);
    events.unshift(event);
    emit({ kind: 'event', event });
    flywheel.onEvent(event);
    const closeIn = between(rng, 20000, 45000);
    timers.push(
      window.setTimeout(() => {
        const current = events.find((e) => e.id === event.id);
        if (!current || current.ended_at != null) return;
        current.ended_at = nowSec();
        emit({ kind: 'event_update', event: current });
      }, closeIn),
    );
  }

  function scheduleLive(delay: number): void {
    liveTimer = window.setTimeout(() => {
      spawnLive();
      scheduleLive(between(rng, 10000, 20000));
    }, delay);
  }

  function start(): void {
    scheduleLive(2500);
    flywheel.start();
    timers.push(window.setInterval(() => emit({ kind: 'metrics', data: metricsSnapshot() }), 2000));
    // первый снимок метрик сразу — UI не ждёт 2 с
    timers.push(window.setTimeout(() => emit({ kind: 'metrics', data: metricsSnapshot() }), 300));
  }

  function stop(): void {
    flywheel.stop();
    if (liveTimer != null) window.clearTimeout(liveTimer);
    liveTimer = null;
    for (const id of timers) {
      window.clearTimeout(id);
      window.clearInterval(id);
    }
    timers.length = 0;
  }

  function filterEvents(q: Omit<EventQuery, 'status' | 'severity'> & { status?: string; severity?: string }, source = events): GuardEvent[] {
    const matches = (filter: string | undefined, value: string | null) =>
      !filter || filter.split(',').map((v) => v.trim()).includes(value ?? '');
    return source.filter(
      (e) =>
        matches(q.type, e.type) &&
        matches(q.camera_id, e.camera_id) &&
        matches(q.zone_id, e.zone_id) &&
        matches(q.status, e.status) &&
        matches(q.severity, e.severity) &&
        (q.from == null || e.confirmed_at >= q.from) &&
        (q.to == null || e.confirmed_at <= q.to),
    );
  }

  function validateCamera(camera: CameraConfig): string | null {
    if (!camera.camera_id.trim()) return 'camera_id: обязательное поле';
    if (!camera.name.trim()) return 'name: обязательное поле';
    if (!camera.source.uri.trim()) return 'source.uri: обязательное поле';
    if (!(camera.sample_fps > 0)) return 'sample_fps: должно быть больше 0';
    for (const z of camera.zones) if (z.polygon.length < 3) return `zones.${z.zone_id}: нужно минимум 3 точки`;
    return null;
  }

  function statsItems(q: StatsQuery, source = events): StatsItem[] {
    const counts = new Map<string, number>();
    for (const e of filterEvents(q, source)) {
      let key: string | null;
      switch (q.group_by) {
        case 'type':
          key = e.type;
          break;
        case 'camera':
          key = e.camera_id;
          break;
        case 'zone':
          key = e.zone_id;
          break;
        case 'site':
          key = e.site;
          break;
        case 'severity':
          key = e.severity;
          break;
        case 'status':
          key = e.status;
          break;
        case 'hour':
          key = String(moscowHour(e.confirmed_at));
          break;
        case 'day':
          key = moscowDayKey(e.confirmed_at);
          break;
        case 'weekday_hour':
          key = moscowWeekdayHour(e.confirmed_at);
          break;
      }
      if (key == null) continue;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const items: StatsItem[] = [...counts.entries()].map(([key, count]) => ({ key, count }));
    return items;
  }

  return {
    mode: 'mock',
    health: () => respond({ status: 'ok', backend: 'linq-tpu', cameras_running: cameras.filter((c) => c.enabled).length }),
    listCameras: () => respond(cameras),
    createCamera: (camera) => {
      const err = validateCamera(camera);
      if (err) return fail(422, err);
      if (findCamera(camera.camera_id)) return fail(409, `Камера ${camera.camera_id} уже существует`);
      cameras = [...cameras, clone(camera)];
      return respond(camera, 200, 400);
    },
    updateCamera: (id, camera) => {
      const err = validateCamera(camera);
      if (err) return fail(422, err);
      if (!findCamera(id)) return fail(404, `Камера ${id} не найдена`);
      cameras = cameras.map((c) => (c.camera_id === id ? clone(camera) : c));
      return respond(camera, 200, 450);
    },
    deleteCamera: (id) => {
      if (!findCamera(id)) return fail(404, `Камера ${id} не найдена`);
      cameras = cameras.filter((c) => c.camera_id !== id);
      return respond(undefined as void, 150, 300);
    },
    probeSource: (source) => {
      // мок проверки потока: адрес со словами fail/404 — ошибка, иначе 1280×720, 25 к/с, H.264
      const uri = source.uri.trim();
      const base = { width: null, height: null, fps: null, codec: null, first_frame_ms: null, preview_jpeg_b64: null };
      if (!uri) return respond({ ...base, ok: false, error: 'адрес источника пуст', error_code: 'bad_uri' }, 100, 200);
      if (/fail|404/i.test(uri)) {
        const error = 'поток не открылся: поток не найден (RTSP 404): проверьте путь после адреса камеры';
        return respond({ ...base, ok: false, error, error_code: 'open_failed' }, 600, 900);
      }
      const hls = /\.m3u8/i.test(uri);
      return respond(
        { ...base, ok: true, error: null, width: 1280, height: 720, fps: 25, codec: 'h264', first_frame_ms: hls ? 2400 : 640,
          stream: hls ? 'hls' : source.type, hints: hls ? ['HLS идёт с задержкой в 1–3 сегмента (обычно 5–30 с)'] : [] },
        500,
        900,
      );
    },
    listRuleTypes: () => respond(RULE_TYPES),
    listEvents: (q) => {
      const filtered = filterEvents(q);
      const offset = q.offset ?? 0;
      const limit = q.limit ?? 50;
      return respond({ items: filtered.slice(offset, offset + limit), total: filtered.length });
    },
    getEvent: (id) => {
      const ev = events.find((e) => e.id === id);
      return ev ? respond(ev, 60, 150) : fail(404, 'Событие не найдено');
    },
    patchEvent: (id, patch) => {
      const ev = events.find((e) => e.id === id);
      if (!ev) return fail(404, 'Событие не найдено');
      if (patch.status) ev.status = patch.status;
      if (patch.operator_note !== undefined) ev.operator_note = patch.operator_note;
      if (patch.status) flywheel.onEventPatch(ev);
      window.setTimeout(() => emit({ kind: 'event_update', event: ev }), 140);
      return respond(ev, 100, 220);
    },
    getStats: (q: StatsQuery) => respond({ group_by: q.group_by, items: statsItems(q) }),
    getAnalytics: (q) => {
      // Все поля вычисляются синхронно до respond: обновления не вклиниваются между ними.
      const source = q.employee_id ? employees.forAnalytics(q.employee_id) : events;
      const aggregate = (group_by: StatsQuery['group_by'], filters: { status?: string; severity?: string } = {}) =>
        statsItems({ ...q, group_by, ...filters }, source);
      return respond({
        types: aggregate('type'), cameras: aggregate('camera'), sites: aggregate('site'),
        days: aggregate('day'), weekdayHours: aggregate('weekday_hour'),
        criticalTypes: aggregate('type', { severity: 'critical' }),
        statusTypes: {
          new: aggregate('type', { status: 'new' }), confirmed: aggregate('type', { status: 'confirmed' }),
          false: aggregate('type', { status: 'false' }), forwarded: aggregate('type', { status: 'forwarded' }),
        },
      });
    },
    getMetrics: () => respond(metricsSnapshot(), 60, 140),
    subscribe: (subscription) => {
      subscribers.add(subscription);
      subscription.onStatus?.('connecting');
      window.setTimeout(() => subscription.onStatus?.('open'), 250);
      if (subscribers.size === 1) start();
      return () => {
        subscribers.delete(subscription);
        if (subscribers.size === 0) stop();
      };
    },
    media: {
      streamUrl: (cameraId) => `mock://stream/${cameraId}`,
      frameUrl: (camera) => {
        const cached = frameCache.get(camera.camera_id);
        if (cached) return cached;
        const url = renderToDataUrl({ camera, t: nowSec(), overlays: false, hud: false }, 1280, 720);
        frameCache.set(camera.camera_id, url);
        return url;
      },
      snapshotUrl: (event) => {
        const cached = snapshotCache.get(event.id);
        if (cached) return cached;
        const camera =
          findCamera(event.camera_id) ??
          ({ camera_id: event.camera_id, name: event.camera_id, site: event.site, zones: [], rules: [] } as unknown as CameraConfig);
        const meta = getViolationMeta(event.type);
        const url = renderToDataUrl({
          camera,
          t: event.confirmed_at,
          overlays: true,
          highlight: { bbox: event.bbox, label: `НАРУШЕНИЕ · ${meta.short}`, type: event.type },
        });
        if (snapshotCache.size > 400) snapshotCache.clear();
        snapshotCache.set(event.id, url);
        return url;
      },
      clipUrl: () => null,
    },
    flywheel: flywheel.api,
    employees,
  };
}
