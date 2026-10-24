import type {
  CameraConfig,
  EventPage,
  EventPatch,
  EventQuery,
  GuardEvent,
  Health,
  MetricsSnapshot,
  RuleTypeInfo,
  StatsResponse,
} from '../domain/types';
import { ApiError, type GuardApi, type StatsQuery } from './types';
import { EventSocket } from './ws';

/**
 * Базовый адрес API. По умолчанию — тот же origin (прод раздаёт FastAPI с /,
 * в dev — прокси Vite). Переопределяется VITE_API_BASE=http://host:8000.
 */
export const API_BASE = String(import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '');

function wsUrl(path: string): string {
  if (API_BASE) return API_BASE.replace(/^http/, 'ws') + path;
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${window.location.host}${path}`;
}

function toQuery(params: Record<string, string | number | undefined | null>): string {
  const sp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    sp.set(key, String(value));
  }
  const s = sp.toString();
  return s ? `?${s}` : '';
}

/** FastAPI отдаёт {detail: string | [{msg, loc}]} — превращаем в читаемый текст */
async function readError(res: Response): Promise<string> {
  try {
    const body: unknown = await res.json();
    const detail = (body as { detail?: unknown })?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((d: { msg?: string; loc?: unknown[] }) => [d.loc?.slice(1).join('.'), d.msg].filter(Boolean).join(': '))
        .join('; ');
    }
  } catch {
    // тело не JSON
  }
  return `${res.status} ${res.statusText || 'Ошибка запроса'}`;
}

async function request<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init;
  let res: Response;
  try {
    res = await fetch(API_BASE + path, {
      ...rest,
      headers: json !== undefined ? { 'Content-Type': 'application/json', ...headers } : headers,
      body: json !== undefined ? JSON.stringify(json) : rest.body,
    });
  } catch (err) {
    if ((err as Error).name === 'AbortError') throw err;
    throw new ApiError(0, 'Сервер недоступен');
  }
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  return (text ? JSON.parse(text) : undefined) as T;
}

const enc = encodeURIComponent;

/**
 * Тело PUT/POST — только поля CameraConfig из контракта: GET может отдавать
 * рантайм-поля (например status), их нельзя отправлять обратно.
 */
function cameraPayload(c: CameraConfig): CameraConfig {
  return {
    camera_id: c.camera_id,
    name: c.name,
    site: c.site,
    source: { type: c.source.type, uri: c.source.uri, loop: c.source.loop, realtime: c.source.realtime },
    sample_fps: c.sample_fps,
    enabled: c.enabled,
    zones: c.zones.map((z) => ({ zone_id: z.zone_id, name: z.name, kind: z.kind, polygon: z.polygon, active_hours: z.active_hours })),
    rules: c.rules.map((r) => ({
      rule_id: r.rule_id,
      type: r.type,
      enabled: r.enabled,
      zones: r.zones,
      params: r.params,
      min_duration_s: r.min_duration_s,
      min_ratio: r.min_ratio,
      close_after_s: r.close_after_s,
      cooldown_s: r.cooldown_s,
      severity: r.severity,
    })),
  };
}

export function createHttpApi(): GuardApi {
  const socket = new EventSocket(wsUrl('/ws/events'));

  return {
    mode: 'http',
    health: () => request<Health>('/health'),
    listCameras: () => request<CameraConfig[]>('/api/cameras'),
    createCamera: (camera) => request<CameraConfig>('/api/cameras', { method: 'POST', json: cameraPayload(camera) }),
    updateCamera: (id, camera) => request<CameraConfig>(`/api/cameras/${enc(id)}`, { method: 'PUT', json: cameraPayload(camera) }),
    deleteCamera: (id) => request<void>(`/api/cameras/${enc(id)}`, { method: 'DELETE' }),
    listRuleTypes: () => request<RuleTypeInfo[]>('/api/rules/types'),
    listEvents: (q: EventQuery, signal?: AbortSignal) =>
      request<EventPage>(
        `/api/events${toQuery({
          type: q.type,
          camera_id: q.camera_id,
          zone_id: q.zone_id,
          status: q.status,
          severity: q.severity,
          from: q.from,
          to: q.to,
          limit: q.limit,
          offset: q.offset,
        })}`,
        { signal },
      ),
    getEvent: (id) => request<GuardEvent>(`/api/events/${enc(id)}`),
    patchEvent: (id, patch: EventPatch) => request<GuardEvent>(`/api/events/${enc(id)}`, { method: 'PATCH', json: patch }),
    getStats: (q: StatsQuery) =>
      request<StatsResponse>(`/api/stats${toQuery({ group_by: q.group_by, from: q.from, to: q.to })}`),
    getMetrics: () => request<MetricsSnapshot>('/api/metrics'),
    subscribe: (subscription) => socket.subscribe(subscription),
    media: {
      streamUrl: (cameraId) => `${API_BASE}/api/cameras/${enc(cameraId)}/stream.mjpg`,
      frameUrl: (camera) => `${API_BASE}/api/cameras/${enc(camera.camera_id)}/frame.jpg?t=${Date.now()}`,
      snapshotUrl: (event) => `${API_BASE}/api/events/${enc(event.id)}/snapshot.jpg`,
      clipUrl: (event) => `${API_BASE}/api/events/${enc(event.id)}/clip.mp4`,
    },
  };
}
