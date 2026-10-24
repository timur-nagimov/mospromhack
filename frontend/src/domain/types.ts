// TS-зеркало контрактов guard/contracts.py и docs/CONTRACTS.md (раздел «HTTP API»).
// Меняется синхронно с контрактом; всё, что сверх контракта, помечено как опциональное.

import type { FlywheelSampleBrief } from './flywheel';

export type Severity = 'critical' | 'high' | 'medium' | 'low';
export type EventStatus = 'new' | 'confirmed' | 'false' | 'forwarded';

export type ZoneKind =
  | 'danger'
  | 'ppe_required'
  | 'no_smoking'
  | 'smoking_area'
  | 'vehicle_path'
  | 'height'
  | 'fence';

/** Точка полигона в нормированных координатах кадра 0..1 */
export type Point = [number, number];

/** x1, y1, x2, y2 в пикселях исходного кадра */
export type BBox = [number, number, number, number];

export interface Zone {
  zone_id: string;
  name: string;
  kind: ZoneKind;
  polygon: Point[];
  /** ["07:55-12:00"] по местному времени; null — активна всегда */
  active_hours: string[] | null;
}

export interface RuleConfig {
  rule_id: string;
  type: string;
  enabled: boolean;
  /** null — весь кадр */
  zones: string[] | null;
  params: Record<string, unknown>;
  min_duration_s: number;
  min_ratio: number;
  close_after_s: number;
  cooldown_s: number;
  severity: Severity;
}

export type SourceType = 'file' | 'rtsp' | 'images' | 'webcam';

export interface SourceConfig {
  type: SourceType;
  uri: string;
  loop: boolean;
  realtime: boolean;
}

/** Состояние источника: status.source.state в /api/cameras и metrics.cameras[id].source (docs/STREAMS.md) */
export type SourceState = 'connecting' | 'online' | 'reconnecting' | 'offline' | 'error';

/** Компактный статус источника из метрик (WS metrics каждые 2 с) */
export interface SourceBrief {
  state: SourceState;
  fps_in: number;
  last_frame_age_s: number | null;
  width: number | null;
  height: number | null;
  reconnects: number;
  drops: number;
  last_error: string | null;
}

/** POST /api/cameras/probe — проверка потока до добавления камеры */
export interface SourceProbeResult {
  ok: boolean;
  width: number | null;
  height: number | null;
  fps: number | null;
  codec: string | null;
  first_frame_ms: number | null;
  error: string | null;
  preview_jpeg_b64: string | null;
  error_code?: string | null;
  stream?: string | null;
  open_ms?: number | null;
  fps_nominal?: number | null;
  fps_measured?: number | null;
  resolved_via?: string | null;
  hints?: string[];
}

export interface CameraConfig {
  camera_id: string;
  name: string;
  /** участок */
  site: string;
  source: SourceConfig;
  sample_fps: number;
  enabled: boolean;
  zones: Zone[];
  rules: RuleConfig[];
}

export interface GuardEvent {
  id: string;
  camera_id: string;
  site: string;
  zone_id: string | null;
  rule_id: string;
  type: string;
  severity: Severity;
  track_id: number | null;
  /** epoch, секунды: первый нарушающий кадр */
  started_at: number;
  /** epoch, секунды: подтверждение = создание оповещения */
  confirmed_at: number;
  ended_at: number | null;
  bbox: BBox;
  snapshot_path: string | null;
  clip_path: string | null;
  status: EventStatus;
  operator_note: string | null;
  details: Record<string, unknown>;
  /** замеры обработки, мс; ключи из TIMING_KEYS */
  latency_ms: Record<string, number>;
}

export interface EventPage {
  items: GuardEvent[];
  total: number;
}

export interface EventQuery {
  type?: string;
  camera_id?: string;
  zone_id?: string;
  status?: EventStatus;
  /** Нет в контракте: бэкенд может игнорировать, UI дофильтровывает на клиенте */
  severity?: Severity;
  /** epoch, секунды */
  from?: number;
  to?: number;
  limit?: number;
  offset?: number;
}

export interface EventPatch {
  status?: EventStatus;
  operator_note?: string | null;
}

export type StatsGroupBy = 'type' | 'camera' | 'zone' | 'site' | 'severity' | 'status' | 'hour' | 'day' | 'weekday_hour';

export interface StatsItem {
  key: string;
  count: number;
}

export interface StatsResponse {
  group_by: StatsGroupBy;
  items: StatsItem[];
}

/** Все показатели страницы рассчитаны сервером в одном снимке журнала. */
export interface AnalyticsResponse {
  types: StatsItem[];
  cameras: StatsItem[];
  sites: StatsItem[];
  days: StatsItem[];
  weekdayHours: StatsItem[];
  criticalTypes: StatsItem[];
  statusTypes: Record<EventStatus, StatsItem[]>;
}

export const TIMING_KEYS = [
  'decode',
  'preprocess',
  'transfer',
  'queue',
  'infer',
  'postprocess',
  'tracking',
  'rules',
  'alert',
  'total',
] as const;

export type TimingKey = (typeof TIMING_KEYS)[number];
export type StageKey = Exclude<TimingKey, 'total'>;

export interface Percentiles {
  p50: number;
  p95: number;
}

export interface CameraMetrics {
  fps_in: number;
  fps_processed: number;
  lag_s: number;
  timings_ms: Partial<Record<string, Percentiles>>;
  frames_processed?: number;
  errors?: number;
  /** состояние источника; нет у старых бэкендов */
  source?: SourceBrief;
}

/** Статистика модели из BackendStats.snapshot() (guard/inference/base.py) */
export interface ModelStats {
  timings_ms?: Partial<Record<string, Percentiles & { n?: number }>>;
  fps?: number;
  calls?: number;
  items?: number;
  max_batch_seen?: number;
  errors?: number;
  last_error?: string | null;
}

/** backend.stats(): форма в контракте не зафиксирована, читаем известные поля мягко */
export interface InferenceStats {
  backend?: string;
  models?: Record<string, ModelStats>;
  /** доля занятости ускорителя 0..1 — ожидание сверх контракта */
  utilization?: number;
  [key: string]: unknown;
}

export interface MetricsSnapshot {
  backend: string;
  cameras: Record<string, CameraMetrics>;
  inference: InferenceStats;
}

export interface RuleTypeInfo {
  type: string;
  title: string;
  required_models: string[];
  default_severity: Severity;
  params_schema: JsonSchema;
  /** Если отсутствует, действует совместимый выбор «весь кадр» / несколько зон. */
  zone_schema?: JsonSchema;
}

export interface Health {
  status: string;
  backend: string;
  cameras_running: number;
}

export type WsMessage =
  | { kind: 'event'; event: GuardEvent }
  | { kind: 'event_update'; event: GuardEvent }
  | { kind: 'metrics'; data: MetricsSnapshot }
  /** Новый сэмпл маховика (docs/FLYWHEEL.md); бэкенд может прислать его с ключом type вместо kind */
  | { kind: 'flywheel_sample'; sample: FlywheelSampleBrief };

/** Подмножество JSON Schema, которое понимает генератор форм */
export interface JsonSchema {
  type?: 'object' | 'number' | 'integer' | 'boolean' | 'string' | 'array';
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
  exclusiveMaximum?: number;
  multipleOf?: number;
  minLength?: number;
  maxLength?: number;
  minItems?: number;
  maxItems?: number;
  uniqueItems?: boolean;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  additionalProperties?: boolean;
  /** Расширение: подписи для enum */
  enumNames?: string[];
  'x-unit'?: string;
}
