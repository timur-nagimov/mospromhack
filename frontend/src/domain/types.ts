// TS-зеркало контрактов guard/contracts.py и docs/CONTRACTS.md (раздел «HTTP API»).
// Меняется синхронно с контрактом; всё, что сверх контракта, помечено как опциональное.

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

export type StatsGroupBy = 'type' | 'camera' | 'zone' | 'hour' | 'day';

export interface StatsItem {
  key: string;
  count: number;
}

export interface StatsResponse {
  group_by: StatsGroupBy;
  items: StatsItem[];
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
}

/** backend.stats(): форма в контракте не зафиксирована, читаем известные поля мягко */
export interface InferenceStats {
  fps?: number;
  errors?: number;
  /** доля занятости ускорителя 0..1 — ожидание сверх контракта */
  utilization?: number;
  timings_ms?: Partial<Record<string, Percentiles>>;
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
}

export interface Health {
  status: string;
  backend: string;
  cameras_running: number;
}

export type WsMessage =
  | { kind: 'event'; event: GuardEvent }
  | { kind: 'event_update'; event: GuardEvent }
  | { kind: 'metrics'; data: MetricsSnapshot };

/** Подмножество JSON Schema, которое понимает генератор форм */
export interface JsonSchema {
  type?: 'object' | 'number' | 'integer' | 'boolean' | 'string' | 'array';
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  multipleOf?: number;
  minLength?: number;
  maxLength?: number;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  /** Расширение: подписи для enum */
  enumNames?: string[];
  'x-unit'?: string;
}
