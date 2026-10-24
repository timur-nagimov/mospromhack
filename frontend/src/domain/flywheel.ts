// TS-зеркало контракта маховика данных (docs/FLYWHEEL.md: «Модель данных», «Карточки моделей», «API»).
// Бэкенд пишется параллельно, поэтому всё, без чего страница может жить, — опционально,
// а разбор ответов мягкий: неизвестные поля и значения не ломают интерфейс.

import type { BBox } from './types';

export type SampleSource = 'event' | 'uncertain' | 'flicker' | 'disagreement' | 'operator' | 'import';
export type SampleStatus = 'new' | 'auto' | 'accepted' | 'rejected';
export type OperatorLabel = 'positive' | 'hard_negative' | 'forwarded';
export type LabelOrigin = 'teacher' | 'operator' | 'model' | 'dataset';

/** Детекция модели на сохранённом кадре; bbox — пиксели этого кадра */
export interface SampleDetection {
  model_id: string;
  cls: string;
  conf: number;
  bbox: BBox;
  track_id?: number | null;
}

/** Предложенная (учитель) или итоговая (оператор) разметка */
export interface SampleLabel {
  cls: string;
  bbox: BBox;
  origin: LabelOrigin | string;
  conf?: number | null;
}

export interface FlywheelSample {
  id: string;
  /** epoch, секунды: время кадра */
  created_at: number;
  camera_id: string;
  source: SampleSource | string;
  event_id: string | null;
  frame_path?: string;
  width: number;
  height: number;
  detections: SampleDetection[];
  labels: SampleLabel[];
  status: SampleStatus | string;
  operator_label: OperatorLabel | string | null;
  reason: string;
  /** 1 — кадр отложенного ролика, в обучение не экспортируется */
  holdout: number | boolean;
  reviewed_at: number | null;
  dataset_version: string | null;
  frame_url?: string;
}

/** Короткая форма сэмпла из WS-сообщения flywheel_sample: гарантирован только id */
export type FlywheelSampleBrief = Pick<FlywheelSample, 'id'> & Partial<Omit<FlywheelSample, 'id'>>;

export interface SamplePage {
  items: FlywheelSample[];
  total: number;
}

export interface SampleQuery {
  status?: string;
  source?: string;
  camera_id?: string;
  cls?: string;
  holdout?: 0 | 1;
  event_id?: string;
  limit?: number;
  offset?: number;
  order?: string;
}

export interface SamplePatch {
  status?: 'accepted' | 'rejected';
  labels?: SampleLabel[];
}

export interface DatasetBuildRequest {
  include_holdout?: boolean;
  classes?: string[];
}

export interface DatasetVersion {
  /** fw-YYYYMMDD-HHMM */
  version: string;
  created_at?: number | string | null;
  path?: string | null;
  classes?: string[];
  n_train?: number;
  n_val?: number;
  n_hard_negative?: number;
  /** Источники и смеси открытых наборов, sha256 списков файлов; форма не зафиксирована */
  manifest?: Record<string, unknown> | null;
}

export interface ModelEventsMetrics {
  videos?: string[];
  tp?: number;
  fp?: number;
  fn?: number;
  duplicates?: number;
  latency_p50_s?: number;
  /** ложных тревог в час: сверх примера контракта, читаем если есть */
  fp_per_hour?: number;
  hours?: number;
}

export interface ModelDatasetTest {
  name?: string;
  images?: number;
  map50?: number;
  per_class?: Record<string, number>;
}

export interface ModelHoldout {
  dataset_test?: ModelDatasetTest | null;
  events?: ModelEventsMetrics | null;
}

export type ModelCardStatus = 'candidate' | 'deployed' | 'rejected' | 'archived';

export interface ModelCard {
  model_id: string;
  version: string;
  parent?: string | null;
  created_at?: string | number | null;
  task?: string;
  classes?: string[];
  train?: {
    datasets?: { name: string; images?: number; url?: string }[];
    epochs?: number;
    imgsz?: number;
    device?: string;
    minutes?: number;
  } | null;
  weights?: string | null;
  onnx?: string | null;
  tpu_program?: string | null;
  holdout?: ModelHoldout | null;
  baseline?: { version?: string; holdout?: ModelHoldout | null } | null;
  tpu_check?: { frames?: number; recall?: number; precision?: number } | null;
  gate?: { promote: boolean; reasons?: string[] } | null;
  status: ModelCardStatus | string;
}

export type JobKind = 'autolabel' | 'export' | 'train' | 'gate' | 'deploy';

export interface FlywheelJob {
  id: string;
  kind: JobKind | string;
  /** queued / running / done / failed — значения бэкенда читаем мягко (jobPhase) */
  status: string;
  log_url?: string | null;
  args?: Record<string, unknown>;
  created_at?: number | string | null;
  started_at?: number | string | null;
  finished_at?: number | string | null;
  /** хвост лога: строка или строки */
  log_tail?: string | string[] | null;
  tail?: string | string[] | null;
  error?: string | null;
  returncode?: number | null;
}

export interface FlywheelSummary {
  total: number;
  today: number;
  by_source: Record<string, number>;
  by_status: Record<string, number>;
  by_class: Record<string, number>;
  reviewed: number;
  holdout: number;
  /** сэмплы, отброшенные при переполнении очереди записи */
  dropped: number;
  last_dataset: DatasetVersion | string | null;
  models: { deployed: (string | Partial<ModelCard>)[]; candidate: (string | Partial<ModelCard>)[] };
  collecting: boolean;
  /** сверх контракта: разрешены ли train/deploy (GUARD_FLYWHEEL_ALLOW_TRAIN) */
  allow_train?: boolean;
}

// ---------------------------------------------------------------- словари

/** Порядок источников в счётчиках и фильтрах (как в контракте) */
export const SOURCE_ORDER: SampleSource[] = ['event', 'uncertain', 'flicker', 'disagreement', 'operator', 'import'];

export const SOURCE_LABELS: Record<SampleSource, string> = {
  event: 'Событие',
  uncertain: 'Неуверенность',
  flicker: 'Мигание трека',
  disagreement: 'Расхождение TPU и CPU',
  operator: 'Оператор',
  import: 'Импорт',
};

export const SOURCE_HINTS: Record<SampleSource, string> = {
  event: 'Кадр подтверждения каждого события',
  uncertain: 'Уверенность модели у порога правила',
  flicker: 'Трек пропал, рядом родился новый ID',
  disagreement: 'TPU int8 и CPU fp32 не сошлись',
  operator: 'Оператор отправил кадр вручную',
  import: 'Открытый набор или ролик (CLI)',
};

/** Категориальная палитра тёмного режима (та же, что на графиках) */
export const SOURCE_COLORS: Record<SampleSource, string> = {
  event: '#d95926',
  uncertain: '#c98500',
  flicker: '#9085e9',
  disagreement: '#3987e5',
  operator: '#199e70',
  import: '#8a96a3',
};

export const SAMPLE_STATUS_ORDER: SampleStatus[] = ['new', 'auto', 'accepted', 'rejected'];

export const SAMPLE_STATUS_LABELS: Record<SampleStatus, string> = {
  new: 'Новый',
  auto: 'Размечен учителем',
  accepted: 'Принят',
  rejected: 'Отклонён',
};

export const SAMPLE_STATUS_COLORS: Record<SampleStatus, string> = {
  new: 'var(--st-new)',
  auto: 'var(--series-1)',
  accepted: 'var(--st-confirmed)',
  rejected: 'var(--st-false)',
};

export const OPERATOR_LABEL_TEXT: Record<OperatorLabel, string> = {
  positive: 'Подтверждено оператором',
  hard_negative: 'Ложное: рамка пойдёт как hard negative',
  forwarded: 'Передано мастеру',
};

export const ORIGIN_LABELS: Record<LabelOrigin, string> = {
  teacher: 'учитель',
  operator: 'оператор',
  model: 'модель',
  dataset: 'набор',
};

export const MODEL_STATUS_LABELS: Record<ModelCardStatus, string> = {
  candidate: 'Кандидат',
  deployed: 'Выкачена',
  rejected: 'Отклонена гейтом',
  archived: 'В архиве',
};

export const JOB_KIND_LABELS: Record<JobKind, string> = {
  autolabel: 'Авторазметка',
  export: 'Сборка датасета',
  train: 'Дообучение',
  gate: 'Гейт',
  deploy: 'Выкатка',
};

/** Названия классов детекторов и учителя по-русски; неизвестный класс показываем как есть */
export const CLASS_LABELS: Record<string, string> = {
  person: 'человек',
  forklift: 'погрузчик',
  truck: 'грузовик',
  car: 'автомобиль',
  helmet: 'каска',
  'hard hat': 'каска',
  hardhat: 'каска',
  no_helmet: 'без каски',
  'no-hardhat': 'без каски',
  vest: 'жилет',
  'safety vest': 'жилет',
  no_vest: 'без жилета',
  'no-safety vest': 'без жилета',
  cigarette: 'сигарета',
  fire: 'огонь',
  smoke: 'дым',
  phone: 'телефон',
  'cell phone': 'телефон',
  fallen: 'упавший',
};

/** Классы, которые всегда есть в списке смены класса рамки */
export const DEFAULT_CLASSES = ['person', 'forklift', 'helmet', 'no_helmet', 'vest', 'no_vest', 'cigarette', 'fire', 'smoke'];

const CLASS_COLORS: Record<string, string> = {
  person: '#3dd68c',
  forklift: '#f2b51f',
  truck: '#f2b51f',
  helmet: '#5b9cf5',
  no_helmet: '#ff5a5f',
  vest: '#2fc6c6',
  no_vest: '#f0803c',
  cigarette: '#e070b0',
  fire: '#ff7a3d',
  smoke: '#b9b6ae',
};
const EXTRA_COLORS = ['#9085e9', '#e9b22b', '#5fd0e8', '#d55181', '#8bd450', '#f09a6a'];

// ---------------------------------------------------------------- помощники

export function sourceLabel(source: string | null | undefined): string {
  if (!source) return '—';
  return SOURCE_LABELS[source as SampleSource] ?? source;
}

export function sourceColor(source: string | null | undefined): string {
  return SOURCE_COLORS[source as SampleSource] ?? 'var(--text-3)';
}

export function sampleStatusLabel(status: string | null | undefined): string {
  if (!status) return '—';
  return SAMPLE_STATUS_LABELS[status as SampleStatus] ?? status;
}

export function sampleStatusColor(status: string | null | undefined): string {
  return SAMPLE_STATUS_COLORS[status as SampleStatus] ?? 'var(--text-3)';
}

export function modelStatusLabel(status: string | null | undefined): string {
  if (!status) return '—';
  return MODEL_STATUS_LABELS[status as ModelCardStatus] ?? status;
}

export function jobKindLabel(kind: string | null | undefined): string {
  if (!kind) return 'Задача';
  return JOB_KIND_LABELS[kind as JobKind] ?? kind;
}

export function classLabel(cls: string): string {
  return CLASS_LABELS[cls] ?? CLASS_LABELS[cls.toLowerCase()] ?? cls;
}

/** Стабильный цвет рамки по классу: известные классы — фиксированно, прочие — по хешу */
export function classColor(cls: string): string {
  const known = CLASS_COLORS[cls] ?? CLASS_COLORS[cls.toLowerCase()];
  if (known) return known;
  let h = 0;
  for (const ch of cls) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return EXTRA_COLORS[h % EXTRA_COLORS.length];
}

export function isHoldout(sample: Pick<FlywheelSample, 'holdout'>): boolean {
  return sample.holdout === true || Number(sample.holdout) === 1;
}

/** Сэмпл ещё ждёт решения человека */
export function needsReview(sample: Pick<FlywheelSample, 'status'>): boolean {
  return sample.status === 'new' || sample.status === 'auto';
}

/** Фаза задачи по статусу бэкенда: значения не зафиксированы контрактом */
export type JobPhase = 'active' | 'done' | 'failed' | 'unknown';

export function jobPhase(status: string | null | undefined): JobPhase {
  const s = (status ?? '').toLowerCase();
  if (['queued', 'pending', 'running', 'started', 'in_progress'].includes(s)) return 'active';
  if (['done', 'ok', 'success', 'succeeded', 'finished', 'completed'].includes(s)) return 'done';
  if (['failed', 'error', 'crashed', 'cancelled', 'canceled', 'forbidden'].includes(s)) return 'failed';
  return 'unknown';
}

export function jobLogTail(job: FlywheelJob): string[] {
  const tail = job.log_tail ?? job.tail;
  if (!tail) return [];
  return (Array.isArray(tail) ? tail : tail.split('\n')).filter((line) => line.trim() !== '');
}

/** Время из epoch-секунд, epoch-миллисекунд или ISO-строки → epoch-секунды */
export function toEpoch(value: number | string | null | undefined): number | null {
  if (value == null || value === '') return null;
  if (typeof value === 'number') return value > 1e12 ? value / 1000 : value;
  const asNumber = Number(value);
  if (Number.isFinite(asNumber)) return asNumber > 1e12 ? asNumber / 1000 : asNumber;
  const ms = Date.parse(value);
  return Number.isFinite(ms) ? ms / 1000 : null;
}

/** Версия датасета из summary.last_dataset: объект или строка */
export function datasetVersionOf(value: DatasetVersion | string | null | undefined): string | null {
  if (!value) return null;
  return typeof value === 'string' ? value : value.version ?? null;
}

/** Версии моделей из summary.models: строки или объекты */
export function modelVersions(list: (string | Partial<ModelCard>)[] | null | undefined): string[] {
  return (list ?? [])
    .map((m) => (typeof m === 'string' ? m : m.version ?? m.model_id ?? ''))
    .filter((v): v is string => Boolean(v));
}

/** Ссылка на источник набора: только http(s), иначе не делаем ссылку (javascript: и т. п.) */
export function safeUrl(url: unknown): string | undefined {
  return typeof url === 'string' && /^https?:\/\//i.test(url) ? url : undefined;
}

export interface ManifestSource {
  name: string;
  count: number | null;
  url?: string;
}

/**
 * Источники из манифеста версии датасета. Форма в контракте не зафиксирована,
 * поэтому понимаем массив объектов и словарь «имя → число или объект».
 */
export function manifestSources(manifest: Record<string, unknown> | null | undefined): ManifestSource[] {
  if (!manifest) return [];
  const raw = manifest.sources ?? manifest.mix ?? manifest.mixes ?? manifest.datasets;
  const count = (v: Record<string, unknown>): number | null => {
    const n = v.images ?? v.count ?? v.n ?? v.n_images ?? v.samples;
    return typeof n === 'number' ? n : null;
  };
  if (Array.isArray(raw)) {
    return raw
      .filter((v): v is Record<string, unknown> => Boolean(v) && typeof v === 'object')
      .map((v) => ({
        name: String(v.name ?? v.source ?? v.dataset ?? '—'),
        count: count(v),
        url: safeUrl(v.url),
      }));
  }
  if (raw && typeof raw === 'object') {
    return Object.entries(raw as Record<string, unknown>).map(([name, v]) => {
      if (typeof v === 'number') return { name, count: v };
      if (v && typeof v === 'object') {
        const obj = v as Record<string, unknown>;
        return { name, count: count(obj), url: safeUrl(obj.url) };
      }
      return { name, count: null };
    });
  }
  return [];
}

/** Recall и precision по событиям из tp/fp/fn; null — данных нет */
export function eventRates(events: ModelEventsMetrics | null | undefined): { recall: number | null; precision: number | null } {
  if (!events) return { recall: null, precision: null };
  const tp = events.tp ?? null;
  const fp = events.fp ?? null;
  const fn = events.fn ?? null;
  const recall = tp != null && fn != null && tp + fn > 0 ? tp / (tp + fn) : null;
  const precision = tp != null && fp != null && tp + fp > 0 ? tp / (tp + fp) : null;
  return { recall, precision };
}
