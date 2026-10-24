import type { EventStatus, Severity, SourceType, StageKey, ZoneKind } from './types';

export const SEVERITY_ORDER: Severity[] = ['critical', 'high', 'medium', 'low'];

export const SEVERITY_LABELS: Record<Severity, string> = {
  critical: 'Критическая',
  high: 'Высокая',
  medium: 'Средняя',
  low: 'Низкая',
};

export const STATUS_ORDER: EventStatus[] = ['new', 'confirmed', 'forwarded', 'false'];

export const STATUS_LABELS: Record<EventStatus, string> = {
  new: 'Не разобрано',
  confirmed: 'Подтверждено',
  forwarded: 'Передано мастеру',
  false: 'Ложное',
};

/** Решения оператора в порядке хоткеев 1/2/3 */
export const OPERATOR_DECISIONS: { status: Exclude<EventStatus, 'new'>; label: string; hotkey: string }[] = [
  { status: 'confirmed', label: 'Подтверждено', hotkey: '1' },
  { status: 'false', label: 'Ложное', hotkey: '2' },
  { status: 'forwarded', label: 'Передано мастеру', hotkey: '3' },
];

export const ZONE_KIND_LABELS: Record<ZoneKind, string> = {
  danger: 'Опасная зона',
  ppe_required: 'Обязательны СИЗ',
  no_smoking: 'Курение запрещено',
  smoking_area: 'Место для курения',
  vehicle_path: 'Проезд техники',
  height: 'Работа на высоте',
  fence: 'Ограждение',
};

/** Цвет полигона зоны в редакторе и на кадре */
export const ZONE_KIND_COLORS: Record<ZoneKind, string> = {
  danger: '#e5484d',
  ppe_required: '#f2b51f',
  no_smoking: '#f0803c',
  smoking_area: '#3dbb72',
  vehicle_path: '#f0803c',
  height: '#9085e9',
  fence: '#5b9cf5',
};

export const SOURCE_TYPE_LABELS: Record<SourceType, string> = {
  file: 'Видеофайл',
  rtsp: 'Сетевой поток',
  images: 'Папка изображений',
  webcam: 'Веб-камера',
};

/** Подписи в списке «Тип источника»: тип rtsp принимает RTSP, HTTP MJPEG, HLS и ссылки YouTube */
export const SOURCE_TYPE_OPTION_LABELS: Record<SourceType, string> = {
  ...SOURCE_TYPE_LABELS,
  rtsp: 'Сетевой поток (RTSP / HTTP / HLS)',
};

export const STAGE_KEYS: StageKey[] = [
  'decode',
  'preprocess',
  'transfer',
  'queue',
  'infer',
  'postprocess',
  'tracking',
  'rules',
  'alert',
];

export const TIMING_LABELS: Record<string, string> = {
  decode: 'Декодирование',
  preprocess: 'Препроцессинг',
  transfer: 'Передача',
  queue: 'Очередь',
  infer: 'Инференс',
  postprocess: 'Постпроцессинг',
  tracking: 'Трекинг',
  rules: 'Правила',
  alert: 'Оповещение',
  total: 'Сквозная',
};

/** Где выполняется этап: определяет оттенок сегмента в графике задержек */
export const STAGE_PHASE: Record<StageKey, 'host-pre' | 'link' | 'accel' | 'host-post'> = {
  decode: 'host-pre',
  preprocess: 'host-pre',
  transfer: 'link',
  queue: 'link',
  infer: 'accel',
  postprocess: 'host-post',
  tracking: 'host-post',
  rules: 'host-post',
  alert: 'host-post',
};

export const STAGE_PHASE_LABELS = {
  'host-pre': 'Сервер, до инференса',
  link: 'Канал до ускорителя',
  accel: 'Ускоритель',
  'host-post': 'Сервер, после инференса',
} as const;

export const BACKEND_LABELS: Record<string, string> = {
  'linq-tpu': 'LinQ TPU',
  'onnx-cpu': 'ONNX CPU',
};

export function backendLabel(name: string | null | undefined): string {
  if (!name) return 'нет данных';
  return BACKEND_LABELS[name] ?? name;
}
