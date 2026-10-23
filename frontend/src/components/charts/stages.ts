import { STAGE_KEYS, STAGE_PHASE, TIMING_LABELS } from '../../domain/labels';
import type { StageKey } from '../../domain/types';

/**
 * Цвет этапа = где он выполняется (оттенок) + порядок внутри фазы (светлота):
 * сервер до инференса — синий, канал до ускорителя — фиолетовый,
 * ускоритель — оранжевый (главный акцент), сервер после — бирюзовый.
 * Соседние сегменты разделены зазором 2 px, легенда и таблица дублируют цвет.
 */
export const STAGE_COLORS: Record<StageKey, string> = {
  decode: '#8fbcf0',
  preprocess: '#4f8fe0',
  transfer: '#b0a6f2',
  queue: '#7d70d9',
  infer: '#e8703a',
  postprocess: '#74d8b1',
  tracking: '#3fbf92',
  rules: '#22a077',
  alert: '#167c5a',
};

export const STAGES = STAGE_KEYS.map((key) => ({
  key,
  label: TIMING_LABELS[key],
  color: STAGE_COLORS[key],
  phase: STAGE_PHASE[key],
}));
