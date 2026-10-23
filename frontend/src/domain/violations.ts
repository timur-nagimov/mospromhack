// Единый словарь типов нарушений: русские названия, иконки, приоритет.
// Новый тип из плагина (R7) без записи здесь тоже отображается — с названием
// из /api/rules/types и общей иконкой (см. getViolationMeta).

import type { ComponentType } from 'react';
import {
  Fence,
  Flame,
  Footprints,
  Forklift,
  HardHat,
  Package,
  ShieldAlert,
  Smartphone,
  SquareDashedMousePointer,
} from 'lucide-react';
import { LadderIcon, PersonFallenIcon } from '../components/ui/customIcons';
import type { GuardEvent, Severity } from './types';

export type IconComponent = ComponentType<{
  size?: number | string;
  strokeWidth?: number | string;
  className?: string;
  'aria-hidden'?: boolean | 'true' | 'false';
}>;

export const VIOLATION_TYPES = [
  'zone_intrusion',
  'ppe_missing',
  'fall_immobile',
  'vehicle_proximity',
  'running',
  'phone_usage',
  'smoking_fire',
  'fence_climb',
  'height_no_harness',
  'unsafe_carry',
] as const;

export type ViolationType = (typeof VIOLATION_TYPES)[number];

export interface ViolationMeta {
  type: string;
  /** Полное название — карточка события, журнал */
  title: string;
  /** Короткое — бейджи, оси графиков */
  short: string;
  description: string;
  /** Сценарий кейса S1…S10 */
  scenario: string | null;
  icon: IconComponent;
  defaultSeverity: Severity;
  /** Повышенный приоритет: красные, сверху ленты, со звуком */
  critical: boolean;
}

export const VIOLATIONS: Record<ViolationType, ViolationMeta> = {
  zone_intrusion: {
    type: 'zone_intrusion',
    title: 'Вход в опасную зону',
    short: 'Опасная зона',
    description: 'Человек находится внутри опасной зоны в часы её активности',
    scenario: 'S1',
    icon: SquareDashedMousePointer as IconComponent,
    defaultSeverity: 'high',
    critical: false,
  },
  ppe_missing: {
    type: 'ppe_missing',
    title: 'Нет каски или жилета',
    short: 'Нет СИЗ',
    description: 'Отсутствуют обязательные СИЗ: каска, сигнальный жилет или респиратор',
    scenario: 'S2',
    icon: HardHat as IconComponent,
    defaultSeverity: 'high',
    critical: false,
  },
  fall_immobile: {
    type: 'fall_immobile',
    title: 'Падение, неподвижность',
    short: 'Падение',
    description: 'Человек упал и не двигается — возможна потеря сознания',
    scenario: 'S3',
    icon: PersonFallenIcon,
    defaultSeverity: 'critical',
    critical: true,
  },
  vehicle_proximity: {
    type: 'vehicle_proximity',
    title: 'Опасное сближение с техникой',
    short: 'Сближение с техникой',
    description: 'Человек ближе допустимого к движущейся технике',
    scenario: 'S4',
    icon: Forklift as IconComponent,
    defaultSeverity: 'critical',
    critical: true,
  },
  running: {
    type: 'running',
    title: 'Бег в зоне работы техники',
    short: 'Бег',
    description: 'Бег и резкие перемещения там, где работает техника',
    scenario: 'S5',
    icon: Footprints as IconComponent,
    defaultSeverity: 'medium',
    critical: false,
  },
  phone_usage: {
    type: 'phone_usage',
    title: 'Телефон при работе',
    short: 'Телефон',
    description: 'Мобильный телефон при работе с оборудованием или при перемещении',
    scenario: 'S6',
    icon: Smartphone as IconComponent,
    defaultSeverity: 'medium',
    critical: false,
  },
  smoking_fire: {
    type: 'smoking_fire',
    title: 'Курение или открытый огонь',
    short: 'Курение, огонь',
    description: 'Курение или открытое пламя вне отведённых мест',
    scenario: 'S7',
    icon: Flame as IconComponent,
    defaultSeverity: 'high',
    critical: false,
  },
  fence_climb: {
    type: 'fence_climb',
    title: 'Перелезание через ограждение',
    short: 'Ограждение',
    description: 'Перелезание через ограждение, пролезание под конвейером или грузом',
    scenario: 'S8',
    icon: Fence as IconComponent,
    defaultSeverity: 'high',
    critical: false,
  },
  height_no_harness: {
    type: 'height_no_harness',
    title: 'Высота без страховки',
    short: 'Высота без привязи',
    description: 'Работа на высоте без страховочной привязи',
    scenario: 'S9',
    icon: LadderIcon,
    defaultSeverity: 'high',
    critical: false,
  },
  unsafe_carry: {
    type: 'unsafe_carry',
    title: 'Опасная переноска груза',
    short: 'Переноска груза',
    description: 'Переноска груза с нарушением правил',
    scenario: 'S10',
    icon: Package as IconComponent,
    defaultSeverity: 'medium',
    critical: false,
  },
};

const pluginTitles = new Map<string, string>();

/** Названия плагинов из /api/rules/types — для типов, которых нет в словаре */
export function registerPluginTitles(items: { type: string; title: string }[]): void {
  for (const item of items) pluginTitles.set(item.type, item.title);
}

export function isKnownViolation(type: string): type is ViolationType {
  return Object.prototype.hasOwnProperty.call(VIOLATIONS, type);
}

export function getViolationMeta(type: string): ViolationMeta {
  if (isKnownViolation(type)) return VIOLATIONS[type];
  const title = pluginTitles.get(type) ?? type;
  return {
    type,
    title,
    short: title,
    description: 'Тип нарушения из подключаемого модуля правил',
    scenario: null,
    icon: ShieldAlert as IconComponent,
    defaultSeverity: 'high',
    critical: false,
  };
}

/** Критическое событие: критичный тип или критическая важность в правиле */
export function isCriticalEvent(event: Pick<GuardEvent, 'type' | 'severity'>): boolean {
  return getViolationMeta(event.type).critical || event.severity === 'critical';
}
