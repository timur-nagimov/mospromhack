// Демо-данные: камеры завода, типы правил с JSON Schema, история событий за 30 дней

import type {
  CameraConfig,
  EventStatus,
  GuardEvent,
  RuleConfig,
  RuleTypeInfo,
  Severity,
  StageKey,
} from '../../domain/types';
import { STAGE_KEYS } from '../../domain/labels';
import { isZoneActive } from '../../domain/zones';
import { between, intBetween, mulberry32, pick, poisson, weighted, type Rng } from './random';

/** Размер «исходного кадра» мок-камер; bbox событий — в этих пикселях */
export const FRAME_W = 1280;
export const FRAME_H = 720;

const VEHICLE_CLASSES = ['forklift', 'truck', 'car', 'bus'];
const VEHICLE_NAMES = ['погрузчик', 'грузовик', 'легковой автомобиль', 'автобус'];

export const RULE_TYPES: RuleTypeInfo[] = [
  {
    type: 'zone_intrusion',
    title: 'Вход в опасную зону',
    required_models: ['det'],
    default_severity: 'high',
    params_schema: {
      type: 'object',
      properties: {
        classes: {
          type: 'array',
          title: 'Классы объектов',
          items: { type: 'string', enum: ['person', ...VEHICLE_CLASSES], enumNames: ['человек', ...VEHICLE_NAMES] },
          default: ['person'],
        },
        min_conf: { type: 'number', title: 'Мин. уверенность детектора', minimum: 0.1, maximum: 0.95, multipleOf: 0.05, default: 0.4 },
        anchor: {
          type: 'string',
          title: 'Точка проверки',
          enum: ['foot', 'center'],
          enumNames: ['точка опоры (низ рамки)', 'центр рамки'],
          default: 'foot',
        },
      },
    },
  },
  {
    type: 'ppe_missing',
    title: 'Нет каски или жилета',
    required_models: ['det', 'ppe'],
    default_severity: 'high',
    params_schema: {
      type: 'object',
      properties: {
        require_helmet: { type: 'boolean', title: 'Каска обязательна', default: true },
        require_vest: { type: 'boolean', title: 'Сигнальный жилет обязателен', default: true },
        require_respirator: { type: 'boolean', title: 'Респиратор обязателен', default: false },
        min_person_height_px: {
          type: 'integer',
          title: 'Мин. рост человека в кадре',
          description: 'Мельче — СИЗ не различимы, проверка пропускается',
          minimum: 24,
          maximum: 400,
          default: 80,
          'x-unit': 'px',
        },
        vote_ratio: { type: 'number', title: 'Доля кадров трека без СИЗ', minimum: 0.3, maximum: 1, multipleOf: 0.05, default: 0.7 },
      },
    },
  },
  {
    type: 'fall_immobile',
    title: 'Падение, неподвижность',
    required_models: ['det', 'pose'],
    default_severity: 'critical',
    params_schema: {
      type: 'object',
      properties: {
        immobile_s: { type: 'number', title: 'Неподвижность дольше', minimum: 2, maximum: 120, default: 8, 'x-unit': 'с' },
        lying_aspect: { type: 'number', title: 'Отношение ширины рамки к высоте', minimum: 0.8, maximum: 3, multipleOf: 0.1, default: 1.3 },
        use_pose: { type: 'boolean', title: 'Учитывать позу (ключевые точки)', default: true },
      },
    },
  },
  {
    type: 'vehicle_proximity',
    title: 'Опасное сближение с техникой',
    required_models: ['det'],
    default_severity: 'critical',
    params_schema: {
      type: 'object',
      properties: {
        distance_bh: {
          type: 'number',
          title: 'Дистанция',
          description: 'В высотах тела человека',
          minimum: 0.5,
          maximum: 5,
          multipleOf: 0.1,
          default: 1.5,
          'x-unit': 'роста',
        },
        vehicle_classes: {
          type: 'array',
          title: 'Техника',
          items: { type: 'string', enum: VEHICLE_CLASSES, enumNames: VEHICLE_NAMES },
          default: ['forklift', 'truck'],
        },
        moving_only: { type: 'boolean', title: 'Только движущаяся техника', default: true },
      },
    },
  },
  {
    type: 'running',
    title: 'Бег в зоне работы техники',
    required_models: ['det'],
    default_severity: 'medium',
    params_schema: {
      type: 'object',
      properties: {
        speed_bh: { type: 'number', title: 'Скорость выше', minimum: 0.5, maximum: 6, multipleOf: 0.1, default: 2, 'x-unit': 'роста/с' },
      },
    },
  },
  {
    type: 'phone_usage',
    title: 'Телефон при работе',
    required_models: ['det'],
    default_severity: 'medium',
    params_schema: {
      type: 'object',
      properties: {
        min_conf: { type: 'number', title: 'Мин. уверенность «cell phone»', minimum: 0.1, maximum: 0.95, multipleOf: 0.05, default: 0.35 },
        only_moving: { type: 'boolean', title: 'Только при перемещении', default: false },
      },
    },
  },
  {
    type: 'smoking_fire',
    title: 'Курение или открытый огонь',
    required_models: ['det', 'fire'],
    default_severity: 'high',
    params_schema: {
      type: 'object',
      properties: {
        detect_fire: { type: 'boolean', title: 'Пламя', default: true },
        detect_smoke: { type: 'boolean', title: 'Дым', default: true },
        min_area: { type: 'number', title: 'Мин. площадь очага', minimum: 0, maximum: 0.2, multipleOf: 0.005, default: 0.01, 'x-unit': 'доли кадра' },
      },
    },
  },
  {
    type: 'fence_climb',
    title: 'Перелезание через ограждение',
    required_models: ['det', 'pose'],
    default_severity: 'high',
    params_schema: {
      type: 'object',
      properties: {
        min_lift: { type: 'number', title: 'Подъём точки опоры', minimum: 0.1, maximum: 1.5, multipleOf: 0.05, default: 0.35, 'x-unit': 'роста' },
      },
    },
  },
  {
    type: 'height_no_harness',
    title: 'Высота без страховки',
    required_models: ['det', 'ppe'],
    default_severity: 'high',
    params_schema: {
      type: 'object',
      properties: {
        harness_conf: { type: 'number', title: 'Мин. уверенность «привязь»', minimum: 0.1, maximum: 0.95, multipleOf: 0.05, default: 0.4 },
      },
    },
  },
  {
    type: 'unsafe_carry',
    title: 'Опасная переноска груза',
    required_models: ['det', 'pose'],
    default_severity: 'medium',
    params_schema: {
      type: 'object',
      properties: {
        max_load_ratio: { type: 'number', title: 'Груз больше', minimum: 0.2, maximum: 2, multipleOf: 0.05, default: 0.6, 'x-unit': 'роста' },
        obstructed_view: { type: 'boolean', title: 'Груз закрывает обзор', default: true },
      },
    },
  },
];

function rule(
  rule_id: string,
  type: string,
  severity: Severity,
  zones: string[] | null,
  params: Record<string, unknown> = {},
  overrides: Partial<RuleConfig> = {},
): RuleConfig {
  return {
    rule_id,
    type,
    enabled: true,
    zones,
    params,
    min_duration_s: 2,
    min_ratio: 0.7,
    close_after_s: 3,
    cooldown_s: 30,
    severity,
    ...overrides,
  };
}

export function seedCameras(): CameraConfig[] {
  return [
    {
      camera_id: 'cam-01',
      name: 'Стеллажи, ряд А',
      site: 'Склад',
      source: { type: 'file', uri: 'data/videos/warehouse_racks.mp4', loop: true, realtime: true },
      sample_fps: 5,
      enabled: true,
      zones: [
        {
          zone_id: 'z-aisle',
          name: 'Проход у стеллажей',
          kind: 'danger',
          polygon: [
            [0.36, 0.5],
            [0.58, 0.5],
            [0.74, 0.97],
            [0.2, 0.97],
          ],
          active_hours: ['07:55-12:00'],
        },
        {
          zone_id: 'z-forklift',
          name: 'Проезд погрузчиков',
          kind: 'vehicle_path',
          polygon: [
            [0.6, 0.5],
            [0.97, 0.52],
            [0.99, 0.8],
            [0.76, 0.86],
          ],
          active_hours: null,
        },
      ],
      rules: [
        rule('r-01-intr', 'zone_intrusion', 'high', ['z-aisle'], { classes: ['person'], min_conf: 0.4, anchor: 'foot' }),
        rule('r-01-veh', 'vehicle_proximity', 'critical', ['z-forklift'], { distance_bh: 1.5, vehicle_classes: ['forklift'], moving_only: true }, { min_duration_s: 1 }),
        rule('r-01-fall', 'fall_immobile', 'critical', null, { immobile_s: 8, lying_aspect: 1.3, use_pose: true }, { min_duration_s: 3 }),
      ],
    },
    {
      camera_id: 'cam-02',
      name: 'Сборочная линия №1',
      site: 'Цех №1',
      source: { type: 'rtsp', uri: 'rtsp://10.0.12.21:8554/line1', loop: false, realtime: true },
      sample_fps: 5,
      enabled: true,
      zones: [
        {
          zone_id: 'z-ppe-line',
          name: 'Зона обязательных СИЗ',
          kind: 'ppe_required',
          polygon: [
            [0.04, 0.46],
            [0.96, 0.46],
            [0.99, 0.98],
            [0.01, 0.98],
          ],
          active_hours: null,
        },
        {
          zone_id: 'z-conveyor',
          name: 'Конвейер',
          kind: 'fence',
          polygon: [
            [0.08, 0.6],
            [0.92, 0.55],
            [0.94, 0.66],
            [0.06, 0.72],
          ],
          active_hours: null,
        },
      ],
      rules: [
        rule('r-02-ppe', 'ppe_missing', 'high', ['z-ppe-line'], { require_helmet: true, require_vest: true, require_respirator: false, min_person_height_px: 80, vote_ratio: 0.7 }, { min_duration_s: 3 }),
        rule('r-02-fence', 'fence_climb', 'high', ['z-conveyor'], { min_lift: 0.35 }),
        rule('r-02-phone', 'phone_usage', 'medium', null, { min_conf: 0.35, only_moving: false }, { min_duration_s: 4, severity: 'medium' }),
      ],
    },
    {
      camera_id: 'cam-03',
      name: 'Сварочный пост',
      site: 'Цех №1',
      source: { type: 'file', uri: 'data/videos/welding.mp4', loop: true, realtime: true },
      sample_fps: 4,
      enabled: true,
      zones: [
        {
          zone_id: 'z-weld',
          name: 'Участок сварки',
          kind: 'no_smoking',
          polygon: [
            [0.1, 0.42],
            [0.66, 0.42],
            [0.72, 0.96],
            [0.04, 0.96],
          ],
          active_hours: null,
        },
        {
          zone_id: 'z-smoke',
          name: 'Курилка',
          kind: 'smoking_area',
          polygon: [
            [0.74, 0.48],
            [0.96, 0.48],
            [0.98, 0.78],
            [0.76, 0.8],
          ],
          active_hours: null,
        },
      ],
      rules: [
        rule('r-03-smoke', 'smoking_fire', 'high', ['z-weld'], { detect_fire: true, detect_smoke: true, min_area: 0.01 }),
        rule('r-03-ppe', 'ppe_missing', 'high', ['z-weld'], { require_helmet: true, require_vest: false, require_respirator: true, min_person_height_px: 80, vote_ratio: 0.7 }, { min_duration_s: 3 }),
      ],
    },
    {
      camera_id: 'cam-04',
      name: 'Погрузочная рампа',
      site: 'Логистика',
      source: { type: 'rtsp', uri: 'rtsp://10.0.12.34:8554/dock', loop: false, realtime: true },
      sample_fps: 5,
      enabled: true,
      zones: [
        {
          zone_id: 'z-dock-path',
          name: 'Проезд техники',
          kind: 'vehicle_path',
          polygon: [
            [0.3, 0.52],
            [0.7, 0.52],
            [0.86, 0.98],
            [0.14, 0.98],
          ],
          active_hours: null,
        },
      ],
      rules: [
        rule('r-04-veh', 'vehicle_proximity', 'critical', ['z-dock-path'], { distance_bh: 1.5, vehicle_classes: ['forklift', 'truck'], moving_only: true }, { min_duration_s: 1 }),
        rule('r-04-run', 'running', 'medium', ['z-dock-path'], { speed_bh: 2 }),
        rule('r-04-carry', 'unsafe_carry', 'medium', null, { max_load_ratio: 0.6, obstructed_view: true }),
      ],
    },
    {
      camera_id: 'cam-05',
      name: 'Монтаж металлоконструкций',
      site: 'Цех №3',
      source: { type: 'images', uri: 'data/images/assembly/', loop: true, realtime: true },
      sample_fps: 3,
      enabled: true,
      zones: [
        {
          zone_id: 'z-scaffold',
          name: 'Леса, отметка +6 м',
          kind: 'height',
          polygon: [
            [0.12, 0.08],
            [0.88, 0.08],
            [0.88, 0.44],
            [0.12, 0.44],
          ],
          active_hours: ['08:00-20:00'],
        },
      ],
      rules: [
        rule('r-05-height', 'height_no_harness', 'high', ['z-scaffold'], { harness_conf: 0.4 }),
        rule('r-05-fall', 'fall_immobile', 'critical', null, { immobile_s: 8, lying_aspect: 1.3, use_pose: true }, { min_duration_s: 3 }),
      ],
    },
    {
      camera_id: 'cam-06',
      name: 'Северное ограждение',
      site: 'Периметр',
      source: { type: 'rtsp', uri: 'rtsp://10.0.12.50:8554/north', loop: false, realtime: true },
      sample_fps: 3,
      enabled: true,
      zones: [
        {
          zone_id: 'z-fence-north',
          name: 'Ограждение',
          kind: 'fence',
          polygon: [
            [0.0, 0.42],
            [1.0, 0.36],
            [1.0, 0.56],
            [0.0, 0.62],
          ],
          active_hours: null,
        },
      ],
      rules: [
        rule('r-06-fence', 'fence_climb', 'high', ['z-fence-north'], { min_lift: 0.35 }),
        rule('r-06-smoke', 'smoking_fire', 'medium', null, { detect_fire: true, detect_smoke: false, min_area: 0.01 }, { severity: 'medium' }),
      ],
    },
  ];
}

// ---------- Генерация событий ----------

const TYPE_WEIGHT: Record<string, number> = {
  ppe_missing: 3.2,
  zone_intrusion: 2.4,
  phone_usage: 1.6,
  vehicle_proximity: 0.45,
  running: 1.0,
  smoking_fire: 0.8,
  height_no_harness: 0.8,
  unsafe_carry: 0.7,
  fence_climb: 0.6,
  fall_immobile: 0.12,
};

/** Доля ложных срабатываний по типам — даёт правдоподобную «точность» */
const FALSE_RATE: Record<string, number> = {
  phone_usage: 0.27,
  running: 0.21,
  unsafe_carry: 0.19,
  ppe_missing: 0.16,
  smoking_fire: 0.14,
  height_no_harness: 0.13,
  fence_climb: 0.11,
  fall_immobile: 0.12,
  vehicle_proximity: 0.09,
  zone_intrusion: 0.07,
};

const CAMERA_WEIGHT: Record<string, number> = {
  'cam-01': 1.3,
  'cam-02': 1.5,
  'cam-03': 0.8,
  'cam-04': 1.2,
  'cam-05': 0.7,
  'cam-06': 0.5,
};

const HOUR_PROFILE = [
  0.12, 0.08, 0.06, 0.06, 0.08, 0.15, 0.35, 0.8, 1.15, 1.3, 1.45, 1.2, 0.65, 1.05, 1.4, 1.25, 1.0, 0.7, 0.45, 0.35, 0.3, 0.25,
  0.2, 0.15,
];

const NOTES: Record<Exclude<EventStatus, 'new'>, string[]> = {
  confirmed: [
    'Нарушитель проинструктирован на месте',
    'Бригадир уведомлён, внесено в журнал смены',
    'Подтверждено по клипу',
    'Повторное нарушение, бригада 3',
  ],
  forwarded: ['Передано мастеру смены', 'Мастер участка выехал на место', 'Вызван медпункт, мастер уведомлён'],
  false: [
    'Ложное: отражение в остеклении',
    'Ложное: манекен у стеллажа',
    'Ложное: каска в руке у мастера, проверка',
    'Ложное: блик сварки',
  ],
};

function makeLatency(rng: Rng): Record<string, number> {
  const ranges: Record<StageKey, [number, number]> = {
    decode: [2.6, 5.5],
    preprocess: [1.8, 3.9],
    transfer: [12, 31],
    queue: [1.5, 14],
    infer: [8.2, 12.4],
    postprocess: [1.2, 3.1],
    tracking: [0.7, 2.1],
    rules: [0.3, 1.3],
    alert: [0.5, 3.2],
  };
  const out: Record<string, number> = {};
  let total = 0;
  for (const key of STAGE_KEYS) {
    const [a, b] = ranges[key];
    const v = Math.round(between(rng, a, b) * 10) / 10;
    out[key] = v;
    total += v;
  }
  out.total = Math.round(total * 10) / 10;
  return out;
}

function makeBBox(rng: Rng, type: string): GuardEvent['bbox'] {
  if (type === 'fall_immobile') {
    const w = between(rng, 190, 250);
    const h = between(rng, 70, 100);
    const x = between(rng, 260, 820);
    const y = between(rng, 470, 590);
    return [Math.round(x), Math.round(y), Math.round(x + w), Math.round(y + h)];
  }
  const h = between(rng, 170, 270);
  const w = h * between(rng, 0.36, 0.46);
  const x = between(rng, 200, 980);
  const y = between(rng, 300, 680 - h);
  return [Math.round(x), Math.round(y), Math.round(x + w), Math.round(y + h)];
}

function makeDetails(rng: Rng, type: string): Record<string, unknown> {
  switch (type) {
    case 'ppe_missing': {
      const missing = rng() < 0.55 ? ['helmet'] : rng() < 0.6 ? ['vest'] : ['helmet', 'vest'];
      return { missing, person_conf: Math.round(between(rng, 0.71, 0.96) * 100) / 100 };
    }
    case 'vehicle_proximity':
      return { vehicle: rng() < 0.7 ? 'forklift' : 'truck', distance_bh: Math.round(between(rng, 0.5, 1.4) * 10) / 10 };
    case 'fall_immobile':
      return { immobile_s: Math.round(between(rng, 8, 25)), pose_state: 'lying' };
    case 'running':
      return { speed_bh: Math.round(between(rng, 2.1, 3.6) * 10) / 10 };
    case 'phone_usage':
      return { phone_conf: Math.round(between(rng, 0.38, 0.9) * 100) / 100 };
    case 'smoking_fire':
      return { source: rng() < 0.7 ? 'smoking' : 'fire' };
    default:
      return {};
  }
}

let seq = 0;

export function makeEventId(ts: number): string {
  seq += 1;
  const d = new Date(ts * 1000);
  const code = `${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
  return `ev-${code}-${String(seq).padStart(4, '0')}`;
}

/** Правила камеры, которые могут сработать в момент ts (с учётом расписания зон) */
function activeRules(camera: CameraConfig, ts: number): RuleConfig[] {
  return camera.rules.filter((r) => {
    if (!r.enabled) return false;
    if (!r.zones || r.zones.length === 0) return true;
    return r.zones.some((zid) => {
      const zone = camera.zones.find((z) => z.zone_id === zid);
      return zone ? isZoneActive(zone, ts) : false;
    });
  });
}

export interface EventDraft {
  camera: CameraConfig;
  rule: RuleConfig;
  confirmedAt: number;
}

export function buildEvent(rng: Rng, draft: EventDraft, status: EventStatus, ended: boolean): GuardEvent {
  const { camera, rule: r, confirmedAt } = draft;
  const window = r.min_duration_s + between(rng, 0.15, 0.9);
  const zone_id = r.zones && r.zones.length > 0 ? r.zones[0] : null;
  const note = status === 'new' ? null : rng() < 0.55 ? pick(rng, NOTES[status]) : null;
  const id = makeEventId(confirmedAt);
  return {
    id,
    camera_id: camera.camera_id,
    site: camera.site,
    zone_id,
    rule_id: r.rule_id,
    type: r.type,
    severity: r.severity,
    track_id: intBetween(rng, 3, 480),
    started_at: confirmedAt - window,
    confirmed_at: confirmedAt,
    ended_at: ended ? confirmedAt + between(rng, 4, 75) : null,
    bbox: makeBBox(rng, r.type),
    snapshot_path: `media/${id}.jpg`,
    clip_path: `media/${id}.mp4`,
    status,
    operator_note: note,
    details: makeDetails(rng, r.type),
    latency_ms: makeLatency(rng),
  };
}

export function pickDraft(rng: Rng, cameras: CameraConfig[], ts: number): EventDraft | null {
  const candidates = cameras.filter((c) => c.enabled && activeRules(c, ts).length > 0);
  if (candidates.length === 0) return null;
  const camera = weighted(rng, candidates, (c) => CAMERA_WEIGHT[c.camera_id] ?? 1);
  const rules = activeRules(camera, ts);
  const r = weighted(rng, rules, (x) => TYPE_WEIGHT[x.type] ?? 0.5);
  return { camera, rule: r, confirmedAt: ts };
}

function reviewedStatus(rng: Rng, type: string, severity: Severity): Exclude<EventStatus, 'new'> {
  if (rng() < (FALSE_RATE[type] ?? 0.15)) return 'false';
  const forwardShare = severity === 'critical' ? 0.6 : 0.22;
  return rng() < forwardShare ? 'forwarded' : 'confirmed';
}

/** История за days суток до now: ~20 событий в будний день, пики в смены */
export function seedHistory(cameras: CameraConfig[], now: number, days = 30): GuardEvent[] {
  const rng = mulberry32(20261023);
  const events: GuardEvent[] = [];
  const startHour = Math.floor(now / 3600) - days * 24;
  for (let hour = startHour; hour < Math.floor(now / 3600); hour += 1) {
    const ts0 = hour * 3600;
    const d = new Date(ts0 * 1000);
    const weekday = (d.getDay() + 6) % 7;
    const dayFactor = weekday === 5 ? 0.45 : weekday === 6 ? 0.3 : 1;
    const lambda = 1.25 * HOUR_PROFILE[d.getHours()] * dayFactor;
    const n = poisson(rng, lambda);
    for (let i = 0; i < n; i += 1) {
      const ts = ts0 + rng() * 3600;
      if (ts > now - 90) continue;
      const draft = pickDraft(rng, cameras, ts);
      if (!draft) continue;
      const age = now - ts;
      let status: EventStatus;
      if (age < 2.5 * 3600) status = rng() < 0.72 ? 'new' : reviewedStatus(rng, draft.rule.type, draft.rule.severity);
      else status = rng() < 0.035 ? 'new' : reviewedStatus(rng, draft.rule.type, draft.rule.severity);
      events.push(buildEvent(rng, draft, status, true));
    }
  }
  return events.sort((a, b) => b.confirmed_at - a.confirmed_at);
}
