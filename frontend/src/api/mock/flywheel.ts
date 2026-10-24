// Мок маховика данных: сэмплы с кадрами синтетической сцены (рамки совпадают с людьми
// и техникой на картинке), сводка, версии датасетов, карточки моделей и задачи.
// Новые сэмплы приходят по фейковому WS (flywheel_sample): по таймеру и на каждое
// живое событие. train и deploy отвечают 403, как бэкенд без GUARD_FLYWHEEL_ALLOW_TRAIN.

import type {
  DatasetVersion,
  FlywheelJob,
  FlywheelSample,
  FlywheelSampleBrief,
  FlywheelSummary,
  JobKind,
  ModelCard,
  OperatorLabel,
  SampleDetection,
  SampleLabel,
  SampleQuery,
  SampleSource,
  SampleStatus,
} from '../../domain/flywheel';
import { isHoldout } from '../../domain/flywheel';
import { nowSec, startOfDay } from '../../domain/format';
import type { BBox, CameraConfig, EventStatus, GuardEvent, WsMessage } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';
import type { FlywheelApi } from '../types';
import { FRAME_W } from './data';
import { between, intBetween, mulberry32, pick, weighted, type Rng } from './random';
import { renderToDataUrl, sceneObjects, type SceneObject } from './scene';

export interface FlywheelMockDeps {
  cameras: () => CameraConfig[];
  events: () => GuardEvent[];
  emit: (message: WsMessage) => void;
  respond: <T>(value: T, min?: number, max?: number) => Promise<T>;
  fail: (status: number, message: string) => Promise<never>;
}

export interface FlywheelMock {
  api: FlywheelApi;
  start(): void;
  stop(): void;
  /** Новое живое событие → сэмпл source = event */
  onEvent(event: GuardEvent): void;
  /** Решение оператора по событию переносится в operator_label сэмпла */
  onEventPatch(event: GuardEvent): void;
}

const DAY = 86400;
const HISTORY_SAMPLES = 420;
const HOLDOUT_CAMERA = 'cam_s4';

/** Открытые наборы: источник записываем (docs/DATASETS.md) */
const OPEN_SETS = {
  forklift: { id: 'forklift-rf', name: 'Roboflow forklift', url: 'https://huggingface.co/datasets/keremberke/forklift-object-detection' },
  sh17: { id: 'sh17', name: 'SH17', url: 'https://huggingface.co/datasets/fathansanum/SH-17-Dataset' },
  ppe: { id: 'construction-ppe', name: 'Construction-PPE (Ultralytics)', url: 'https://docs.ultralytics.com/datasets/detect/construction-ppe/' },
} as const;

/** Камеры, кадры которых сохраняются целиком (шлюз на ноутбуке); edge присылает ≤ 960 */
const FULL_SIZE = new Set(['cam-01', 'cam-03']);

const CAMERA_WEIGHT: Record<string, number> = {
  'cam-01': 1.4,
  'cam-02': 1.3,
  'cam-03': 0.8,
  'cam-04': 1.2,
  'cam-05': 0.6,
  'cam-06': 0.4,
  [HOLDOUT_CAMERA]: 0.35,
};

const SOURCE_WEIGHT: Record<Exclude<SampleSource, 'import'>, number> = {
  event: 0.2,
  uncertain: 0.38,
  flicker: 0.18,
  disagreement: 0.1,
  operator: 0.05,
};

const OPERATOR_LABEL_BY_STATUS: Partial<Record<EventStatus, OperatorLabel>> = {
  confirmed: 'positive',
  false: 'hard_negative',
  forwarded: 'forwarded',
};

/** Объект сцены; нарушитель события может лежать (падение) */
type MockObject = SceneObject & { lying?: boolean; violator?: boolean };

/** Что нужно, чтобы нарисовать кадр сэмпла */
interface RenderInfo {
  sceneCamera: string;
  t: number;
  hud: boolean;
  highlight?: { bbox: BBox; type: string };
}

const r2 = (v: number) => Math.round(v * 100) / 100;
const px = (v: number) => Math.round(v * 10) / 10;

function scaleBox(b: BBox, k: number): BBox {
  return [px(b[0] * k), px(b[1] * k), px(b[2] * k), px(b[3] * k)];
}

function jitter(rng: Rng, b: BBox, amount: number): BBox {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  const d = () => between(rng, -amount, amount);
  return [px(b[0] + w * d()), px(b[1] + h * d()), px(b[2] + w * d()), px(b[3] + h * d())];
}

/** Голова человека по рамке (как в drawPerson): для рамок каски и сигареты */
function headBox(person: BBox, wide = 1.5): BBox {
  const h = (person[3] - person[1]) / 1.02;
  const cx = (person[0] + person[2]) / 2;
  const cy = person[1] + h - h * 0.9;
  const r = h * 0.075;
  return [px(cx - r * wide), px(cy - r * 1.6), px(cx + r * wide), px(cy + r * 1.1)];
}

function vestBox(person: BBox): BBox {
  const h = (person[3] - person[1]) / 1.02;
  const cx = (person[0] + person[2]) / 2;
  const foot = person[1] + h;
  return [px(cx - h * 0.13), px(foot - h * 0.82), px(cx + h * 0.13), px(foot - h * 0.45)];
}

function versionFor(ts: number): string {
  const d = new Date(ts * 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `fw-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

function isoAt(ts: number): string {
  return new Date(ts * 1000).toISOString();
}

export function createFlywheelMock(deps: FlywheelMockDeps): FlywheelMock {
  const rng = mulberry32(20261024);
  const startedAt = nowSec();
  const forkliftDeployedAt = startedAt - 5.5 * 3600;
  const samples: FlywheelSample[] = [];
  const render = new Map<string, RenderInfo>();
  const frames = new Map<string, string>();
  let dropped = 3;
  let liveTimer: number | null = null;

  /** uuid4 hex, как в flywheel.db */
  const nextId = (): string => {
    let hex = '';
    for (let i = 0; i < 4; i += 1) hex += Math.floor(rng() * 0xffffffff).toString(16).padStart(8, '0');
    return hex;
  };

  // ------------------------------------------------------------ детекции и разметка

  function modelDetections(objects: MockObject[], k: number, t: number): SampleDetection[] {
    const out: SampleDetection[] = [];
    for (const o of objects) {
      const box = scaleBox(o.bbox, k);
      if (o.cls === 'forklift') {
        // до выкатки forklift-v1 COCO-детектор видел погрузчик как грузовик, и то не всегда
        if (t >= forkliftDeployedAt) out.push({ model_id: 'forklift', cls: 'forklift', conf: r2(between(rng, 0.71, 0.93)), bbox: jitter(rng, box, 0.02), track_id: o.track });
        else if (rng() < 0.6) out.push({ model_id: 'det', cls: 'truck', conf: r2(between(rng, 0.27, 0.46)), bbox: jitter(rng, box, 0.05), track_id: o.track });
        continue;
      }
      out.push({ model_id: 'det', cls: 'person', conf: r2(between(rng, 0.58, 0.94)), bbox: jitter(rng, box, 0.015), track_id: o.track });
      if (o.lying) continue;
      if (rng() < 0.85) {
        out.push({ model_id: 'ppe', cls: o.helmet ? 'helmet' : 'no_helmet', conf: r2(between(rng, 0.42, 0.88)), bbox: jitter(rng, headBox(box), 0.04), track_id: null });
      }
      if (o.vest && rng() < 0.5) {
        out.push({ model_id: 'ppe', cls: 'vest', conf: r2(between(rng, 0.45, 0.86)), bbox: jitter(rng, vestBox(box), 0.04), track_id: null });
      }
    }
    return out;
  }

  /** Учитель (YOLOv8x / YOLO-World) размечает кадр; иногда ошибается — есть что поправить */
  function teacherLabels(objects: MockObject[], k: number, extra: SampleLabel[] = []): SampleLabel[] {
    const out: SampleLabel[] = [];
    for (const o of objects) {
      const box = scaleBox(o.bbox, k);
      if (o.cls === 'forklift') {
        out.push({ cls: 'forklift', bbox: jitter(rng, box, 0.01), origin: 'teacher', conf: r2(between(rng, 0.74, 0.95)) });
        continue;
      }
      out.push({ cls: 'person', bbox: jitter(rng, box, 0.008), origin: 'teacher', conf: r2(between(rng, 0.81, 0.97)) });
      if (o.lying) continue;
      // учитель иногда путает каску — есть что поправить; на нарушителе события не ошибается
      const wrong = !o.violator && rng() < 0.12;
      const helmetCls = o.helmet ? 'helmet' : 'no_helmet';
      out.push({ cls: wrong ? (o.helmet ? 'no_helmet' : 'helmet') : helmetCls, bbox: jitter(rng, headBox(box), 0.02), origin: 'teacher', conf: r2(between(rng, 0.52, 0.9)) });
      if (o.vest) out.push({ cls: 'vest', bbox: jitter(rng, vestBox(box), 0.02), origin: 'teacher', conf: r2(between(rng, 0.6, 0.9)) });
    }
    return [...out, ...extra];
  }

  function operatorEdits(labels: SampleLabel[]): SampleLabel[] {
    // часть принятых меток человек поправил: класс или лишняя рамка
    return labels
      .filter(() => rng() > 0.05)
      .map((l) => (rng() < 0.12 ? { ...l, origin: 'operator', conf: null } : l));
  }

  // ------------------------------------------------------------ генерация сэмпла

  interface Draft {
    cameraId: string;
    t: number;
    source: SampleSource;
    status: SampleStatus;
    event?: GuardEvent;
    openSet?: (typeof OPEN_SETS)[keyof typeof OPEN_SETS];
  }

  function buildSample(d: Draft): FlywheelSample {
    const holdout = d.cameraId === HOLDOUT_CAMERA;
    const full = FULL_SIZE.has(d.cameraId);
    const width = full ? 1280 : 960;
    const height = full ? 720 : 540;
    const k = width / FRAME_W;
    let objects: MockObject[] = sceneObjects(d.cameraId, d.t);
    let detections: SampleDetection[] = [];
    let extra: SampleLabel[] = [];
    let reason = '';
    let operatorLabel: OperatorLabel | null = null;
    let highlight: RenderInfo['highlight'];

    switch (d.source) {
      case 'event': {
        const ev = d.event;
        if (ev) {
          const violator: MockObject = {
            cls: 'person',
            bbox: ev.bbox,
            track: ev.track_id ?? 0,
            helmet: ev.type !== 'ppe_missing',
            vest: ev.type !== 'ppe_missing',
            lying: ev.type === 'fall_immobile',
            violator: true,
          };
          // нарушителя сцена рисует поверх обычных людей и техники
          objects = [...objects, violator];
          highlight = { bbox: ev.bbox, type: ev.type };
          reason = `событие «${getViolationMeta(ev.type).title}», трек #${ev.track_id ?? '—'}`;
          operatorLabel = OPERATOR_LABEL_BY_STATUS[ev.status] ?? null;
          if (ev.type === 'smoking_fire') extra = [{ cls: 'cigarette', bbox: headBox(scaleBox(ev.bbox, k), 2.6), origin: 'teacher', conf: 0.41 }];
        }
        detections = modelDetections(objects, k, d.t);
        break;
      }
      case 'uncertain': {
        detections = modelDetections(objects, k, d.t);
        const target = detections.find((x) => x.cls === 'no_helmet' || x.cls === 'helmet') ?? detections.find((x) => x.cls === 'person');
        if (target) {
          const thr = target.cls === 'person' ? 0.4 : 0.35;
          target.conf = r2(between(rng, thr - 0.1, thr + 0.05));
          reason = `${target.cls} ${target.conf.toFixed(2)} ∈ [${(thr - 0.1).toFixed(2)}, ${(thr + 0.05).toFixed(2)}]`;
        } else {
          reason = 'person 0.33 ∈ [0.30, 0.45]';
        }
        break;
      }
      case 'flicker': {
        detections = modelDetections(objects, k, d.t);
        const a = intBetween(rng, 12, 90);
        reason = `трек #${a} пропал, через ${between(rng, 0.2, 0.9).toFixed(1)} с рядом родился #${a + intBetween(rng, 1, 6)}`;
        break;
      }
      case 'disagreement': {
        detections = modelDetections(objects, k, d.t);
        const target = pick(rng, detections.length ? detections : [{ cls: 'person', conf: 0.52 } as SampleDetection]);
        reason = rng() < 0.6 ? `TPU: ${target.cls} ${target.conf.toFixed(2)} без пары на CPU (IoU < 0.5)` : `CPU fp32: ${target.cls} ${target.conf.toFixed(2)}, на TPU рамки нет`;
        break;
      }
      case 'operator':
        detections = modelDetections(objects, k, d.t);
        reason = 'оператор отправил кадр вручную';
        break;
      case 'import':
        reason = `импорт: ${d.openSet?.name ?? 'открытый набор'}, train`;
        break;
    }

    let labels: SampleLabel[] = [];
    if (d.source === 'import') {
      labels = teacherLabels(objects, k).map((l) => ({ ...l, origin: 'dataset', conf: null }));
    } else if (d.status !== 'new') {
      labels = teacherLabels(objects, k, extra);
      if (d.status === 'accepted') labels = operatorEdits(labels);
    }

    const id = nextId();
    const reviewed = d.source !== 'import' && (d.status === 'accepted' || d.status === 'rejected');
    render.set(id, { sceneCamera: d.cameraId, t: d.t, hud: d.source !== 'import', highlight });
    return {
      id,
      created_at: d.t,
      camera_id: d.cameraId,
      source: d.source,
      event_id: d.event?.id ?? null,
      frame_path: `frames/${d.cameraId}/${id}.jpg`,
      width,
      height,
      detections,
      labels,
      status: d.status,
      operator_label: operatorLabel,
      reason,
      holdout: holdout ? 1 : 0,
      reviewed_at: reviewed ? Math.min(nowSec(), d.t + between(rng, 120, 5400)) : null,
      dataset_version: null,
    };
  }

  function statusFor(age: number): SampleStatus {
    const r = rng();
    if (age < 3 * 3600) return r < 0.32 ? 'new' : r < 0.86 ? 'auto' : r < 0.96 ? 'accepted' : 'rejected';
    if (age < DAY) return r < 0.08 ? 'new' : r < 0.52 ? 'auto' : r < 0.88 ? 'accepted' : 'rejected';
    return r < 0.03 ? 'new' : r < 0.2 ? 'auto' : r < 0.86 ? 'accepted' : 'rejected';
  }

  // ------------------------------------------------------------ история

  function seed(): void {
    const cams = deps.cameras().filter((c) => c.enabled).map((c) => c.camera_id);
    const camPool = [...cams, HOLDOUT_CAMERA];
    // источник event — каждое событие без лимита (контракт), за последние трое суток
    for (const ev of deps.events()) {
      if (startedAt - ev.confirmed_at > 3 * DAY) continue;
      samples.push(buildSample({ cameraId: ev.camera_id, t: ev.confirmed_at, source: 'event', status: statusFor(startedAt - ev.confirmed_at), event: ev }));
    }
    const others = (Object.keys(SOURCE_WEIGHT) as (keyof typeof SOURCE_WEIGHT)[]).filter((s) => s !== 'event');
    while (samples.length < HISTORY_SAMPLES) {
      // плотнее к текущему моменту: треть выборки — последние 6 часов
      const age = rng() < 0.36 ? between(rng, 60, 6 * 3600) : between(rng, 6 * 3600, 3 * DAY);
      const source = weighted(rng, others, (s) => SOURCE_WEIGHT[s]);
      const cameraId = weighted(rng, camPool, (c) => CAMERA_WEIGHT[c] ?? 1);
      samples.push(buildSample({ cameraId, t: startedAt - age, source, status: statusFor(age) }));
    }
    // импорт открытых наборов: готовая разметка, статус accepted без проверки
    const importAt = startedAt - 26 * 3600;
    for (let i = 0; i < 36; i += 1) {
      const openSet = i < 18 ? OPEN_SETS.forklift : i < 28 ? OPEN_SETS.sh17 : OPEN_SETS.ppe;
      samples.push(buildSample({ cameraId: openSet.id, t: importAt + i * 7 + rng() * 1000, source: 'import', status: 'accepted', openSet }));
    }
    samples.sort((a, b) => b.created_at - a.created_at);
  }

  seed();

  // ------------------------------------------------------------ датасеты

  const datasets: DatasetVersion[] = [];

  function exportDataset(at: number, includeHoldout: boolean, classes?: string[]): DatasetVersion {
    const pool = samples.filter((s) => s.status === 'accepted' && s.created_at <= at && (includeHoldout || !isHoldout(s)));
    const flywheel = pool.filter((s) => s.source !== 'import');
    const imported = pool.filter((s) => s.source === 'import');
    const hardNegative = flywheel.filter((s) => s.operator_label === 'hard_negative').length;
    const openForklift = 1172;
    const openSh17 = 468;
    const total = flywheel.length + imported.length + openForklift + openSh17;
    const nVal = Math.round(total * 0.12);
    let version = versionFor(at);
    while (datasets.some((d) => d.version === version)) version = `${version.split('.')[0]}.${datasets.length + 1}`;
    const allClasses = classes?.length ? classes : ['person', 'forklift', 'helmet', 'no_helmet', 'vest', 'cigarette'];
    const ds: DatasetVersion = {
      version,
      created_at: at,
      path: `data/flywheel/datasets/${version}`,
      classes: allClasses,
      n_train: total - nVal,
      n_val: nVal,
      n_hard_negative: hardNegative,
      manifest: {
        split: 'по камере и времени',
        include_holdout: includeHoldout,
        sources: [
          { name: 'маховик: принятые сэмплы', images: flywheel.length },
          { name: `${OPEN_SETS.forklift.name} (train)`, images: openForklift, url: OPEN_SETS.forklift.url },
          { name: `${OPEN_SETS.sh17.name} (подвыборка train)`, images: openSh17, url: OPEN_SETS.sh17.url },
          { name: 'импорт открытых наборов через CLI', images: imported.length },
        ],
        sha256: { train: '9f2c41d07a5e83b16c4d2e8f0a7b9c31d5e6f708192a3b4c5d6e7f8091a2b3c4', val: '41b07e9a2c5d8f13e6a9b2c5d8e1f4a7b0c3d6e9f2a5b8c1d4e7f0a3b6c9d2e5' },
      },
    };
    for (const s of pool) s.dataset_version = version;
    datasets.unshift(ds);
    return ds;
  }

  exportDataset(startedAt - 20.5 * 3600, false);
  exportDataset(startedAt - 2.4 * 3600, false);

  // ------------------------------------------------------------ карточки моделей

  const models: ModelCard[] = [
    {
      model_id: 'fall',
      version: 'fall-v1',
      parent: 'yolov8n-pose (COCO)',
      created_at: isoAt(startedAt - 1.6 * 3600),
      task: 'pose',
      classes: ['person'],
      train: {
        datasets: [
          { name: 'UR Fall Detection', images: 2870, url: 'http://fenix.ur.edu.pl/~mkepski/ds/uf.html' },
          { name: 'Le2i Fall', images: 3410, url: 'https://imvia.u-bourgogne.fr/en/database/fall-detection-dataset-2.html' },
          { name: 'CAUCAFall', images: 1925, url: 'https://data.mendeley.com/datasets/7w7fccy7ky/4' },
          { name: datasets[0].version, images: 214 },
        ],
        epochs: 40,
        imgsz: 480,
        device: 'mps',
        minutes: 52,
      },
      weights: 'models/weights/fall_v1.pt',
      onnx: 'fall_v1-480.onnx',
      tpu_program: 'tpu/fall_v1_480/fall_v1_480_b1_o5_h1m.tpu',
      holdout: {
        dataset_test: { name: 'UR Fall + CAUCAFall test (другие люди)', images: 640, map50: 0.88, per_class: { person: 0.88 } },
        events: { videos: ['s3_01', 's3_02'], tp: 7, fp: 0, fn: 0, duplicates: 0, latency_p50_s: 3.2, fp_per_hour: 0 },
      },
      baseline: {
        version: 'pose (yolov8n-pose COCO)',
        holdout: { dataset_test: { map50: 0.81 }, events: { tp: 6, fp: 0, fn: 1, duplicates: 0, latency_p50_s: 3.5, fp_per_hour: 0 } },
      },
      tpu_check: { frames: 40, recall: 0.97, precision: 0.98 },
      gate: { promote: true, reasons: ['S3: 7/7 против 6/7 — падение с кровати теперь находится', 'Ложных тревог 0 против 0', 'mAP50 0.88 против 0.81 на тесте наборов'] },
      status: 'candidate',
    },
    {
      model_id: 'ppe',
      version: 'ppe-v2',
      parent: 'ppe-v0.3 (constructelligence)',
      created_at: isoAt(startedAt - 4.2 * 3600),
      task: 'detect',
      classes: ['person', 'helmet', 'no_helmet', 'vest', 'no_vest'],
      train: {
        datasets: [
          { name: 'SH17', images: 6480, url: OPEN_SETS.sh17.url },
          { name: 'Construction-PPE (Ultralytics)', images: 1132, url: OPEN_SETS.ppe.url },
          { name: datasets[1].version, images: 168 },
        ],
        epochs: 30,
        imgsz: 640,
        device: 'mps',
        minutes: 71,
      },
      weights: 'models/weights/ppe_v2.pt',
      onnx: 'ppe_v2-640.onnx',
      tpu_program: null,
      holdout: {
        dataset_test: { name: 'Construction-PPE test', images: 284, map50: 0.79, per_class: { helmet: 0.83, no_helmet: 0.41, vest: 0.81 } },
        events: { videos: ['s2_01', 's2_02'], tp: 5, fp: 2, fn: 1, duplicates: 0, latency_p50_s: 2.6, fp_per_hour: 13.6 },
      },
      baseline: {
        version: 'ppe-v0.3',
        holdout: { dataset_test: { map50: 0.82 }, events: { tp: 5, fp: 0, fn: 1, duplicates: 0, latency_p50_s: 2.4, fp_per_hour: 0 } },
      },
      tpu_check: null,
      gate: {
        promote: false,
        reasons: ['mAP50 0.79 против 0.82: падение 0.03 больше допуска 0.01', 'Precision по событиям 0.71 против 1.00: 2 ложные тревоги на s2_02 (спецовка без жилета)', 'Recall 5/6 — как у базы'],
      },
      status: 'rejected',
    },
    {
      model_id: 'forklift',
      version: 'forklift-v1',
      parent: 'yolov8n (COCO)',
      created_at: isoAt(startedAt - 7.3 * 3600),
      task: 'detect',
      classes: ['person', 'forklift'],
      train: {
        datasets: [{ name: 'Roboflow forklift', images: 1172, url: OPEN_SETS.forklift.url }],
        epochs: 30,
        imgsz: 480,
        device: 'mps',
        minutes: 25,
      },
      weights: 'models/weights/forklift_v1.pt',
      onnx: 'forklift_v1-480.onnx',
      tpu_program: 'tpu/forklift_v1_480/forklift_v1_480_b1_o5_h1m.tpu',
      holdout: {
        dataset_test: { name: 'forklift-rf test', images: 120, map50: 0.91, per_class: { forklift: 0.93, person: 0.89 } },
        events: { videos: ['s4_01', 's4_02', 's4_03'], tp: 4, fp: 0, fn: 0, duplicates: 0, latency_p50_s: 2.1, fp_per_hour: 0 },
      },
      baseline: { version: 'det (yolov8n COCO)', holdout: { events: { tp: 2, fp: 0, fn: 2, duplicates: 0, latency_p50_s: 2.3, fp_per_hour: 0 } } },
      tpu_check: { frames: 40, recall: 0.95, precision: 0.97 },
      gate: { promote: true, reasons: ['S4: 4/4 против 2/4', 'Ложных тревог 0 против 0', 'TPU против CPU: recall 0.95, precision 0.97'] },
      status: 'deployed',
    },
  ];

  // ------------------------------------------------------------ задачи

  const jobs: FlywheelJob[] = [
    {
      id: 'job-0003',
      kind: 'gate',
      status: 'done',
      args: { version: 'fall-v1' },
      created_at: startedAt - 1.4 * 3600,
      finished_at: startedAt - 1.4 * 3600 + 214,
      log_tail: ['eval s3_01, s3_02 (onnx-cpu): tp 7, fp 0, fn 0', 'база pose: tp 6, fp 0, fn 1', 'mAP50 0.88 против 0.81', 'решение: promote=true → ml/model_cards/fall-v1.json'],
    },
    {
      id: 'job-0002',
      kind: 'export',
      status: 'done',
      args: { include_holdout: false },
      created_at: startedAt - 2.4 * 3600,
      finished_at: startedAt - 2.4 * 3600 + 38,
      log_tail: [`версия ${datasets[0].version}: train ${datasets[0].n_train}, val ${datasets[0].n_val}`, 'отложенные ролики пропущены: 23 сэмпла'],
    },
    {
      id: 'job-0001',
      kind: 'autolabel',
      status: 'done',
      args: { teacher: 'yolov8x+yolo-world' },
      created_at: startedAt - 2.9 * 3600,
      finished_at: startedAt - 2.9 * 3600 + 412,
      log_tail: ['учитель: yolov8x (COCO) + YOLO-World (forklift, cigarette, hard hat)', 'размечено 186 сэмплов, 1 124 рамки', 'статус new → auto'],
    },
  ];
  let jobSeq = 3;

  function runJob(job: FlywheelJob): void {
    const log = (line: string) => {
      const tail = Array.isArray(job.log_tail) ? job.log_tail : [];
      job.log_tail = [...tail, line].slice(-12);
    };
    const steps: Record<string, string[]> = {
      autolabel: ['загрузка учителя yolov8x + YOLO-World', 'разметка новых сэмплов…', 'рамки учителя записаны, статус new → auto'],
      export: ['отбор принятых сэмплов', 'разбиение по камере и времени', 'манифест и sha256 списков файлов'],
      gate: ['прогон отложенных роликов (onnx-cpu)', 'сравнение с базовой версией', 'решение гейта записано в карточку'],
    };
    const lines = steps[job.kind] ?? ['выполняется'];
    // таймеры задачи не зависят от подписки WS: задача доходит до конца
    lines.forEach((line, i) => window.setTimeout(() => log(line), 700 + i * 1300));
    window.setTimeout(() => {
        if (job.kind === 'autolabel') {
          let n = 0;
          for (const s of samples) {
            if (s.status !== 'new') continue;
            const info = render.get(s.id);
            const k = s.width / FRAME_W;
            s.labels = teacherLabels(info ? sceneObjects(info.sceneCamera, info.t) : [], k);
            s.status = 'auto';
            n += 1;
          }
          log(`размечено ${n} сэмплов`);
        }
        if (job.kind === 'export') log(`версия ${exportDataset(nowSec(), false).version}`);
        job.status = 'done';
        job.finished_at = nowSec();
      }, 900 + lines.length * 1300);
  }

  // ------------------------------------------------------------ живой сбор

  function publish(sample: FlywheelSample): void {
    samples.unshift(sample);
    const brief: FlywheelSampleBrief = {
      id: sample.id,
      created_at: sample.created_at,
      camera_id: sample.camera_id,
      source: sample.source,
      status: sample.status,
      event_id: sample.event_id,
      reason: sample.reason,
      holdout: sample.holdout,
    };
    deps.emit({ kind: 'flywheel_sample', sample: brief });
  }

  function spawnLive(): void {
    const cams = deps.cameras().filter((c) => c.enabled).map((c) => c.camera_id);
    if (cams.length === 0) return;
    const cameraId = weighted(rng, cams, (c) => CAMERA_WEIGHT[c] ?? 1);
    const source = weighted(rng, ['uncertain', 'flicker', 'disagreement'] as const, (s) => SOURCE_WEIGHT[s]);
    // иногда очередь записи переполнена — сэмпл отброшен (счётчик «отброшено»)
    if (rng() < 0.04) {
      dropped += 1;
      return;
    }
    publish(buildSample({ cameraId, t: nowSec(), source, status: 'new' }));
  }

  function scheduleLive(delay: number): void {
    liveTimer = window.setTimeout(() => {
      spawnLive();
      scheduleLive(between(rng, 6000, 12000));
    }, delay);
  }

  // ------------------------------------------------------------ сводка и выборка

  function summary(): FlywheelSummary {
    const today = startOfDay(nowSec());
    const bySource: Record<string, number> = {};
    const byStatus: Record<string, number> = {};
    const byClass: Record<string, number> = {};
    let reviewed = 0;
    let holdout = 0;
    let todayCount = 0;
    for (const s of samples) {
      bySource[s.source] = (bySource[s.source] ?? 0) + 1;
      byStatus[s.status] = (byStatus[s.status] ?? 0) + 1;
      for (const cls of new Set((s.labels.length ? s.labels : s.detections).map((x) => x.cls))) byClass[cls] = (byClass[cls] ?? 0) + 1;
      if (s.reviewed_at != null) reviewed += 1;
      if (isHoldout(s)) holdout += 1;
      if (s.created_at >= today) todayCount += 1;
    }
    return {
      total: samples.length,
      today: todayCount,
      by_source: bySource,
      by_status: byStatus,
      by_class: byClass,
      reviewed,
      holdout,
      dropped,
      last_dataset: datasets[0] ?? null,
      models: {
        deployed: models.filter((m) => m.status === 'deployed').map((m) => m.version),
        candidate: models.filter((m) => m.status === 'candidate').map((m) => m.version),
      },
      collecting: true,
    };
  }

  function matches(s: FlywheelSample, q: SampleQuery): boolean {
    if (q.status && !q.status.split(',').includes(s.status)) return false;
    if (q.source && !q.source.split(',').includes(s.source)) return false;
    if (q.camera_id && s.camera_id !== q.camera_id) return false;
    if (q.event_id && s.event_id !== q.event_id) return false;
    if (q.holdout != null && (isHoldout(s) ? 1 : 0) !== q.holdout) return false;
    if (q.cls && !(s.labels.length ? s.labels : s.detections).some((x) => x.cls === q.cls)) return false;
    return true;
  }

  function frameUrl(sample: FlywheelSample, size: 'thumb' | 'full' = 'full'): string {
    const key = `${sample.id}:${size}`;
    const cached = frames.get(key);
    if (cached) return cached;
    const info = render.get(sample.id) ?? { sceneCamera: sample.camera_id, t: sample.created_at, hud: true };
    const camera = { camera_id: info.sceneCamera, name: info.sceneCamera, zones: [], rules: [] } as unknown as CameraConfig;
    const [w, h] = size === 'thumb' ? [320, Math.round((320 * sample.height) / sample.width)] : [sample.width, sample.height];
    const url = renderToDataUrl(
      {
        camera,
        t: info.t,
        overlays: false,
        hud: info.hud,
        highlight: info.highlight ? { bbox: info.highlight.bbox, type: info.highlight.type, label: '', frame: false } : null,
      },
      w,
      h,
    );
    if (frames.size > 360) frames.clear();
    frames.set(key, url);
    return url;
  }

  const findSample = (id: string) => samples.find((s) => s.id === id);

  const api: FlywheelApi = {
    summary: () => deps.respond(summary(), 80, 180),
    listSamples: (q) => {
      const filtered = samples.filter((s) => matches(s, q));
      const order = q.order ?? '-created_at';
      const sorted = order === 'created_at' ? [...filtered].reverse() : filtered;
      const offset = q.offset ?? 0;
      const limit = q.limit ?? 50;
      return deps.respond({ items: sorted.slice(offset, offset + limit), total: filtered.length });
    },
    getSample: (id) => {
      const s = findSample(id);
      return s ? deps.respond(s, 50, 120) : deps.fail(404, 'Сэмпл не найден');
    },
    patchSample: (id, patch) => {
      const s = findSample(id);
      if (!s) return deps.fail(404, 'Сэмпл не найден');
      if (patch.labels) {
        // изменённые метки получают origin = operator
        const before = new Map(s.labels.map((l) => [JSON.stringify([l.cls, l.bbox]), l]));
        s.labels = patch.labels.map((l) => {
          const same = before.get(JSON.stringify([l.cls, l.bbox]));
          return same ?? { ...l, origin: 'operator', conf: null };
        });
      }
      if (patch.status) s.status = patch.status;
      s.reviewed_at = nowSec();
      return deps.respond(s, 90, 200);
    },
    capture: (cameraId) => {
      const camera = deps.cameras().find((c) => c.camera_id === cameraId);
      if (!camera) return deps.fail(404, `Камера ${cameraId} не найдена`);
      if (!camera.enabled) return deps.fail(409, `Камера ${cameraId} выключена`);
      const sample = buildSample({ cameraId, t: nowSec(), source: 'operator', status: 'new' });
      publish(sample);
      return deps.respond(sample, 200, 400);
    },
    buildDataset: (request) => deps.respond(exportDataset(nowSec(), Boolean(request.include_holdout), request.classes), 900, 1500),
    listDatasets: () => deps.respond(datasets),
    listModels: () =>
      deps.respond([...models].sort((a, b) => String(b.created_at ?? '').localeCompare(String(a.created_at ?? '')))),
    startJob: (kind: JobKind, args = {}) => {
      if (kind === 'train' || kind === 'deploy') {
        return deps.fail(403, 'Задачи train и deploy выключены: запустите шлюз с GUARD_FLYWHEEL_ALLOW_TRAIN=1');
      }
      jobSeq += 1;
      const job: FlywheelJob = {
        id: `job-${String(jobSeq).padStart(4, '0')}`,
        kind,
        status: 'running',
        args,
        created_at: nowSec(),
        log_url: null,
        log_tail: ['задача запущена'],
      };
      jobs.unshift(job);
      runJob(job);
      return deps.respond(job, 150, 300);
    },
    listJobs: () => deps.respond(jobs, 60, 140),
    getJob: (id) => {
      const job = jobs.find((j) => j.id === id);
      return job ? deps.respond(job) : deps.fail(404, 'Задача не найдена');
    },
    frameUrl,
  };

  return {
    api,
    start() {
      scheduleLive(4000);
    },
    stop() {
      if (liveTimer != null) window.clearTimeout(liveTimer);
      liveTimer = null;
    },
    onEvent(event) {
      publish(buildSample({ cameraId: event.camera_id, t: event.confirmed_at, source: 'event', status: 'new', event }));
    },
    onEventPatch(event) {
      const label = OPERATOR_LABEL_BY_STATUS[event.status] ?? null;
      for (const s of samples) if (s.event_id === event.id) s.operator_label = label;
    },
  };
}
