import { formatNumber } from './format';

// Человекочитаемые подписи для Event.details (ключи зависят от плагина правил)

const PPE_NAMES: Record<string, string> = {
  helmet: 'каска',
  vest: 'сигнальный жилет',
  respirator: 'респиратор',
  harness: 'страховочная привязь',
};

const VEHICLE_NAMES: Record<string, string> = {
  forklift: 'погрузчик',
  truck: 'грузовик',
  car: 'автомобиль',
  bus: 'автобус',
};

/** Служебные ключи: уже показаны в карточке или нужны только конвейеру */
const HIDDEN = new Set(['keypoints', 'zone_name', 'zone_kind', 'camera_name', 'foot_xy', 'confirm_window_s', 'bbox']);

export function describeDetails(details: Record<string, unknown>): { label: string; value: string }[] {
  const out: { label: string; value: string }[] = [];
  for (const [key, raw] of Object.entries(details ?? {})) {
    if (raw == null || HIDDEN.has(key)) continue;
    switch (key) {
      case 'missing':
        out.push({
          label: 'Отсутствует',
          value: (Array.isArray(raw) ? raw : [raw]).map((v) => PPE_NAMES[String(v)] ?? String(v)).join(', '),
        });
        break;
      case 'vehicle':
        out.push({ label: 'Техника', value: VEHICLE_NAMES[String(raw)] ?? String(raw) });
        break;
      case 'distance_bh':
        out.push({ label: 'Дистанция', value: `${formatNumber(Number(raw), 1)} роста` });
        break;
      case 'distance_m':
        out.push({ label: 'Дистанция', value: `${formatNumber(Number(raw), 1)} м` });
        break;
      case 'immobile_s':
        out.push({ label: 'Без движения', value: `${formatNumber(Number(raw), 0)} с` });
        break;
      case 'speed_bh':
        out.push({ label: 'Скорость', value: `${formatNumber(Number(raw), 1)} роста/с` });
        break;
      case 'pose_state':
        out.push({ label: 'Поза', value: raw === 'lying' ? 'лежит' : raw === 'standing' ? 'стоит' : String(raw) });
        break;
      case 'source':
        out.push({ label: 'Источник', value: raw === 'smoking' ? 'курение' : raw === 'fire' ? 'открытое пламя' : String(raw) });
        break;
      default: {
        const label = key.replace(/_/g, ' ');
        const value =
          typeof raw === 'number' ? formatNumber(raw) : typeof raw === 'string' || typeof raw === 'boolean' ? String(raw) : JSON.stringify(raw);
        out.push({ label, value });
      }
    }
  }
  return out;
}
