import { formatDateTime } from '../../domain/format';
import { SEVERITY_LABELS, STATUS_LABELS } from '../../domain/labels';
import type { CameraConfig, GuardEvent } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';

// CSV для Excel в русской локали: разделитель «;», BOM для кириллицы,
// защита от формул (ячейки, начинающиеся с = + - @, экранируются апострофом).

function cell(value: unknown): string {
  let s = value == null ? '' : String(value);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
  if (/[";\n\r]/.test(s)) s = `"${s.replace(/"/g, '""')}"`;
  return s;
}

const BOM = String.fromCharCode(0xfeff);

const HEADER = [
  'ID события',
  'Начало нарушения',
  'Оповещение',
  'Окончание',
  'Тип (код)',
  'Тип нарушения',
  'Важность',
  'Камера (ID)',
  'Камера',
  'Участок',
  'Зона',
  'Трек',
  'Статус',
  'Комментарий оператора',
  'Окно подтверждения, с',
  'Обработка, мс',
  'Снапшот',
  'Клип',
];

export function eventsToCsv(events: GuardEvent[], cameras: Map<string, CameraConfig>, origin: string): string {
  const rows = events.map((e) => {
    const camera = cameras.get(e.camera_id);
    const zone = camera?.zones.find((z) => z.zone_id === e.zone_id);
    return [
      e.id,
      formatDateTime(e.started_at),
      formatDateTime(e.confirmed_at),
      e.ended_at ? formatDateTime(e.ended_at) : '',
      e.type,
      getViolationMeta(e.type).title,
      SEVERITY_LABELS[e.severity] ?? e.severity,
      e.camera_id,
      camera?.name ?? '',
      e.site,
      zone?.name ?? e.zone_id ?? '',
      e.track_id ?? '',
      STATUS_LABELS[e.status] ?? e.status,
      e.operator_note ?? '',
      (e.confirmed_at - e.started_at).toFixed(1).replace('.', ','),
      e.latency_ms?.total != null ? String(Math.round(e.latency_ms.total)) : '',
      `${origin}/api/events/${encodeURIComponent(e.id)}/snapshot.jpg`,
      `${origin}/api/events/${encodeURIComponent(e.id)}/clip.mp4`,
    ]
      .map(cell)
      .join(';');
  });
  return BOM + [HEADER.map(cell).join(';'), ...rows].join('\r\n');
}

export function downloadText(filename: string, text: string, mime = 'text/csv;charset=utf-8'): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
