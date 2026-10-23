import { Clock, ImageOff, MousePointer2, Pentagon, Plus, RefreshCw, Trash2, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../api';
import { ZONE_KIND_COLORS, ZONE_KIND_LABELS } from '../../domain/labels';
import type { CameraConfig, Point, Zone, ZoneKind } from '../../domain/types';
import { describeHours, isValidHoursRange, isZoneActive, makeId, polygonCentroid, roundPoint } from '../../domain/zones';
import { getViolationMeta } from '../../domain/violations';
import { useNow } from '../../hooks/useNow';

interface Props {
  camera: CameraConfig;
  onChange: (zones: Zone[], removedZoneId?: string) => void;
}

const CLOSE_RADIUS_PX = 12;

function pointsAttr(points: Point[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

export function ZoneEditor({ camera, onChange }: Props) {
  const now = useNow();
  const zones = camera.zones;
  const [selectedId, setSelectedId] = useState<string | null>(zones[0]?.zone_id ?? null);
  const [drawing, setDrawing] = useState<Point[] | null>(null);
  const [hoverPt, setHoverPt] = useState<Point | null>(null);
  const [frameNonce, setFrameNonce] = useState(0);
  const [aspect, setAspect] = useState(16 / 9);
  const [frameError, setFrameError] = useState(false);
  const stageRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ zoneId: string; index: number } | null>(null);

  const frameUrl = useMemo(
    () => api().media.frameUrl(camera),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [camera.camera_id, frameNonce],
  );

  useEffect(() => {
    setSelectedId((id) => (id && zones.some((z) => z.zone_id === id) ? id : zones[0]?.zone_id ?? null));
  }, [zones]);

  useEffect(() => {
    setDrawing(null);
    setFrameError(false);
  }, [camera.camera_id]);

  const selected = zones.find((z) => z.zone_id === selectedId) ?? null;

  const toNorm = (clientX: number, clientY: number): Point => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return [0, 0];
    return roundPoint([(clientX - rect.left) / rect.width, (clientY - rect.top) / rect.height]);
  };

  const nearFirst = (pt: Point): boolean => {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect || !drawing || drawing.length < 3) return false;
    const [fx, fy] = drawing[0];
    return Math.hypot((pt[0] - fx) * rect.width, (pt[1] - fy) * rect.height) < CLOSE_RADIUS_PX;
  };

  const updateZone = (id: string, patch: Partial<Zone>) => {
    onChange(zones.map((z) => (z.zone_id === id ? { ...z, ...patch } : z)));
  };

  const finishDrawing = (points: Point[]) => {
    if (points.length < 3) return;
    const zone: Zone = {
      zone_id: makeId('z'),
      name: `Зона ${zones.length + 1}`,
      kind: 'danger',
      polygon: points,
      active_hours: null,
    };
    onChange([...zones, zone]);
    setSelectedId(zone.zone_id);
    setDrawing(null);
  };

  // Клавиатура в режиме рисования: Enter — замкнуть, Backspace — убрать точку, Esc — отмена
  useEffect(() => {
    if (!drawing) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDrawing(null);
      else if (e.key === 'Enter') finishDrawing(drawing);
      else if (e.key === 'Backspace') setDrawing((d) => (d ? d.slice(0, -1) : d));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const onStageClick = (e: React.MouseEvent) => {
    if (!drawing) return;
    const pt = toNorm(e.clientX, e.clientY);
    if (nearFirst(pt)) finishDrawing(drawing);
    else setDrawing([...drawing, pt]);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const pt = toNorm(e.clientX, e.clientY);
    if (drawing) setHoverPt(pt);
    const drag = dragRef.current;
    if (drag) {
      const zone = zones.find((z) => z.zone_id === drag.zoneId);
      if (!zone) return;
      const polygon = zone.polygon.map((p, i) => (i === drag.index ? pt : p));
      updateZone(zone.zone_id, { polygon });
    }
  };

  const startDrag = (e: React.PointerEvent, zoneId: string, index: number) => {
    e.stopPropagation();
    if (e.altKey || e.button === 2) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = { zoneId, index };
  };

  const removeVertex = (zone: Zone, index: number) => {
    if (zone.polygon.length <= 3) return;
    updateZone(zone.zone_id, { polygon: zone.polygon.filter((_, i) => i !== index) });
  };

  const insertVertex = (zone: Zone, index: number) => {
    const a = zone.polygon[index];
    const b = zone.polygon[(index + 1) % zone.polygon.length];
    const mid = roundPoint([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]);
    const polygon = [...zone.polygon.slice(0, index + 1), mid, ...zone.polygon.slice(index + 1)];
    updateZone(zone.zone_id, { polygon });
  };

  const deleteZone = (id: string) => {
    onChange(
      zones.filter((z) => z.zone_id !== id),
      id,
    );
  };

  const rulesUsing = (zoneId: string) => camera.rules.filter((r) => r.zones?.includes(zoneId));

  const hoursError = selected?.active_hours?.some((h) => !isValidHoursRange(h));

  return (
    <div className="zone-editor">
      <div className="zone-editor__canvas">
        <div className="zone-toolbar">
          {drawing ? (
            <>
              <span className="zone-toolbar__hint">
                <Pentagon size={15} /> Кликайте по кадру, чтобы ставить вершины. Замкнуть — клик по первой точке или <span className="kbd">Enter</span>,
                отмена — <span className="kbd">Esc</span>
              </span>
              <button type="button" className="btn btn--sm" disabled={drawing.length < 3} onClick={() => finishDrawing(drawing)}>
                Замкнуть ({drawing.length})
              </button>
              <button type="button" className="btn btn--sm btn--ghost" onClick={() => setDrawing(null)}>
                <X size={14} /> Отмена
              </button>
            </>
          ) : (
            <>
              <button type="button" className="btn btn--sm btn--primary" onClick={() => setDrawing([])}>
                <Plus size={15} /> Нарисовать зону
              </button>
              <span className="zone-toolbar__hint">
                <MousePointer2 size={14} /> Тяните вершины; «+» на ребре — новая вершина; Alt+клик — удалить вершину
              </span>
              <button type="button" className="btn btn--sm btn--ghost" onClick={() => setFrameNonce((n) => n + 1)} title="Обновить стоп-кадр">
                <RefreshCw size={14} /> Кадр
              </button>
            </>
          )}
        </div>

        <div
          ref={stageRef}
          className={`zone-stage${drawing ? ' is-drawing' : ''}`}
          style={{ aspectRatio: String(aspect) }}
          onClick={onStageClick}
          onPointerMove={onPointerMove}
          onPointerUp={() => (dragRef.current = null)}
          onPointerLeave={() => setHoverPt(null)}
          onContextMenu={(e) => e.preventDefault()}
        >
          {frameError ? (
            <div className="snapshot__missing">
              <ImageOff size={26} />
              Стоп-кадр недоступен — камера не передаёт кадры
            </div>
          ) : (
            <img
              src={frameUrl}
              alt={`Стоп-кадр камеры ${camera.name}`}
              draggable={false}
              onLoad={(e) => {
                const img = e.currentTarget;
                if (img.naturalWidth && img.naturalHeight) setAspect(img.naturalWidth / img.naturalHeight);
              }}
              onError={() => setFrameError(true)}
            />
          )}
          <svg className="zone-svg" viewBox="0 0 1 1" preserveAspectRatio="none">
            {zones.map((z) => {
              const color = ZONE_KIND_COLORS[z.kind];
              const isSel = z.zone_id === selectedId;
              return (
                <polygon
                  key={z.zone_id}
                  points={pointsAttr(z.polygon)}
                  fill={color}
                  fillOpacity={isSel ? 0.26 : 0.13}
                  stroke={color}
                  strokeOpacity={isSel ? 1 : 0.75}
                  strokeWidth={isSel ? 2.5 : 1.6}
                  vectorEffect="non-scaling-stroke"
                  strokeDasharray={isZoneActive(z, now) ? undefined : '6 5'}
                  onClick={(e) => {
                    if (drawing) return;
                    e.stopPropagation();
                    setSelectedId(z.zone_id);
                  }}
                  style={{ cursor: drawing ? 'crosshair' : 'pointer' }}
                />
              );
            })}
            {drawing && drawing.length > 0 && (
              <polyline
                points={pointsAttr(hoverPt ? [...drawing, hoverPt] : drawing)}
                fill="rgba(237,235,230,0.08)"
                stroke="#edebe6"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
                strokeDasharray="5 4"
              />
            )}
          </svg>
          {/* подписи зон и ручки вершин — HTML поверх, чтобы не искажались при масштабе */}
          {zones.map((z) => {
            const [cx, cy] = polygonCentroid(z.polygon);
            return (
              <span key={z.zone_id} className="zone-label" style={{ left: `${cx * 100}%`, top: `${cy * 100}%`, ['--c' as string]: ZONE_KIND_COLORS[z.kind] }}>
                {z.name}
              </span>
            );
          })}
          {selected &&
            !drawing &&
            selected.polygon.map((p, i) => {
              const next = selected.polygon[(i + 1) % selected.polygon.length];
              return (
                <span key={`v-${i}`}>
                  <button
                    type="button"
                    className="zone-handle"
                    style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%`, ['--c' as string]: ZONE_KIND_COLORS[selected.kind] }}
                    onPointerDown={(e) => startDrag(e, selected.zone_id, i)}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (e.altKey) removeVertex(selected, i);
                    }}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      removeVertex(selected, i);
                    }}
                    aria-label={`Вершина ${i + 1}: ${p[0].toFixed(3)}, ${p[1].toFixed(3)}`}
                  />
                  <button
                    type="button"
                    className="zone-mid"
                    style={{ left: `${((p[0] + next[0]) / 2) * 100}%`, top: `${((p[1] + next[1]) / 2) * 100}%` }}
                    onClick={(e) => {
                      e.stopPropagation();
                      insertVertex(selected, i);
                    }}
                    aria-label="Добавить вершину"
                  >
                    +
                  </button>
                </span>
              );
            })}
          {drawing &&
            drawing.map((p, i) => (
              <span
                key={`d-${i}`}
                className={`zone-handle zone-handle--draw${i === 0 && drawing.length >= 3 ? ' is-first' : ''}`}
                style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%` }}
              />
            ))}
        </div>
        <p className="zone-editor__legend muted">
          Координаты нормированы 0…1 относительно кадра. Пунктир — зона сейчас неактивна по расписанию.
        </p>
      </div>

      <aside className="zone-side">
        <div className="zone-list">
          <div className="zone-list__head">
            <span className="eyebrow">Зоны · {zones.length}</span>
          </div>
          {zones.length === 0 && <p className="muted zone-list__empty">Зон нет — правила работают по всему кадру.</p>}
          <ul>
            {zones.map((z) => (
              <li key={z.zone_id}>
                <button
                  type="button"
                  className={`zone-item${z.zone_id === selectedId ? ' is-selected' : ''}`}
                  onClick={() => setSelectedId(z.zone_id)}
                >
                  <i style={{ background: ZONE_KIND_COLORS[z.kind] }} />
                  <span className="zone-item__text">
                    <strong>{z.name}</strong>
                    <small>
                      {ZONE_KIND_LABELS[z.kind]} · {describeHours(z)}
                    </small>
                  </span>
                  <span className={`zone-item__state${isZoneActive(z, now) ? ' is-on' : ''}`}>
                    {isZoneActive(z, now) ? 'активна' : 'неактивна'}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {selected && (
          <div className="zone-props" key={selected.zone_id}>
            <label className="field">
              <span className="field__label">Название</span>
              <input className="input" value={selected.name} onChange={(e) => updateZone(selected.zone_id, { name: e.target.value })} />
            </label>
            <label className="field">
              <span className="field__label">Тип зоны</span>
              <select
                className="select"
                value={selected.kind}
                onChange={(e) => updateZone(selected.zone_id, { kind: e.target.value as ZoneKind })}
              >
                {(Object.keys(ZONE_KIND_LABELS) as ZoneKind[]).map((k) => (
                  <option key={k} value={k}>
                    {ZONE_KIND_LABELS[k]}
                  </option>
                ))}
              </select>
            </label>
            <div className="field">
              <span className="field__label">
                <Clock size={13} style={{ verticalAlign: '-2px' }} /> Расписание активности
              </span>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={selected.active_hours == null}
                  onChange={(e) => updateZone(selected.zone_id, { active_hours: e.target.checked ? null : ['07:55-12:00'] })}
                />
                <span className="switch__track" />
                Круглосуточно
              </label>
              {selected.active_hours != null && (
                <div className="hours">
                  {selected.active_hours.map((h, i) => (
                    <div key={i} className="hours__row">
                      <input
                        className="input input--sm mono"
                        value={h}
                        placeholder="07:55-12:00"
                        aria-invalid={!isValidHoursRange(h)}
                        onChange={(e) => {
                          const next = [...(selected.active_hours ?? [])];
                          next[i] = e.target.value;
                          updateZone(selected.zone_id, { active_hours: next });
                        }}
                      />
                      <button
                        type="button"
                        className="btn btn--sm btn--ghost btn--icon"
                        aria-label="Удалить интервал"
                        onClick={() => {
                          const next = (selected.active_hours ?? []).filter((_, j) => j !== i);
                          updateZone(selected.zone_id, { active_hours: next.length ? next : null });
                        }}
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                  <button
                    type="button"
                    className="btn btn--sm btn--ghost"
                    onClick={() => updateZone(selected.zone_id, { active_hours: [...(selected.active_hours ?? []), '13:00-17:00'] })}
                  >
                    <Plus size={14} /> Интервал
                  </button>
                  {hoursError && <span className="field__error">Формат ЧЧ:ММ-ЧЧ:ММ, например 07:55-12:00</span>}
                </div>
              )}
            </div>
            <div className="field">
              <span className="field__label">Используют правила</span>
              <div className="zone-rules">
                {rulesUsing(selected.zone_id).length === 0 ? (
                  <span className="muted">ни одно — привяжите на вкладке «Правила»</span>
                ) : (
                  rulesUsing(selected.zone_id).map((r) => (
                    <span key={r.rule_id} className="badge badge--plain">
                      {getViolationMeta(r.type).short}
                    </span>
                  ))
                )}
              </div>
            </div>
            <button type="button" className="btn btn--danger btn--sm" onClick={() => deleteZone(selected.zone_id)}>
              <Trash2 size={14} /> Удалить зону
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}
