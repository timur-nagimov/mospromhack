import { Cctv, Pentagon, Plus, RotateCcw, Save, ShieldCheck, Trash2, Video } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { Navigate, useNavigate, useParams } from 'react-router-dom';
import { useCameras, useDeleteCamera, useSaveCamera } from '../../api/queries';
import { toast } from '../../components/ui/toast';
import { formatNumber } from '../../domain/format';
import { SOURCE_TYPE_LABELS } from '../../domain/labels';
import type { CameraConfig, Zone } from '../../domain/types';
import { isValidHoursRange } from '../../domain/zones';
import { useLive } from '../../state/live';
import { RulesEditor } from './RulesEditor';
import { CAMERA_ID_RE, SourceForm } from './SourceForm';
import { ZoneEditor } from './ZoneEditor';
import './cameras.css';

type Tab = 'zones' | 'rules' | 'source';

function blankCamera(existing: CameraConfig[]): CameraConfig {
  let n = existing.length + 1;
  const ids = new Set(existing.map((c) => c.camera_id));
  while (ids.has(`cam-${String(n).padStart(2, '0')}`)) n += 1;
  return {
    camera_id: `cam-${String(n).padStart(2, '0')}`,
    name: '',
    site: '',
    source: { type: 'rtsp', uri: '', loop: false, realtime: true },
    sample_fps: 5,
    enabled: true,
    zones: [],
    rules: [],
  };
}

function validate(c: CameraConfig, isNew: boolean): string | null {
  if (isNew && !CAMERA_ID_RE.test(c.camera_id)) return 'Некорректный ID камеры';
  if (!c.name.trim()) return 'Укажите название камеры';
  if (!c.source.uri.trim()) return 'Укажите адрес источника';
  if (!(c.sample_fps > 0)) return 'Частота анализа должна быть больше 0';
  for (const z of c.zones) {
    if (!z.name.trim()) return 'У каждой зоны должно быть название';
    if (z.polygon.length < 3) return `Зона «${z.name}»: нужно минимум 3 вершины`;
    if (z.active_hours?.some((h) => !isValidHoursRange(h))) return `Зона «${z.name}»: расписание в формате ЧЧ:ММ-ЧЧ:ММ`;
  }
  for (const r of c.rules) {
    if (r.min_duration_s < 0 || r.cooldown_s < 0) return `Правило ${r.rule_id}: время не может быть отрицательным`;
  }
  return null;
}

function CameraEditor({ original, isNew, existing }: { original: CameraConfig; isNew: boolean; existing: CameraConfig[] }) {
  const navigate = useNavigate();
  const save = useSaveCamera();
  const remove = useDeleteCamera();
  const [draft, setDraft] = useState<CameraConfig>(() => structuredClone(original));
  const [tab, setTab] = useState<Tab>(isNew ? 'source' : 'zones');
  const dirty = useMemo(() => JSON.stringify(draft) !== JSON.stringify(original), [draft, original]);
  const error = validate(draft, isNew);

  useEffect(() => {
    setDraft(structuredClone(original));
  }, [original]);

  // Предупреждение при закрытии вкладки с несохранёнными изменениями
  useEffect(() => {
    if (!dirty) return;
    const onBeforeUnload = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [dirty]);

  const onZonesChange = (zones: Zone[], removedZoneId?: string) => {
    setDraft((d) => ({
      ...d,
      zones,
      rules: removedZoneId
        ? d.rules.map((r) => {
            if (!r.zones?.includes(removedZoneId)) return r;
            const left = r.zones.filter((z) => z !== removedZoneId);
            return { ...r, zones: left.length ? left : null };
          })
        : d.rules,
    }));
  };

  const onSave = () => {
    if (error) {
      toast(error, 'bad');
      return;
    }
    if (isNew && existing.some((c) => c.camera_id === draft.camera_id)) {
      toast(`Камера ${draft.camera_id} уже существует`, 'bad');
      return;
    }
    save.mutate(
      { camera: draft, isNew },
      {
        onSuccess: () => {
          toast(isNew ? 'Камера добавлена' : 'Конфигурация сохранена — применится без перезапуска');
          if (isNew) navigate(`/cameras/${encodeURIComponent(draft.camera_id)}`, { replace: true });
        },
        onError: (err) => toast(`Не сохранено: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  const onDelete = () => {
    if (!window.confirm(`Удалить камеру «${original.name}» (${original.camera_id})? Журнал событий сохранится.`)) return;
    remove.mutate(original.camera_id, {
      onSuccess: () => {
        toast('Камера удалена');
        navigate('/cameras', { replace: true });
      },
      onError: (err) => toast(`Не удалено: ${(err as Error).message}`, 'bad'),
    });
  };

  return (
    <section className="panel cam-editor reveal" style={{ ['--i' as string]: 1 }}>
      <header className="cam-editor__head">
        <div className="cam-editor__title">
          <span className="eyebrow">{isNew ? 'Новая камера' : draft.site || 'Без участка'}</span>
          <h2>{draft.name || 'Без названия'}</h2>
          <span className="mono muted">{draft.camera_id}</span>
        </div>
        <div className="cam-editor__actions">
          {dirty && <span className="dirty-mark">Есть несохранённые изменения</span>}
          {!isNew && (
            <button type="button" className="btn btn--danger btn--sm" onClick={onDelete} disabled={remove.isPending}>
              <Trash2 size={14} /> Удалить
            </button>
          )}
          <button type="button" className="btn btn--sm" onClick={() => setDraft(structuredClone(original))} disabled={!dirty}>
            <RotateCcw size={14} /> Отменить
          </button>
          <button
            type="button"
            className="btn btn--primary btn--sm"
            onClick={onSave}
            disabled={(!dirty && !isNew) || save.isPending}
            title={error ?? 'PUT /api/cameras/{id}'}
          >
            <Save size={14} /> {save.isPending ? 'Сохранение…' : isNew ? 'Создать камеру' : 'Сохранить'}
          </button>
        </div>
      </header>

      <div className="tabs cam-editor__tabs" role="tablist">
        <button type="button" role="tab" aria-selected={tab === 'zones'} onClick={() => setTab('zones')}>
          <Pentagon size={15} /> Зоны <span className="count">{draft.zones.length}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === 'rules'} onClick={() => setTab('rules')}>
          <ShieldCheck size={15} /> Правила <span className="count">{draft.rules.length}</span>
        </button>
        <button type="button" role="tab" aria-selected={tab === 'source'} onClick={() => setTab('source')}>
          <Video size={15} /> Источник
        </button>
      </div>

      <div className="cam-editor__body">
        {error && dirty && <div className="error-box cam-editor__error">{error}</div>}
        {tab === 'zones' &&
          (isNew ? (
            <div className="empty">
              <span className="empty__title">Сначала создайте камеру</span>
              <span>Стоп-кадр для рисования зон появится, когда камера начнёт передавать видео</span>
            </div>
          ) : (
            <ZoneEditor camera={draft} onChange={onZonesChange} />
          ))}
        {tab === 'rules' && <RulesEditor camera={draft} onChange={(rules) => setDraft((d) => ({ ...d, rules }))} />}
        {tab === 'source' && <SourceForm camera={draft} isNew={isNew} onChange={setDraft} />}
      </div>
    </section>
  );
}

export function CamerasPage() {
  const { cameraId } = useParams();
  const navigate = useNavigate();
  const { data: cameras = [], isLoading, isError, error } = useCameras();
  const metrics = useLive((s) => s.metrics);
  const isNew = cameraId === 'new';
  const newCamera = useMemo(() => blankCamera(cameras), [cameras]);

  if (!cameraId && cameras.length > 0) return <Navigate to={`/cameras/${encodeURIComponent(cameras[0].camera_id)}`} replace />;

  const current = isNew ? newCamera : cameras.find((c) => c.camera_id === cameraId);

  return (
    <div className="page cameras">
      <aside className="cam-list panel reveal">
        <div className="cam-list__head">
          <span className="eyebrow">Камеры · {cameras.length}</span>
          <button type="button" className="btn btn--sm" onClick={() => navigate('/cameras/new')}>
            <Plus size={14} /> Добавить
          </button>
        </div>
        {isError && <div className="error-box">Список камер не загрузился: {(error as Error).message}</div>}
        <ul>
          {cameras.map((c) => {
            const m = metrics?.cameras?.[c.camera_id];
            const online = c.enabled && (m?.fps_processed ?? 0) > 0;
            return (
              <li key={c.camera_id}>
                <button
                  type="button"
                  className={`cam-item${c.camera_id === cameraId ? ' is-selected' : ''}`}
                  onClick={() => navigate(`/cameras/${encodeURIComponent(c.camera_id)}`)}
                >
                  <span className={`live-dot${online ? '' : ' live-dot--bad'}`} />
                  <span className="cam-item__text">
                    <strong>{c.name}</strong>
                    <small>
                      {c.site} · {SOURCE_TYPE_LABELS[c.source.type]}
                    </small>
                  </span>
                  <span className="cam-item__meta">
                    <span title="Зоны">{c.zones.length} з.</span>
                    <span title="Правила">{c.rules.filter((r) => r.enabled).length} пр.</span>
                    <span className="num">{m ? `${formatNumber(m.fps_processed, 1)} к/с` : '—'}</span>
                  </span>
                </button>
              </li>
            );
          })}
          {isNew && (
            <li>
              <button type="button" className="cam-item is-selected">
                <Plus size={14} />
                <span className="cam-item__text">
                  <strong>Новая камера</strong>
                  <small>не сохранена</small>
                </span>
              </button>
            </li>
          )}
        </ul>
      </aside>

      {current ? (
        <CameraEditor key={isNew ? 'new' : current.camera_id} original={current} isNew={isNew} existing={cameras} />
      ) : (
        <section className="panel empty">
          {isLoading ? (
            <span className="spinner" />
          ) : (
            <>
              <Cctv size={30} />
              <span className="empty__title">{cameraId ? `Камера ${cameraId} не найдена` : 'Камер пока нет'}</span>
              <button type="button" className="btn btn--primary" onClick={() => navigate('/cameras/new')}>
                <Plus size={15} /> Добавить камеру
              </button>
            </>
          )}
        </section>
      )}
    </div>
  );
}
