import { SOURCE_TYPE_LABELS } from '../../domain/labels';
import type { CameraConfig, SourceType } from '../../domain/types';

const URI_HINT: Record<SourceType, string> = {
  file: 'Путь к видеофайлу на сервере, например data/videos/warehouse.mp4',
  rtsp: 'rtsp://адрес:порт/поток',
  images: 'Папка с кадрами одного места (имитация стоп-кадров)',
  webcam: 'Индекс устройства, например 0',
};

export const CAMERA_ID_RE = /^[a-z0-9][a-z0-9_-]{1,31}$/;

export function SourceForm({ camera, isNew, onChange }: { camera: CameraConfig; isNew: boolean; onChange: (c: CameraConfig) => void }) {
  const set = (patch: Partial<CameraConfig>) => onChange({ ...camera, ...patch });
  const setSource = (patch: Partial<CameraConfig['source']>) => onChange({ ...camera, source: { ...camera.source, ...patch } });
  const idInvalid = !CAMERA_ID_RE.test(camera.camera_id);

  return (
    <div className="source-form">
      <div className="source-form__grid">
        <label className="field">
          <span className="field__label">ID камеры</span>
          <input
            className="input mono"
            value={camera.camera_id}
            disabled={!isNew}
            aria-invalid={isNew && idInvalid}
            onChange={(e) => set({ camera_id: e.target.value.trim().toLowerCase() })}
            placeholder="cam-07"
          />
          <span className={isNew && idInvalid ? 'field__error' : 'field__hint'}>
            {isNew ? 'латиница, цифры, «-» и «_»; после создания не меняется' : 'ID не меняется после создания'}
          </span>
        </label>
        <label className="field">
          <span className="field__label">Название</span>
          <input className="input" value={camera.name} onChange={(e) => set({ name: e.target.value })} placeholder="Стеллажи, ряд Б" aria-invalid={!camera.name.trim()} />
        </label>
        <label className="field">
          <span className="field__label">Участок</span>
          <input className="input" value={camera.site} onChange={(e) => set({ site: e.target.value })} placeholder="Склад" />
          <span className="field__hint">По участкам строится аналитика</span>
        </label>
        <label className="field">
          <span className="field__label">Частота анализа</span>
          <div className="num-field__input">
            <input
              className="input"
              type="number"
              min={0.5}
              max={30}
              step={0.5}
              value={camera.sample_fps}
              onChange={(e) => set({ sample_fps: Number(e.target.value) })}
            />
            <span className="num-field__unit">кадров/с</span>
          </div>
          <span className="field__hint">Сколько кадров в секунду отправлять в нейросеть</span>
        </label>
        <label className="field">
          <span className="field__label">Тип источника</span>
          <select className="select" value={camera.source.type} onChange={(e) => setSource({ type: e.target.value as SourceType })}>
            {(Object.keys(SOURCE_TYPE_LABELS) as SourceType[]).map((t) => (
              <option key={t} value={t}>
                {SOURCE_TYPE_LABELS[t]}
              </option>
            ))}
          </select>
        </label>
        <label className="field source-form__uri">
          <span className="field__label">Адрес источника</span>
          <input className="input mono" value={camera.source.uri} onChange={(e) => setSource({ uri: e.target.value })} aria-invalid={!camera.source.uri.trim()} />
          <span className="field__hint">{URI_HINT[camera.source.type]}</span>
        </label>
      </div>
      <div className="source-form__switches">
        <label className="switch">
          <input type="checkbox" checked={camera.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
          <span className="switch__track" />
          Камера включена
        </label>
        <label className="switch">
          <input type="checkbox" checked={camera.source.loop} onChange={(e) => setSource({ loop: e.target.checked })} />
          <span className="switch__track" />
          Зациклить файл
        </label>
        <label className="switch">
          <input type="checkbox" checked={camera.source.realtime} onChange={(e) => setSource({ realtime: e.target.checked })} />
          <span className="switch__track" />
          Отдавать файл в реальном темпе
        </label>
      </div>
    </div>
  );
}
