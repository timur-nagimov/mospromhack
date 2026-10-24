import { CheckCircle2, Plug, TriangleAlert } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import { formatMs, formatNumber } from '../../domain/format';
import { SOURCE_TYPE_OPTION_LABELS } from '../../domain/labels';
import type { CameraConfig, SourceProbeResult, SourceType } from '../../domain/types';

const URI_HINT: Record<SourceType, string> = {
  file: 'Путь к видеофайлу на сервере, например videos/warehouse.mp4',
  rtsp: 'rtsp://, http:// или https:// — форматы адресов ниже',
  images: 'Папка с кадрами одного места (имитация стоп-кадров)',
  webcam: 'Индекс устройства на сервере, например 0',
};

/** Форматы адресов популярных камер и потоков (docs/STREAMS.md) */
const NETWORK_EXAMPLES: { title: string; uri: string; note?: string; template?: boolean }[] = [
  { title: 'Hikvision', uri: 'rtsp://admin:пароль@192.168.1.64:554/Streaming/Channels/102', note: '101 — основной поток, 102 — субпоток', template: true },
  { title: 'Dahua', uri: 'rtsp://admin:пароль@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1', note: 'subtype=0 — основной, 1 — субпоток', template: true },
  { title: 'Axis', uri: 'rtsp://root:пароль@192.168.1.90/axis-media/media.amp?resolution=1280x720', template: true },
  { title: 'ONVIF', uri: 'rtsp://…', note: 'адрес потока из GetStreamUri (ONVIF Device Manager)' },
  { title: 'Телефон (IP Webcam)', uri: 'http://192.168.1.50:8080/video', note: 'HTTP MJPEG', template: true },
  { title: 'HLS', uri: 'https://сервер/камера/index.m3u8', note: 'задержка 1–3 сегмента' },
  { title: 'YouTube Live', uri: 'https://www.youtube.com/watch?v=…', note: 'адрес видео получается заново при подключении' },
];

const STREAM_LABELS: Record<string, string> = {
  rtsp: 'RTSP',
  hls: 'HLS',
  mjpeg: 'HTTP MJPEG',
  youtube: 'YouTube Live (HLS)',
  'http-file': 'Видеофайл по HTTP',
  rtmp: 'RTMP',
  srt: 'SRT',
  file: 'Видеофайл',
  images: 'Папка изображений',
  webcam: 'Веб-камера',
};

const CODEC_LABELS: Record<string, string> = { h264: 'H.264', h265: 'H.265 (HEVC)', mjpeg: 'MJPEG', images: 'JPEG/PNG' };

const PROBE_TIMEOUT_S = 15;

export const CAMERA_ID_RE = /^[a-z0-9][a-z0-9_-]{1,31}$/;

type ProbeState =
  | { status: 'idle' }
  | { status: 'running'; startedAt: number }
  | { status: 'done'; result: SourceProbeResult }
  | { status: 'failed'; error: string };

/** «Проверить подключение»: POST /api/cameras/probe — превью, разрешение, к/с, кодек или понятная ошибка */
function SourceProbe({ camera }: { camera: CameraConfig }) {
  const [probe, setProbe] = useState<ProbeState>({ status: 'idle' });
  const [, setTick] = useState(0);
  const sourceKey = `${camera.source.type}|${camera.source.uri.trim()}`;
  const pending = useRef<string | null>(null);

  // смена адреса или типа делает прежний результат неактуальным
  useEffect(() => {
    pending.current = null;
    setProbe({ status: 'idle' });
  }, [sourceKey]);

  useEffect(() => {
    if (probe.status !== 'running') return;
    const id = window.setInterval(() => setTick((t) => t + 1), 500);
    return () => window.clearInterval(id);
  }, [probe.status]);

  const run = async () => {
    const key = sourceKey;
    pending.current = key;
    setProbe({ status: 'running', startedAt: Date.now() });
    try {
      const result = await api().probeSource(camera.source, PROBE_TIMEOUT_S);
      if (pending.current === key) setProbe({ status: 'done', result });
    } catch (err) {
      if (pending.current === key) setProbe({ status: 'failed', error: (err as Error).message });
    }
  };

  const running = probe.status === 'running';
  const result = probe.status === 'done' ? probe.result : null;
  const error = probe.status === 'failed' ? probe.error : result && !result.ok ? result.error ?? 'источник не ответил' : null;

  return (
    <div className="probe">
      <div className="probe__bar">
        <button type="button" className="btn btn--sm" onClick={run} disabled={running || !camera.source.uri.trim()}>
          <Plug size={14} /> {running ? 'Подключаюсь…' : 'Проверить подключение'}
        </button>
        <span className="field__hint">
          {running
            ? `${Math.round((Date.now() - probe.startedAt) / 1000)} с — RTSP отвечает за 1–3 с, HLS до 15 с`
            : 'Сервис откроет поток так же, как камеру, и покажет кадр'}
        </span>
      </div>
      {camera.source.type === 'webcam' && (
        <span className="field__hint">Проверка откроет веб-камеру сервера: macOS может спросить разрешение на доступ к камере</span>
      )}
      {result?.ok && (
        <div className="probe-result" role="status">
          {result.preview_jpeg_b64 && (
            <img className="probe-result__preview" src={`data:image/jpeg;base64,${result.preview_jpeg_b64}`} alt="Кадр из потока" />
          )}
          <div className="probe-result__facts">
            <div className="probe-result__title">
              <CheckCircle2 size={16} /> Поток открывается
            </div>
            <dl>
              <dt>Разрешение</dt>
              <dd className="num">{result.width && result.height ? `${result.width}×${result.height}` : '—'}</dd>
              <dt>Кадров в секунду</dt>
              <dd className="num">{result.fps != null ? formatNumber(result.fps, 1) : '—'}</dd>
              <dt>Кодек</dt>
              <dd>{result.codec ? CODEC_LABELS[result.codec] ?? result.codec.toUpperCase() : '—'}</dd>
              <dt>Первый кадр</dt>
              <dd className="num">{formatMs(result.first_frame_ms)}</dd>
              <dt>Протокол</dt>
              <dd>{STREAM_LABELS[result.stream ?? camera.source.type] ?? result.stream ?? '—'}</dd>
            </dl>
            {result.hints?.map((h) => (
              <p key={h} className="probe-result__hint">
                {h}
              </p>
            ))}
          </div>
        </div>
      )}
      {error && (
        <div className="error-box" role="alert">
          <TriangleAlert size={16} />
          <span>Не подключилось: {error}</span>
        </div>
      )}
    </div>
  );
}

export function SourceForm({ camera, isNew, onChange }: { camera: CameraConfig; isNew: boolean; onChange: (c: CameraConfig) => void }) {
  const set = (patch: Partial<CameraConfig>) => onChange({ ...camera, ...patch });
  const setSource = (patch: Partial<CameraConfig['source']>) => onChange({ ...camera, source: { ...camera.source, ...patch } });
  const idInvalid = !CAMERA_ID_RE.test(camera.camera_id);
  const type = camera.source.type;

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
          <select className="select" value={type} onChange={(e) => setSource({ type: e.target.value as SourceType })}>
            {(Object.keys(SOURCE_TYPE_OPTION_LABELS) as SourceType[]).map((t) => (
              <option key={t} value={t}>
                {SOURCE_TYPE_OPTION_LABELS[t]}
              </option>
            ))}
          </select>
        </label>
        <label className="field source-form__uri">
          <span className="field__label">Адрес источника</span>
          <input
            className="input mono"
            value={camera.source.uri}
            onChange={(e) => setSource({ uri: e.target.value })}
            aria-invalid={!camera.source.uri.trim()}
            placeholder={type === 'rtsp' ? 'rtsp://логин:пароль@адрес:554/путь' : undefined}
            spellCheck={false}
          />
          <span className="field__hint">{URI_HINT[type]}</span>
        </label>
      </div>

      {type === 'rtsp' && (
        <div className="stream-formats">
          <span className="field__label">Форматы адресов</span>
          <table>
            <tbody>
              {NETWORK_EXAMPLES.map((ex) => (
                <tr key={ex.title}>
                  <th scope="row">{ex.title}</th>
                  <td>
                    {ex.template ? (
                      <button type="button" className="stream-formats__uri mono" onClick={() => setSource({ uri: ex.uri })} title="Подставить и поправить адрес">
                        {ex.uri}
                      </button>
                    ) : (
                      <span className="stream-formats__uri mono">{ex.uri}</span>
                    )}
                  </td>
                  <td className="muted">{ex.note ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <span className="field__hint">
            Для анализа достаточно субпотока 640–1280 px. H.265 работает, но декодируется дороже H.264. RTSP идёт по TCP.
          </span>
        </div>
      )}

      <SourceProbe camera={camera} />

      <div className="source-form__switches">
        <label className="switch">
          <input type="checkbox" checked={camera.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
          <span className="switch__track" />
          Камера включена
        </label>
        {(type === 'file' || type === 'images') && (
          <label className="switch">
            <input type="checkbox" checked={camera.source.loop} onChange={(e) => setSource({ loop: e.target.checked })} />
            <span className="switch__track" />
            Зациклить файл
          </label>
        )}
        {type !== 'webcam' && (
          <label className="switch" title={type === 'rtsp' ? 'HLS и файлы по HTTP приходят пачками: кадры выдаются в темпе их меток времени' : undefined}>
            <input type="checkbox" checked={camera.source.realtime} onChange={(e) => setSource({ realtime: e.target.checked })} />
            <span className="switch__track" />
            {type === 'rtsp' ? 'Выравнивать темп HLS по меткам времени' : 'Отдавать файл в реальном темпе'}
          </label>
        )}
      </div>
    </div>
  );
}
