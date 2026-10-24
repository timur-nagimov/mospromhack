import { ArrowLeft, ArrowRight, Ban, Check, CornerDownRight, ExternalLink, Plus, RotateCcw, Save, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { api } from '../../api';
import { usePatchSample } from '../../api/flywheelQueries';
import { useCameraMap } from '../../api/queries';
import { toast } from '../../components/ui/toast';
import {
  classColor,
  classLabel,
  isHoldout,
  OPERATOR_LABEL_TEXT,
  ORIGIN_LABELS,
  sampleStatusLabel,
  sourceLabel,
  type FlywheelSample,
  type LabelOrigin,
  type OperatorLabel,
  type SampleLabel,
} from '../../domain/flywheel';
import { formatDateTime, formatSmartDateTime, shortId } from '../../domain/format';
import { useEventModal } from '../event/useEventModal';
import { FrameBoxes, type FrameBox } from './FrameBoxes';
import { HoldoutBadge, SampleStatusBadge, SourceTag } from './parts';

type View = 'labels' | 'detections' | 'both';

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable;
}

function originLabel(origin: string): string {
  return ORIGIN_LABELS[origin as LabelOrigin] ?? origin;
}

const sameLabels = (a: SampleLabel[], b: SampleLabel[]) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Карточка сэмпла: кадр с рамками меток и детекций, правка меток, решение 1/2.
 * Черновик меток живёт, пока открыт этот сэмпл (key={sample.id} у родителя).
 */
export function SampleCard({
  sample,
  classes,
  hasPrev,
  hasNext,
  onPrev,
  onNext,
  onDecided,
}: {
  sample: FlywheelSample;
  /** варианты класса для смены */
  classes: string[];
  hasPrev: boolean;
  hasNext: boolean;
  onPrev: () => void;
  onNext: () => void;
  /** после «принять» / «отклонить» — к следующему сэмплу */
  onDecided: () => void;
}) {
  const cameras = useCameraMap();
  const { eventId: openEventId, openEvent } = useEventModal();
  const patch = usePatchSample();
  const [draft, setDraft] = useState<SampleLabel[]>(sample.labels ?? []);
  const [dirty, setDirty] = useState(false);
  const [view, setView] = useState<View>((sample.labels ?? []).length > 0 ? 'labels' : 'detections');
  const [hover, setHover] = useState<string | null>(null);
  const detections = sample.detections ?? [];
  const camera = cameras.get(sample.camera_id);
  const holdout = isHoldout(sample);
  const frameSrc = useMemo(() => api().flywheel.frameUrl(sample, 'full'), [sample]);

  // Метки сэмпла обновились на сервере (учитель разметил, сохранение) — черновик без правок догоняет
  useEffect(() => {
    if (!dirty) setDraft(sample.labels ?? []);
  }, [sample.labels, dirty]);

  const boxes: FrameBox[] = [
    ...(view !== 'labels' ? detections.map((d, i) => ({ key: `d${i}`, cls: d.cls, bbox: d.bbox, kind: 'detection' as const, conf: d.conf })) : []),
    ...(view !== 'detections' ? draft.map((l, i) => ({ key: `l${i}`, cls: l.cls, bbox: l.bbox, kind: 'label' as const, conf: l.conf ?? null })) : []),
  ];

  const editLabels = (next: SampleLabel[]) => {
    setDraft(next);
    setDirty(!sameLabels(next, sample.labels ?? []));
  };
  const changeClass = (i: number, cls: string) => editLabels(draft.map((l, j) => (j === i ? { ...l, cls, origin: 'operator', conf: null } : l)));
  const removeBox = (i: number) => {
    editLabels(draft.filter((_, j) => j !== i));
    setHover(null);
  };
  const promote = (i: number) => {
    const d = detections[i];
    editLabels([...draft, { cls: d.cls, bbox: d.bbox, origin: 'operator', conf: null }]);
    if (view === 'detections') setView('both');
  };

  const save = (status?: 'accepted' | 'rejected') => {
    if (patch.isPending) return;
    const body = { ...(status ? { status } : {}), ...(dirty ? { labels: draft } : {}) };
    if (!status && !dirty) return;
    // mutateAsync: после решения карточка сразу уходит к следующему сэмплу,
    // а уведомление об ошибке должно дойти и после размонтирования
    patch
      .mutateAsync({ id: sample.id, patch: body })
      .then((updated) => {
        setDirty(false);
        setDraft(updated.labels ?? []);
        toast(status ? `Сэмпл #${shortId(sample.id)} ${status === 'accepted' ? 'принят' : 'отклонён'}` : 'Метки сохранены');
      })
      .catch((err: Error) => toast(`Сэмпл #${shortId(sample.id)} не сохранён: ${err.message}`, 'bad'));
    if (status) onDecided();
  };

  // Клавиши как в карточке события: 1 — принять, 2 — отклонить; ←/→ — соседний сэмпл.
  // Пока поверх открыта карточка события, её клавиши 1/2/3 главнее.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (openEventId || isTypingTarget(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === '1' && !e.repeat) {
        e.preventDefault();
        save('accepted');
      } else if (e.key === '2' && !e.repeat) {
        e.preventDefault();
        save('rejected');
      } else if (e.key === 'ArrowRight' && hasNext) {
        e.preventDefault();
        onNext();
      } else if (e.key === 'ArrowLeft' && hasPrev) {
        e.preventDefault();
        onPrev();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const opLabel = sample.operator_label ? OPERATOR_LABEL_TEXT[sample.operator_label as OperatorLabel] ?? sample.operator_label : null;

  return (
    <section className="panel fw-card" aria-label="Карточка сэмпла">
      <header className="fw-card__head">
        <div className="fw-card__title">
          <div className="fw-card__kicker">
            <span className="eyebrow">Сэмпл</span>
            <span className="mono muted" title={sample.id}>
              #{shortId(sample.id)}
            </span>
          </div>
          <h2>
            {sourceLabel(sample.source)} · {camera?.name ?? sample.camera_id}
          </h2>
          <p>
            {formatSmartDateTime(sample.created_at)} · {sample.width}×{sample.height}
          </p>
        </div>
        <div className="fw-card__badges">
          <SourceTag source={sample.source} />
          <SampleStatusBadge status={sample.status} />
          {holdout && <HoldoutBadge />}
        </div>
      </header>

      <div className="fw-card__body">
        <div className="fw-card__media">
          <div className="fw-card__toolbar">
            <div className="segmented" role="group" aria-label="Какие рамки показать">
              <button type="button" aria-pressed={view === 'labels'} onClick={() => setView('labels')}>
                Метки <span className="num muted">{draft.length}</span>
              </button>
              <button type="button" aria-pressed={view === 'detections'} onClick={() => setView('detections')}>
                Детекции <span className="num muted">{detections.length}</span>
              </button>
              <button type="button" aria-pressed={view === 'both'} onClick={() => setView('both')}>
                Обе
              </button>
            </div>
            <span className="fw-legend">
              <span className="fw-legend__solid" /> метка
              <span className="fw-legend__dash" /> детекция модели
            </span>
          </div>
          <div className="fw-card__frame">
            <FrameBoxes
              src={frameSrc}
              width={sample.width}
              height={sample.height}
              boxes={boxes}
              highlight={hover}
              onHover={setHover}
              alt={`Кадр сэмпла: ${sourceLabel(sample.source)}, ${camera?.name ?? sample.camera_id}`}
            />
          </div>
          {holdout && (
            <p className="fw-note fw-note--holdout">
              Кадр отложенного ролика: в обучающую выборку не экспортируется, на нём меряется качество.
            </p>
          )}
        </div>

        <aside className="fw-card__side">
          <section className="fw-reason">
            <span className="eyebrow">Причина отбора</span>
            <p className="mono">{sample.reason || '—'}</p>
          </section>

          {view !== 'detections' && (
            <section className="fw-boxes" aria-label="Метки">
              <div className="fw-boxes__head">
                <span className="eyebrow">Метки</span>
                {dirty && <span className="fw-dirty">есть правки</span>}
              </div>
              {draft.length === 0 ? (
                <p className="fw-boxes__empty">
                  {sample.status === 'new' ? 'Учитель ещё не разметил кадр. Рамку можно взять из детекций.' : 'Меток нет: кадр пойдёт как фон.'}
                </p>
              ) : (
                <ul>
                  {draft.map((l, i) => (
                    <li
                      key={`l${i}`}
                      className={hover === `l${i}` ? 'is-hover' : undefined}
                      onMouseEnter={() => setHover(`l${i}`)}
                      onMouseLeave={() => setHover(null)}
                    >
                      <i style={{ background: classColor(l.cls) }} aria-hidden="true" />
                      <select
                        className="select select--sm"
                        value={l.cls}
                        aria-label={`Класс рамки ${i + 1}`}
                        onChange={(e) => changeClass(i, e.target.value)}
                      >
                        {[...new Set([l.cls, ...classes])].map((c) => (
                          <option key={c} value={c}>
                            {classLabel(c) === c ? c : `${classLabel(c)} · ${c}`}
                          </option>
                        ))}
                      </select>
                      <span className="fw-boxes__meta">
                        {originLabel(l.origin)}
                        {l.conf != null ? ` ${l.conf.toFixed(2)}` : ''}
                      </span>
                      <button type="button" className="btn btn--ghost btn--sm btn--icon" onClick={() => removeBox(i)} aria-label={`Удалить рамку ${i + 1}`} title="Удалить рамку">
                        <Trash2 size={14} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          {view !== 'labels' && (
            <section className="fw-boxes" aria-label="Детекции">
              <div className="fw-boxes__head">
                <span className="eyebrow">Детекции моделей</span>
              </div>
              {detections.length === 0 ? (
                <p className="fw-boxes__empty">Моделей на этом кадре не было: импорт с готовой разметкой.</p>
              ) : (
                <ul>
                  {detections.map((d, i) => (
                    <li
                      key={`d${i}`}
                      className={hover === `d${i}` ? 'is-hover' : undefined}
                      onMouseEnter={() => setHover(`d${i}`)}
                      onMouseLeave={() => setHover(null)}
                    >
                      <i className="fw-boxes__dash" style={{ borderColor: classColor(d.cls) }} aria-hidden="true" />
                      <span className="fw-boxes__cls">{classLabel(d.cls)}</span>
                      <span className="fw-boxes__meta mono">
                        {d.model_id} {d.conf.toFixed(2)}
                        {d.track_id != null ? ` · #${d.track_id}` : ''}
                      </span>
                      <button type="button" className="btn btn--ghost btn--sm btn--icon" onClick={() => promote(i)} aria-label={`Взять детекцию ${i + 1} в метки`} title="Взять в метки">
                        <Plus size={14} />
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          )}

          <dl className="fw-facts">
            <div>
              <dt>Камера</dt>
              <dd>
                {camera?.name ?? sample.camera_id} <span className="mono muted">{sample.camera_id}</span>
              </dd>
            </div>
            <div>
              <dt>Время кадра</dt>
              <dd className="num">{formatDateTime(sample.created_at)}</dd>
            </div>
            {sample.event_id && (
              <div>
                <dt>Событие</dt>
                <dd>
                  <button type="button" className="fw-link" onClick={() => openEvent(sample.event_id as string)}>
                    #{shortId(sample.event_id)} <ExternalLink size={13} aria-hidden="true" />
                  </button>
                </dd>
              </div>
            )}
            {opLabel && (
              <div>
                <dt>Решение оператора</dt>
                <dd className={sample.operator_label === 'hard_negative' ? 'fw-hardneg' : undefined}>{opLabel}</dd>
              </div>
            )}
            <div>
              <dt>Статус</dt>
              <dd>
                {sampleStatusLabel(sample.status)}
                {sample.reviewed_at ? <span className="muted"> · {formatSmartDateTime(sample.reviewed_at)}</span> : null}
              </dd>
            </div>
            {sample.dataset_version && (
              <div>
                <dt>В датасете</dt>
                <dd className="mono">{sample.dataset_version}</dd>
              </div>
            )}
          </dl>
        </aside>
      </div>

      <footer className="fw-card__actions">
        <div className="fw-decisions" role="group" aria-label="Решение по сэмплу">
          <button
            type="button"
            className={`decision decision--confirmed${sample.status === 'accepted' ? ' is-active' : ''}`}
            onClick={() => save('accepted')}
            disabled={patch.isPending}
            aria-pressed={sample.status === 'accepted'}
          >
            <Check size={18} strokeWidth={2.4} aria-hidden="true" />
            <span>{dirty ? 'Принять с правками' : 'Принять'}</span>
            <span className="kbd">1</span>
          </button>
          <button
            type="button"
            className={`decision decision--false${sample.status === 'rejected' ? ' is-active' : ''}`}
            onClick={() => save('rejected')}
            disabled={patch.isPending}
            aria-pressed={sample.status === 'rejected'}
          >
            <Ban size={18} strokeWidth={2.4} aria-hidden="true" />
            <span>Отклонить</span>
            <span className="kbd">2</span>
          </button>
        </div>
        <div className="fw-card__nav">
          <button type="button" className="btn btn--sm" onClick={() => save()} disabled={!dirty || patch.isPending} title="Сохранить метки без решения">
            <Save size={14} /> Сохранить метки
          </button>
          <button type="button" className="btn btn--sm btn--ghost" onClick={() => editLabels(sample.labels ?? [])} disabled={!dirty} title="Вернуть метки сервера">
            <RotateCcw size={14} /> Сбросить
          </button>
          <span className="fw-card__spacer" />
          <button type="button" className="btn btn--sm btn--ghost" onClick={onPrev} disabled={!hasPrev} aria-label="Предыдущий сэмпл">
            <ArrowLeft size={14} /> <span className="kbd">←</span>
          </button>
          <button type="button" className="btn btn--sm btn--ghost" onClick={onNext} disabled={!hasNext} aria-label="Следующий сэмпл">
            <span className="kbd">→</span> <ArrowRight size={14} />
          </button>
        </div>
        {sample.operator_label === 'hard_negative' && (
          <p className="fw-note">
            <CornerDownRight size={14} aria-hidden="true" /> Оператор отметил событие ложным: рамка нарушителя войдёт в датасет как hard negative.
          </p>
        )}
      </footer>
    </section>
  );
}
