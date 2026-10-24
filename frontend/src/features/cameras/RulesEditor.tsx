import { ChevronDown, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useRuleTypes } from '../../api/queries';
import { SEVERITY_COLOR } from '../../components/ui/badges';
import { SEVERITY_LABELS, SEVERITY_ORDER, ZONE_KIND_COLORS } from '../../domain/labels';
import type { CameraConfig, RuleConfig, RuleTypeInfo, Severity } from '../../domain/types';
import { getViolationMeta } from '../../domain/violations';
import { schemaDefaults, SchemaForm, validateSchema } from './SchemaForm';

/** Used by cards and the camera Save gate; no effect/callback validity race. */
export function validateRule(rule: RuleConfig, camera: CameraConfig, info?: RuleTypeInfo): string[] {
  if (!info) return [`${rule.rule_id}: описание типа правила ещё не загружено`];
  const errors = validateSchema(info.params_schema, rule.params);
  const selected = rule.zones ?? [];
  if (info.zone_schema) errors.push(...validateSchema(info.zone_schema, selected, 'Зоны'));
  if (new Set(selected).size !== selected.length) errors.push('Зоны: повторяющиеся ID');
  if (selected.some((id) => !camera.zones.some((zone) => zone.zone_id === id))) errors.push('Зоны: выбранная зона отсутствует');
  for (const [name, value, min, max] of [
    ['Длится не меньше', rule.min_duration_s, 0, undefined], ['Не повторять', rule.cooldown_s, 0, undefined],
    ['Доля кадров в окне', rule.min_ratio, 0, 1], ['Закрыть через', rule.close_after_s, 0, undefined],
  ] as const) {
    errors.push(...validateSchema({ type: 'number', minimum: min, maximum: max }, value, name));
  }
  return errors;
}

export function validateRules(camera: CameraConfig, types: RuleTypeInfo[]): string | null {
  for (const rule of camera.rules) {
    const error = validateRule(rule, camera, types.find((info) => info.type === rule.type))[0];
    if (error) return `Правило ${rule.rule_id}: ${error}`;
  }
  return null;
}

/** Strict dependent rules must be reassigned/removed before their zone is deleted. */
export function zoneRemovalError(camera: CameraConfig, zoneId: string, types: RuleTypeInfo[]): string | null {
  const blocked = camera.rules.filter((rule) => {
    if (!rule.zones?.includes(zoneId)) return false;
    const info = types.find((item) => item.type === rule.type);
    return !info || (info.zone_schema && validateSchema(info.zone_schema, rule.zones.filter((id) => id !== zoneId)).length > 0);
  });
  return blocked.length ? `Сначала переназначьте зону или удалите связанные правила: ${blocked.map((r) => r.rule_id).join(', ')}.` : null;
}

interface Props {
  camera: CameraConfig;
  onChange: (rules: RuleConfig[]) => void;
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  unit,
  hint,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max?: number;
  step: number;
  unit: string;
  hint?: string;
}) {
  return (
    <label className="field" title={hint}>
      <span className="field__label">{label}</span>
      <div className="num-field__input">
        <input
          className="input input--sm"
          type="number"
          min={min}
          max={max}
          step={step}
          value={Number.isFinite(value) ? value : ''}
          aria-invalid={!Number.isFinite(value) || value < min || (max != null && value > max)}
          onChange={(e) => onChange(e.target.value === '' ? NaN : Number(e.target.value))}
        />
        <span className="num-field__unit">{unit}</span>
      </div>
    </label>
  );
}

function RuleCard({
  rule,
  camera,
  info,
  onChange,
  onRemove,
}: {
  rule: RuleConfig;
  camera: CameraConfig;
  info: RuleTypeInfo | undefined;
  onChange: (r: RuleConfig) => void;
  onRemove: () => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const meta = getViolationMeta(rule.type);
  const Icon = meta.icon;
  const zonesSel = rule.zones ?? [];
  const zoneSchema = info?.zone_schema;
  const errors = validateRule(rule, camera, info);
  const set = (patch: Partial<RuleConfig>) => onChange({ ...rule, ...patch });

  return (
    <article className={`rule-card${rule.enabled ? '' : ' is-off'}`} style={{ ['--c' as string]: SEVERITY_COLOR[rule.severity] }}>
      <header className="rule-card__head">
        <span className="vicon">
          <Icon size={17} />
        </span>
        <div className="rule-card__title">
          <strong>{info?.title ?? meta.title}</strong>
          <span className="mono muted" title={`rule_id: ${rule.rule_id}`}>
            {rule.type}
            {info?.required_models?.length ? ` · ${info.required_models.join('+')}` : ''}
          </span>
        </div>
        <select
          className="select select--sm rule-card__sev"
          value={rule.severity}
          onChange={(e) => set({ severity: e.target.value as Severity })}
          aria-label="Важность"
        >
          {SEVERITY_ORDER.map((s) => (
            <option key={s} value={s}>
              {SEVERITY_LABELS[s]}
            </option>
          ))}
        </select>
        <label className="switch" title={rule.enabled ? 'Правило включено' : 'Правило выключено'}>
          <input type="checkbox" checked={rule.enabled} onChange={(e) => set({ enabled: e.target.checked })} />
          <span className="switch__track" />
          <span className="sr-only">Включено</span>
        </label>
        <button type="button" className="btn btn--ghost btn--sm btn--icon" onClick={onRemove} aria-label="Удалить правило" title="Удалить правило">
          <Trash2 size={15} />
        </button>
      </header>

      <div className="rule-card__body">
        <div className="field">
          <span className="field__label">Где действует</span>
          <div className="chips">
            {!(zoneSchema?.minItems && zoneSchema.minItems > 0) && <button
              type="button"
              className={`chip${zonesSel.length === 0 ? ' is-on' : ''}`}
              aria-pressed={zonesSel.length === 0}
              onClick={() => set({ zones: null })}
            >
              Весь кадр
            </button>}
            {camera.zones.map((z) => {
              const on = zonesSel.includes(z.zone_id);
              return (
                <button
                  key={z.zone_id}
                  type="button"
                  className={`chip${on ? ' is-on' : ''}`}
                  aria-pressed={on}
                  onClick={() => {
                    const next = on ? zonesSel.filter((id) => id !== z.zone_id)
                      : zoneSchema?.maxItems === 1 ? [z.zone_id] : [...zonesSel, z.zone_id];
                    set({ zones: zoneSchema ? next : next.length ? next : null });
                  }}
                >
                  <i style={{ background: ZONE_KIND_COLORS[z.kind] }} />
                  {z.name}
                </button>
              );
            })}
          </div>
          {zoneSchema && <span className="field__hint">{zoneSchema.minItems === 1 && zoneSchema.maxItems === 1
            ? 'Явно выберите одну зону.' : `Число зон: от ${zoneSchema.minItems ?? 0} до ${zoneSchema.maxItems ?? 'любого'}.`}</span>}
        </div>

        <div className="rule-card__timing">
          <NumberInput
            label="Длится не меньше"
            value={rule.min_duration_s}
            onChange={(v) => set({ min_duration_s: v })}
            min={0}
            step={0.5}
            unit="с"
            hint="Случайный кадр тревогу не поднимает: нарушение должно длиться это время"
          />
          <NumberInput
            label="Не повторять"
            value={rule.cooldown_s}
            onChange={(v) => set({ cooldown_s: v })}
            min={0}
            step={5}
            unit="с"
            hint="Окно дедупликации: повторное нарушение там же не создаёт новое событие"
          />
          {advanced && (
            <>
              <NumberInput
                label="Доля кадров в окне"
                value={rule.min_ratio}
                onChange={(v) => set({ min_ratio: v })}
                min={0}
                max={1}
                step={0.05}
                unit="×"
              />
              <NumberInput
                label="Закрыть через"
                value={rule.close_after_s}
                onChange={(v) => set({ close_after_s: v })}
                min={0}
                step={0.5}
                unit="с"
              />
            </>
          )}
          <button type="button" className="btn btn--ghost btn--sm rule-card__more" onClick={() => setAdvanced((v) => !v)}>
            <ChevronDown size={14} style={{ transform: advanced ? 'rotate(180deg)' : undefined }} />
            {advanced ? 'Меньше' : 'Ещё'}
          </button>
        </div>

        {info ? (
          <div className="field">
            <span className="field__label">Параметры детектора</span>
            <SchemaForm schema={info.params_schema} value={rule.params} onChange={(params) => set({ params })} idPrefix={rule.rule_id} />
          </div>
        ) : (
          <p className="muted">Тип «{rule.type}» не найден в /api/rules/types — параметры недоступны для редактирования</p>
        )}
        {errors.length > 0 && <div className="error-box" role="alert">{errors[0]}</div>}
      </div>
    </article>
  );
}

export function RulesEditor({ camera, onChange }: Props) {
  const { data: types = [], isLoading } = useRuleTypes();
  const [adding, setAdding] = useState('');

  const addRule = (type: string) => {
    const info = types.find((t) => t.type === type);
    if (!info) return;
    const rule: RuleConfig = {
      rule_id: `r-${camera.camera_id.replace(/^cam-/, '')}-${type.split('_')[0]}-${Math.random().toString(36).slice(2, 5)}`,
      type,
      enabled: true,
      zones: info.zone_schema ? [] : null,
      params: schemaDefaults(info.params_schema),
      min_duration_s: 2,
      min_ratio: 0.7,
      close_after_s: 3,
      cooldown_s: 30,
      severity: info.default_severity,
    };
    onChange([...camera.rules, rule]);
    setAdding('');
  };

  return (
    <div className="rules-editor">
      <div className="rules-editor__add">
        <select className="select" value={adding} onChange={(e) => setAdding(e.target.value)} disabled={isLoading} aria-label="Тип правила">
          <option value="">Выберите тип нарушения…</option>
          {types.map((t) => (
            <option key={t.type} value={t.type}>
              {getViolationMeta(t.type).scenario ? `${getViolationMeta(t.type).scenario} · ` : ''}
              {t.title}
            </option>
          ))}
        </select>
        <button type="button" className="btn btn--primary" disabled={!adding} onClick={() => addRule(adding)}>
          <Plus size={16} /> Добавить правило
        </button>
        <span className="muted">Типы и формы параметров приходят из плагинов бэкенда — новый тип не требует правки интерфейса</span>
      </div>
      {camera.rules.length === 0 ? (
        <div className="empty">
          <span className="empty__title">Правил нет</span>
          <span>Камера транслирует видео, но нарушения не ищет</span>
        </div>
      ) : (
        <div className="rules-list">
          {camera.rules.map((r, i) => (
            <RuleCard
              key={r.rule_id}
              rule={r}
              camera={camera}
              info={types.find((t) => t.type === r.type)}
              onChange={(next) => onChange(camera.rules.map((x, j) => (j === i ? next : x)))}
              onRemove={() => onChange(camera.rules.filter((_, j) => j !== i))}
            />
          ))}
        </div>
      )}
    </div>
  );
}
