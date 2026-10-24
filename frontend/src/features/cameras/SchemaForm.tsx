import type { JsonSchema } from '../../domain/types';

// Генератор формы по JSON Schema из /api/rules/types: правила настраиваются без кода (D1, R7)

interface Props {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  idPrefix: string;
}

// Invalid JSON is part of the current draft, never a hidden old valid value.
const JSON_DRAFT = '__rule_config_json_draft__';
const EMPTY_DRAFT = '__rule_config_empty_draft__';
function isEmptyDraft(value: unknown): boolean {
  return value != null && typeof value === 'object' && Object.keys(value).length === 1
    && (value as Record<string, unknown>)[EMPTY_DRAFT] === true;
}
function isJsonDraft(value: unknown): value is Record<typeof JSON_DRAFT, string> {
  return value != null && typeof value === 'object' && Object.keys(value).length === 1
    && typeof (value as Record<string, unknown>)[JSON_DRAFT] === 'string';
}

/** Shared render/save validation: reads the current draft synchronously. */
export function validateSchema(schema: JsonSchema, value: unknown, label = 'Параметры'): string[] {
  if (isEmptyDraft(value)) return [`${label}: обязательное значение`];
  if (isJsonDraft(value)) return [`${label}: некорректный JSON`];
  if (typeof value === 'number' && !Number.isFinite(value)) return [`${label}: требуется конечное число`];
  if (value === undefined) return [`${label}: обязательное значение`];
  if (schema.type === 'number' || schema.type === 'integer') {
    if (typeof value !== 'number' || !Number.isFinite(value)) return [`${label}: требуется конечное число`];
    if (schema.type === 'integer' && !Number.isInteger(value)) return [`${label}: требуется целое число`];
    if (schema.minimum != null && value < schema.minimum) return [`${label}: минимум ${schema.minimum}`];
    if (schema.maximum != null && value > schema.maximum) return [`${label}: максимум ${schema.maximum}`];
    if (schema.exclusiveMinimum != null && value <= schema.exclusiveMinimum) return [`${label}: должно быть больше ${schema.exclusiveMinimum}`];
    if (schema.exclusiveMaximum != null && value >= schema.exclusiveMaximum) return [`${label}: должно быть меньше ${schema.exclusiveMaximum}`];
    if (schema.multipleOf != null && Math.abs(value / schema.multipleOf - Math.round(value / schema.multipleOf)) > 1e-8)
      return [`${label}: шаг ${schema.multipleOf}`];
  } else if (schema.type === 'array') {
    if (!Array.isArray(value)) return [`${label}: требуется массив`];
    if (schema.minItems != null && value.length < schema.minItems) return [`${label}: нужно не меньше ${schema.minItems}`];
    if (schema.maxItems != null && value.length > schema.maxItems) return [`${label}: нужно не больше ${schema.maxItems}`];
    if (schema.uniqueItems && new Set(value.map((v) => JSON.stringify(v))).size !== value.length)
      return [`${label}: значения не должны повторяться`];
    return value.flatMap((item, i) => validateSchema(schema.items ?? {}, item, `${label} [${i + 1}]`));
  } else if (schema.type === 'object') {
    if (value == null || typeof value !== 'object' || Array.isArray(value)) return [`${label}: требуется объект`];
    const object = value as Record<string, unknown>;
    const errors = (schema.required ?? []).filter((key) => object[key] === undefined)
      .map((key) => `${schema.properties?.[key]?.title ?? key}: обязательное значение`);
    for (const [key, item] of Object.entries(object)) {
      const prop = schema.properties?.[key];
      if (prop || schema.additionalProperties !== false) errors.push(...validateSchema(prop ?? {}, item, prop?.title ?? key));
      else errors.push(`${key}: неизвестный параметр`);
    }
    return errors;
  } else if (schema.type === 'string') {
    if (typeof value !== 'string') return [`${label}: требуется строка`];
    if (schema.minLength != null && value.length < schema.minLength) return [`${label}: слишком короткая строка`];
    if (schema.maxLength != null && value.length > schema.maxLength) return [`${label}: слишком длинная строка`];
  } else if (schema.type === 'boolean' && typeof value !== 'boolean') return [`${label}: требуется да/нет`];
  if (schema.enum && !schema.enum.some((item) => JSON.stringify(item) === JSON.stringify(value)))
    return [`${label}: выберите допустимое значение`];
  // Even a free-form JSON field must not hide non-finite numbers or invalid drafts.
  if (!schema.type && value != null && typeof value === 'object') {
    return Object.values(value).flatMap((item) => validateSchema({}, item, label));
  }
  return [];
}

/** Numeric 2x2 schema is rendered as points; no direction is invented. */
function isPointPair(prop: JsonSchema): boolean {
  return prop.type === 'array' && prop.minItems === 2 && prop.maxItems === 2
    && prop.items?.type === 'array' && prop.items.minItems === 2 && prop.items.maxItems === 2
    && (prop.items.items?.type === 'number' || prop.items.items?.type === 'integer');
}

/** Значения по умолчанию из схемы — для нового правила */
export function schemaDefaults(schema: JsonSchema): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, prop] of Object.entries(schema.properties ?? {})) {
    if (prop.default !== undefined) out[key] = structuredClone(prop.default);
    else if (prop.type === 'object') out[key] = schemaDefaults(prop);
  }
  return out;
}

function stepOf(prop: JsonSchema): number {
  if (prop.multipleOf) return prop.multipleOf;
  return prop.type === 'integer' ? 1 : 0.1;
}

function NumberField({ id, prop, value, onChange }: { id: string; prop: JsonSchema; value: unknown; onChange: (v: number | undefined) => void }) {
  const num = typeof value === 'number' && Number.isFinite(value) ? value : undefined;
  const hasRange = prop.minimum != null && prop.maximum != null;
  const invalid = value !== undefined && validateSchema(prop, value).length > 0;
  return (
    <div className="num-field">
      {hasRange && (
        <input
          type="range"
          className="range"
          min={prop.minimum}
          max={prop.maximum}
          step={stepOf(prop)}
          value={num ?? (prop.default as number) ?? prop.minimum}
          onChange={(e) => onChange(Number(e.target.value))}
          aria-label={prop.title}
        />
      )}
      <div className="num-field__input">
        <input
          id={id}
          className="input input--sm"
          type="number"
          min={prop.minimum}
          max={prop.maximum}
          step={stepOf(prop)}
          value={num ?? ''}
          aria-invalid={invalid}
          aria-label={prop.title}
          onChange={(e) => onChange(e.target.value === '' ? undefined : Number(e.target.value))}
        />
        {prop['x-unit'] && <span className="num-field__unit">{prop['x-unit']}</span>}
      </div>
    </div>
  );
}

function ChipsField({ prop, value, onChange }: { prop: JsonSchema; value: unknown; onChange: (v: unknown[]) => void }) {
  const options = prop.items?.enum ?? [];
  const names = prop.items?.enumNames ?? [];
  const selected = Array.isArray(value) ? value : [];
  return (
    <div className="chips" role="group" aria-label={prop.title}>
      {options.map((opt, i) => {
        const on = selected.includes(opt);
        return (
          <button
            key={String(opt)}
            type="button"
            className={`chip${on ? ' is-on' : ''}`}
            aria-pressed={on}
            onClick={() => onChange(on ? selected.filter((v) => v !== opt) : [...selected, opt])}
          >
            {names[i] ?? String(opt)}
          </button>
        );
      })}
    </div>
  );
}

export function SchemaForm({ schema, value, onChange, idPrefix }: Props) {
  const props = Object.entries(schema.properties ?? {});
  if (props.length === 0) return <p className="muted schema-empty">У правила нет параметров</p>;

  const set = (key: string, v: unknown) => {
    const next = { ...value };
    if (v === undefined) delete next[key];
    else next[key] = v;
    onChange(next);
  };

  return (
    <div className="schema-form">
      {props.map(([key, prop]) => {
        const id = `${idPrefix}-${key}`;
        // Omitted optional parameters use the same default as the backend.
        // An explicit invalid/empty draft is never replaced by that default.
        const current = Object.hasOwn(value, key) ? value[key]
          : schema.required?.includes(key) ? undefined : prop.default;
        const label = prop.title ?? key;
        const error = current === undefined && !schema.required?.includes(key) ? undefined
          : validateSchema(prop, current, label)[0];
        if (prop.type === 'boolean') {
          return (
            <label key={key} className="switch schema-form__bool" title={prop.description}>
              <input type="checkbox" checked={Boolean(current)} onChange={(e) => set(key, e.target.checked)} />
              <span className="switch__track" />
              {label}
            </label>
          );
        }
        let control: React.ReactNode;
        if (prop.type === 'number' || prop.type === 'integer') {
          control = <NumberField id={id} prop={prop} value={current} onChange={(v) => set(key, v ?? { [EMPTY_DRAFT]: true })} />;
        } else if (isPointPair(prop)) {
          const cellSchema = prop.items!.items!;
          control = <fieldset className="schema-form__nested" aria-label={label}>
            {[0, 1].map((point) => <div key={point} className="schema-form__field">
              <span className="field__label">Точка {point + 1}</span>
              {[0, 1].map((axis) => <label key={axis} className="field">
                <span>{axis === 0 ? 'X' : 'Y'}</span>
                <NumberField id={`${id}-${point}-${axis}`} prop={{ ...cellSchema, title: `${label}: точка ${point + 1}, ${axis === 0 ? 'X' : 'Y'}` }}
                  value={Array.isArray(current) && Array.isArray(current[point]) ? current[point][axis] : undefined}
                  onChange={(v) => {
                    const next = [0, 1].map((p) => [0, 1].map((a) =>
                      Array.isArray(current) && Array.isArray(current[p]) ? current[p][a] : undefined));
                    next[point][axis] = v;
                    set(key, next);
                  }} />
              </label>)}
            </div>)}
          </fieldset>;
        } else if (prop.type === 'array' && prop.items?.enum) {
          control = <ChipsField prop={prop} value={current} onChange={(v) => set(key, v)} />;
        } else if (prop.type === 'string' && prop.enum) {
          control = (
            <select id={id} className="select select--sm" value={String(current ?? '')} onChange={(e) => set(key, e.target.value)}>
              {prop.enum.map((opt, i) => (
                <option key={String(opt)} value={String(opt)}>
                  {prop.enumNames?.[i] ?? String(opt)}
                </option>
              ))}
            </select>
          );
        } else if (prop.type === 'string') {
          control = <input id={id} className="input input--sm" value={String(current ?? '')} onChange={(e) => set(key, e.target.value)} />;
        } else if (prop.type === 'object') {
          control = (
            <fieldset className="schema-form__nested">
              <SchemaForm
                schema={prop}
                value={current != null && typeof current === 'object' && !Array.isArray(current) ? current as Record<string, unknown> : {}}
                onChange={(v) => set(key, v)}
                idPrefix={id}
              />
            </fieldset>
          );
        } else {
          control = (
            <textarea
              id={id}
              className="textarea mono"
              value={isJsonDraft(current) ? current[JSON_DRAFT] : current === undefined ? '' : JSON.stringify(current)}
              aria-invalid={Boolean(error)}
              onChange={(e) => {
                try {
                  set(key, JSON.parse(e.target.value));
                } catch {
                  set(key, { [JSON_DRAFT]: e.target.value });
                }
              }}
            />
          );
        }
        return (
          <div key={key} className="field schema-form__field">
            <label className="field__label" htmlFor={id}>
              {label}
            </label>
            {control}
            {error && <span className="field__hint" role="alert">{error}</span>}
            {prop.description && <span className="field__hint">{prop.description}</span>}
          </div>
        );
      })}
    </div>
  );
}
