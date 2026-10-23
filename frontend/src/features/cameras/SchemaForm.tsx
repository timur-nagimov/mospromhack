import type { JsonSchema } from '../../domain/types';

// Генератор формы по JSON Schema из /api/rules/types: правила настраиваются без кода (D1, R7)

interface Props {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (next: Record<string, unknown>) => void;
  idPrefix: string;
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
  const num = typeof value === 'number' ? value : undefined;
  const hasRange = prop.minimum != null && prop.maximum != null;
  const invalid =
    num != null && ((prop.minimum != null && num < prop.minimum) || (prop.maximum != null && num > prop.maximum));
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
        const current = value[key] ?? prop.default;
        const label = prop.title ?? key;
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
          control = <NumberField id={id} prop={prop} value={current} onChange={(v) => set(key, v)} />;
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
                value={(current as Record<string, unknown>) ?? {}}
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
              defaultValue={JSON.stringify(current ?? null)}
              onBlur={(e) => {
                try {
                  set(key, JSON.parse(e.target.value));
                } catch {
                  // некорректный JSON — оставляем прежнее значение
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
            {prop.description && <span className="field__hint">{prop.description}</span>}
          </div>
        );
      })}
    </div>
  );
}
