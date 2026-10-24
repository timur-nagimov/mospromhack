import { useState } from 'react';
import { useAssignEmployees, useEmployees, useEventEmployees } from '../../api/employeeQueries';
import { toast } from '../../components/ui/toast';
import { useRole } from '../../state/role';
import './employees.css';

export function IncidentEmployees({ eventId }: { eventId: string }) {
  const role = useRole();
  const canAssign = role === 'operator' || role === 'admin';
  const employees = useEmployees();
  const assigned = useEventEmployees(eventId);
  const save = useAssignEmployees(eventId);
  const [draft, setDraft] = useState<string[] | null>(null);
  const [search, setSearch] = useState('');
  const selected = draft ?? assigned.data?.map((employee) => employee.id) ?? [];
  const changed = draft !== null && (draft.length !== assigned.data?.length || draft.some((id) => !assigned.data?.some((employee) => employee.id === id)));
  const needle = search.trim().toLocaleLowerCase('ru');
  const shown = (employees.data ?? []).filter((employee) =>
    [employee.full_name, employee.personnel_number, employee.department].join(' ').toLocaleLowerCase('ru').includes(needle));
  return (
    <section className="employee-assignment" aria-labelledby="culprits-title">
      <h3 className="panel__title" id="culprits-title">Виновники инцидента</h3>
      {assigned.isError || employees.isError ? <div className="field__error" role="alert">Не удалось загрузить сотрудников.
        <button className="btn btn--sm" onClick={() => { void assigned.refetch(); void employees.refetch(); }}>Повторить</button></div>
        : assigned.isPending || employees.isPending ? <p className="muted" role="status">Загрузка…</p>
        : <>
          <div className="employee-chips" aria-label="Указанные виновники">
            {(employees.data ?? []).filter((employee) => selected.includes(employee.id)).map((employee) =>
              <span className="badge badge--plain" key={employee.id}>{employee.full_name}{employee.personnel_number ? ` · ${employee.personnel_number}` : ''}</span>)}
            {!selected.length && <span className="muted">Не указаны</span>}
          </div>
          {canAssign && <>
            <label className="field"><span className="field__label">Выберите одного или нескольких сотрудников</span>
              <input className="input input--sm" aria-label="Поиск виновников" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="ФИО или табельный номер" /></label>
            <div className="employee-choices" role="group" aria-label="Сотрудники для разбора инцидента">
              {shown.map((employee) => <label className="employee-choice" key={employee.id}>
                <input type="checkbox" checked={selected.includes(employee.id)} disabled={save.isPending}
                  onChange={(event) => setDraft(event.target.checked ? [...selected, employee.id] : selected.filter((id) => id !== employee.id))} />
                <span>{employee.full_name}<small>{[employee.personnel_number, employee.department].filter(Boolean).join(' · ')}</small></span>
              </label>)}
              {!shown.length && <p className="muted">{employees.data?.length ? 'Совпадений нет' : 'Сначала администратор должен добавить сотрудников в справочник'}</p>}
            </div>
            <div className="employee-assignment__actions">
              <span className="muted">Выбрано: {selected.length}</span>
              <button type="button" className="btn btn--sm" disabled={!changed || save.isPending || assigned.isPending}
                onClick={() => save.mutate(selected, { onSuccess: () => { setDraft(null); toast('Список виновников сохранён'); } })}>
                {save.isPending ? 'Сохраняем…' : 'Сохранить виновников'}
              </button>
            </div>
            {save.isError && <p className="field__error" role="alert">{save.error.message}</p>}
          </>}
        </>}
    </section>
  );
}
