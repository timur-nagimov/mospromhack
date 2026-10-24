import { Plus, UsersRound } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { useCreateEmployee, useEmployees } from '../../api/employeeQueries';
import { toast } from '../../components/ui/toast';
import { useRole } from '../../state/role';
import './employees.css';

export function EmployeesPage() {
  const role = useRole();
  const employees = useEmployees();
  const create = useCreateEmployee();
  const [search, setSearch] = useState('');
  const [name, setName] = useState('');
  const [number, setNumber] = useState('');
  const [department, setDepartment] = useState('');
  const [position, setPosition] = useState('');
  const needle = search.trim().toLocaleLowerCase('ru');
  const shown = (employees.data ?? []).filter((employee) =>
    [employee.full_name, employee.personnel_number, employee.department, employee.position].join(' ').toLocaleLowerCase('ru').includes(needle));
  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (role !== 'admin' || !name.trim() || create.isPending) return;
    create.mutate({ full_name: name, personnel_number: number || null, department, position }, { onSuccess: () => {
      setName(''); setNumber(''); setDepartment(''); setPosition(''); toast('Сотрудник добавлен');
    } });
  };
  return (
    <div className="page employees-page">
      {role === 'admin' && <section className="panel" aria-labelledby="add-employee-title">
        <div className="panel__head"><h2 className="panel__title" id="add-employee-title">Добавить сотрудника</h2></div>
        <form className="panel__body employee-form" onSubmit={submit}>
          <label className="field"><span className="field__label">ФИО *</span>
            <input className="input" aria-label="ФИО сотрудника" value={name} onChange={(event) => setName(event.target.value)} required maxLength={200} /></label>
          <label className="field"><span className="field__label">Табельный номер</span>
            <input className="input" aria-label="Табельный номер" value={number} onChange={(event) => setNumber(event.target.value)} maxLength={64} /></label>
          <label className="field"><span className="field__label">Подразделение</span>
            <input className="input" aria-label="Подразделение сотрудника" value={department} onChange={(event) => setDepartment(event.target.value)} maxLength={200} /></label>
          <label className="field"><span className="field__label">Должность</span>
            <input className="input" aria-label="Должность сотрудника" value={position} onChange={(event) => setPosition(event.target.value)} maxLength={200} /></label>
          <button className="btn btn--primary" type="submit" disabled={!name.trim() || create.isPending}><Plus size={16} />{create.isPending ? 'Добавляем…' : 'Добавить сотрудника'}</button>
          {create.isError && <p className="field__error employee-form__error" role="alert">{create.error.message}</p>}
        </form>
      </section>}
      <section className="panel" aria-labelledby="employees-title">
        <div className="panel__head"><div><h2 className="panel__title" id="employees-title">Все сотрудники</h2>
          <p className="panel__sub">Всего: {employees.data?.length ?? '…'} · в том числе без инцидентов</p></div>
          <UsersRound size={20} className="muted" /></div>
        <div className="panel__body">
          <label className="field"><span className="field__label">Поиск сотрудника</span>
            <input className="input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="ФИО, табельный номер, подразделение или должность" /></label>
        </div>
        {employees.isError ? <div className="panel__body error-box" role="alert">Не удалось загрузить сотрудников: {employees.error.message}
          <button className="btn btn--sm" onClick={() => void employees.refetch()}>Повторить</button></div>
          : employees.isPending ? <div className="empty" role="status">Загружаем справочник…</div>
          : !shown.length ? <div className="empty">{search ? 'Сотрудники не найдены' : role === 'admin' ? 'Добавьте первого сотрудника с помощью формы выше' : 'Администратор ещё не добавил сотрудников'}</div>
          : <div className="table-wrap"><table className="table" aria-label="Список сотрудников"><thead><tr>
            <th>ФИО</th><th>Табельный номер</th><th>Подразделение</th><th>Должность</th><th>Отчёт</th>
          </tr></thead><tbody>{shown.map((employee) => <tr key={employee.id}>
            <td>{employee.full_name}</td><td>{employee.personnel_number || '—'}</td><td>{employee.department || '—'}</td><td>{employee.position || '—'}</td>
            <td><Link className="btn btn--sm btn--ghost" to={`/analytics?employee=${encodeURIComponent(employee.id)}`}>Статистика</Link></td>
          </tr>)}</tbody></table></div>}
      </section>
    </div>
  );
}
