import { Download } from 'lucide-react';
import { useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api } from '../../api';
import { useEmployeeReport, useEmployees } from '../../api/employeeQueries';
import type { AnalyticsQuery } from '../../api/types';
import { StatusBadge } from '../../components/ui/badges';
import { toast } from '../../components/ui/toast';
import { formatDateTime } from '../../domain/format';
import { getViolationMeta } from '../../domain/violations';
import { useEventModal } from '../event/useEventModal';
import { downloadText } from '../journal/csv';
import './employees.css';

export function EmployeeReportPanel({ period }: { period: AnalyticsQuery }) {
  const employees = useEmployees();
  const [params, setParams] = useSearchParams();
  const employeeId = params.get('employee') ?? '';
  const report = useEmployeeReport(employeeId, period);
  const [exporting, setExporting] = useState(false);
  const { openEvent } = useEventModal();
  const download = async () => {
    if (!employeeId || !report.data || exporting) return;
    const id = employeeId;
    const name = report.data.employee.full_name.replace(/[^\p{L}\p{N}_-]+/gu, '_');
    setExporting(true);
    try {
      const csv = await api().employees.exportReport(id, period);
      downloadText(`sotrudnik-${name}-${id.slice(0, 8)}.csv`, '\ufeff' + csv.replace(/^\ufeff/, ''));
      toast('Полная выгрузка по сотруднику готова');
    } catch (error) { toast(`Не удалось выгрузить: ${(error as Error).message}`, 'bad'); }
    finally { setExporting(false); }
  };
  return (
    <section className="panel employee-report" aria-labelledby="employee-report-title">
      <header className="panel__head"><div><h2 className="panel__title" id="employee-report-title">Статистика по сотруднику</h2>
        <p className="panel__sub">Только подтверждённые нарушения выбранного сотрудника · время МСК</p></div>
        <button type="button" className="btn btn--sm" onClick={() => void download()} disabled={!report.data || report.isFetching || exporting}>
          <Download size={15} />{exporting ? 'Выгружаем…' : 'Выгрузить CSV по сотруднику'}
        </button></header>
      <div className="panel__body">
        <label className="field"><span className="field__label">Сотрудник для отчёта</span>
          <select className="select" aria-label="Сотрудник для отчёта" value={employeeId} onChange={(event) => {
            const id = event.target.value;
            setParams((previous) => { const next = new URLSearchParams(previous); if (id) next.set('employee', id); else next.delete('employee'); return next; }, { replace: true });
          }}><option value="">Выберите сотрудника</option>
            {(employees.data ?? []).map((employee) => <option value={employee.id} key={employee.id}>{employee.full_name}{employee.personnel_number ? ` · ${employee.personnel_number}` : ''}{employee.department ? ` · ${employee.department}` : ''}</option>)}
          </select></label>
        {employees.isError && <p className="field__error" role="alert">Не удалось загрузить справочник: {employees.error.message}</p>}
        {!employeeId ? <p className="muted">Выберите сотрудника, чтобы отфильтровать всю аналитику по его подтверждённым нарушениям.</p>
          : report.isError ? <p className="field__error" role="alert">Не удалось загрузить отчёт: {report.error.message}</p>
          : !report.data ? <p className="muted" role="status">Загружаем отчёт…</p>
          : <>
            <div className="employee-report__summary" aria-label="Итоги по сотруднику">
              <span>Подтверждённых нарушений: <b>{report.data.total}</b></span>
            </div>
            <p className="field__hint">Список, графики и CSV учитывают только подтверждённые нарушения этого сотрудника за выбранный период.</p>
            {!report.data.items.length ? <p className="muted">За этот период подтверждённых нарушений нет.</p>
              : <div className="table-wrap employee-report__table"><table className="table" aria-label="Инциденты сотрудника"><thead><tr>
                <th>Оповещение (МСК)</th><th>Тип нарушения</th><th>Участок</th><th>Статус</th><th>Событие</th>
              </tr></thead><tbody>{report.data.items.map((event) => <tr key={event.id}>
                <td>{formatDateTime(event.confirmed_at)}</td><td>{getViolationMeta(event.type).title}</td><td>{event.site}</td><td><StatusBadge status={event.status} /></td>
                <td><button type="button" className="btn btn--sm btn--ghost" onClick={() => openEvent(event.id)}>Открыть</button></td>
              </tr>)}</tbody></table></div>}
          </>}
      </div>
    </section>
  );
}
