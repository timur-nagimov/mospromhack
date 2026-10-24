import { employeeReportCsv, type Employee, type EmployeeInput, type EmployeeReport } from '../../domain/employees';
import demoEmployees from '../../domain/demoEmployees.json';
import type { GuardEvent } from '../../domain/types';
import { getRole } from '../../state/role';
import { ApiError, type AnalyticsQuery, type EmployeesApi } from '../types';

export function createEmployeesMock(events: () => GuardEvent[]): EmployeesApi & { forAnalytics(id: string): GuardEvent[] } {
  const employees: Employee[] = demoEmployees.map((employee) => ({
    ...employee, id: `demo-employee-${employee.personnel_number}`, created_at: Date.now() / 1000,
  }));
  const assignments = new Map<string, string[]>();
  const list = () => structuredClone([...employees].sort((a, b) => a.full_name.localeCompare(b.full_name, 'ru')));
  const forEvent = (id: string) => {
    if (!events().some((event) => event.id === id)) throw new ApiError(404, 'Событие не найдено');
    return list().filter((employee) => assignments.get(id)?.includes(employee.id));
  };
  const report = (id: string, range: AnalyticsQuery): EmployeeReport => {
    const employee = employees.find((item) => item.id === id);
    if (!employee) throw new ApiError(404, 'Сотрудник не найден');
    const items = events().filter((event) => event.status === 'confirmed' && assignments.get(event.id)?.includes(id)
      && (range.from == null || event.confirmed_at >= range.from) && (range.to == null || event.confirmed_at <= range.to))
      .sort((a, b) => b.confirmed_at - a.confirmed_at || b.id.localeCompare(a.id));
    const statuses = { new: 0, confirmed: 0, forwarded: 0, false: 0 };
    const types = new Map<string, number>();
    for (const event of items) { statuses[event.status] += 1; types.set(event.type, (types.get(event.type) ?? 0) + 1); }
    return structuredClone({ employee, from: range.from ?? null, to: range.to ?? null, total: items.length, statuses,
      types: [...types].map(([key, count]) => ({ key, count })), items });
  };
  return {
    forAnalytics: (id) => {
      if (!employees.some((employee) => employee.id === id)) throw new ApiError(404, 'Сотрудник не найден');
      return events().filter((event) => event.status === 'confirmed' && assignments.get(event.id)?.includes(id));
    },
    list: async () => list(),
    create: async (body: EmployeeInput) => {
      if (getRole() !== 'admin') throw new ApiError(403, 'Добавлять сотрудников может только администратор');
      const name = body.full_name.trim();
      const number = body.personnel_number?.trim() || null;
      if (!name) throw new ApiError(422, 'Укажите ФИО сотрудника');
      if (number && employees.some((employee) => employee.personnel_number === number)) throw new ApiError(409, 'Сотрудник с таким табельным номером уже существует');
      const employee: Employee = { id: crypto.randomUUID().replaceAll('-', ''), full_name: name, personnel_number: number,
        department: body.department?.trim() ?? '', position: body.position?.trim() ?? '', created_at: Date.now() / 1000 };
      employees.push(employee);
      return structuredClone(employee);
    },
    forEvent: async (id) => forEvent(id),
    assign: async (id, ids) => {
      if (!['operator', 'admin'].includes(getRole())) throw new ApiError(403, 'Указывать виновников может только диспетчер смены или администратор');
      forEvent(id);
      if (new Set(ids).size !== ids.length || ids.some((key) => !employees.some((employee) => employee.id === key))) throw new ApiError(422, 'Проверьте список сотрудников');
      assignments.set(id, [...ids]);
      return forEvent(id);
    },
    report: async (id, range) => report(id, range),
    exportReport: async (id, range) => employeeReportCsv(report(id, range)),
  };
}
