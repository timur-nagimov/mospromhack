import { formatDateTime } from './format';
import { SEVERITY_LABELS, STATUS_LABELS } from './labels';
import type { EventStatus, GuardEvent, StatsItem } from './types';
import { getViolationMeta } from './violations';

export interface EmployeeInput {
  full_name: string;
  personnel_number?: string | null;
  department?: string;
  position?: string;
}

export interface Employee extends EmployeeInput {
  id: string;
  personnel_number: string | null;
  department: string;
  position: string;
  created_at: number;
}

export type EmployeeIncident = Pick<GuardEvent, 'id' | 'confirmed_at' | 'started_at' | 'ended_at' | 'type' | 'severity' | 'status' | 'camera_id' | 'site' | 'zone_id' | 'operator_note'>;

export interface EmployeeReport {
  employee: Employee;
  from: number | null;
  to: number | null;
  total: number;
  statuses: Record<EventStatus, number>;
  types: StatsItem[];
  items: EmployeeIncident[];
}

/** Та же полная CSV-выгрузка для автономного мок-режима. */
export function employeeReportCsv(report: EmployeeReport): string {
  const cell = (value: unknown) => {
    let text = value == null ? '' : String(value);
    if (/^\s*[=+\-@]|^[\t\r\n]/.test(text)) text = `'${text}`;
    return /[";\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const employee = report.employee;
  const rows: unknown[][] = [['ID сотрудника', 'ФИО', 'Табельный номер', 'Подразделение', 'Должность', 'ID события',
    'Оповещение (МСК)', 'Тип (код)', 'Тип нарушения', 'Статус', 'Важность', 'Камера (ID)', 'Участок', 'Зона', 'Комментарий оператора']];
  for (const event of report.items) rows.push([employee.id, employee.full_name, employee.personnel_number, employee.department,
    employee.position, event.id, formatDateTime(event.confirmed_at), event.type, getViolationMeta(event.type).title,
    STATUS_LABELS[event.status], SEVERITY_LABELS[event.severity], event.camera_id, event.site, event.zone_id, event.operator_note]);
  return '\ufeff' + rows.map((row) => row.map(cell).join(';')).join('\r\n');
}
