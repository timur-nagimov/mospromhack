import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { EmployeeInput } from '../domain/employees';
import { api } from './index';
import type { AnalyticsQuery } from './types';

export function useEmployees() {
  return useQuery({ queryKey: ['employees'], queryFn: () => api().employees.list(), staleTime: 15_000, refetchInterval: 30_000 });
}

export function useCreateEmployee() {
  const client = useQueryClient();
  return useMutation({ mutationFn: (body: EmployeeInput) => api().employees.create(body),
    onSuccess: () => client.invalidateQueries({ queryKey: ['employees'] }) });
}

export function useEventEmployees(eventId: string) {
  return useQuery({ queryKey: ['event-employees', eventId], queryFn: () => api().employees.forEvent(eventId), refetchInterval: 5_000 });
}

export function useAssignEmployees(eventId: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (ids: string[]) => api().employees.assign(eventId, ids), onSuccess: (employees) => {
    client.setQueryData(['event-employees', eventId], employees);
    void client.invalidateQueries({ queryKey: ['employee-report'] });
    void client.invalidateQueries({ queryKey: ['stats'] });
  } });
}

export function useEmployeeReport(employeeId: string, period: AnalyticsQuery) {
  return useQuery({ queryKey: ['employee-report', employeeId, period],
    queryFn: ({ signal }) => api().employees.report(employeeId, period, signal),
    enabled: Boolean(employeeId), staleTime: 10_000, refetchInterval: 15_000 });
}
