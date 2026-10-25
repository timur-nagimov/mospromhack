import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import type { CameraConfig, EventPatch, EventQuery, GuardEvent, StatsGroupBy } from '../domain/types';
import { registerPluginTitles } from '../domain/violations';
import { usePermission } from '../state/auth';
import { liveActions } from '../state/live';
import { api } from './index';

export const queryKeys = {
  cameras: ['cameras'] as const,
  ruleTypes: ['rule-types'] as const,
  events: (q: EventQuery) => ['events', q] as const,
  eventsAll: ['events'] as const,
  event: (id: string) => ['event', id] as const,
  stats: (groupBy: StatsGroupBy, from?: number, to?: number) => ['stats', groupBy, from, to] as const,
  statsAll: ['stats'] as const,
  metrics: ['metrics'] as const,
};

/** GET /api/cameras — право cameras.view; без него список пуст, запрос не уходит */
export function useCameras() {
  const allowed = usePermission('cameras.view');
  return useQuery({
    queryKey: queryKeys.cameras,
    queryFn: () => api().listCameras(),
    staleTime: 30_000,
    enabled: allowed,
  });
}

/** Справочник камер по id: имя, участок, зоны */
export function useCameraMap(): Map<string, CameraConfig> {
  const { data } = useCameras();
  return useMemo(() => new Map((data ?? []).map((c) => [c.camera_id, c])), [data]);
}

export function zoneName(cameras: Map<string, CameraConfig>, cameraId: string, zoneId: string | null): string | null {
  if (!zoneId) return null;
  const zone = cameras.get(cameraId)?.zones.find((z) => z.zone_id === zoneId);
  return zone?.name ?? zoneId;
}

/** GET /api/rules/types — тоже cameras.view */
export function useRuleTypes() {
  const allowed = usePermission('cameras.view');
  return useQuery({
    queryKey: queryKeys.ruleTypes,
    queryFn: async () => {
      const items = await api().listRuleTypes();
      registerPluginTitles(items);
      return items;
    },
    staleTime: Infinity,
    enabled: allowed,
  });
}

export function useEvents(query: EventQuery, options: { enabled?: boolean; refetchInterval?: number } = {}) {
  const allowed = usePermission('events.view');
  return useQuery({
    queryKey: queryKeys.events(query),
    queryFn: ({ signal }) => api().listEvents(query, signal),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    ...options,
    enabled: allowed && options.enabled !== false,
  });
}

export function useEvent(id: string | null, initial?: GuardEvent) {
  return useQuery({
    queryKey: queryKeys.event(id ?? ''),
    queryFn: () => api().getEvent(id as string),
    enabled: Boolean(id),
    initialData: initial,
    staleTime: 5_000,
  });
}

export function usePatchEvent() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: EventPatch }) => api().patchEvent(id, patch),
    onSuccess: (event) => {
      qc.setQueryData(queryKeys.event(event.id), event);
      liveActions.upsertEvent(event);
      void qc.invalidateQueries({ queryKey: queryKeys.eventsAll });
      void qc.invalidateQueries({ queryKey: ['employee-report'] });
      void qc.invalidateQueries({ queryKey: queryKeys.statsAll });
      // событие могло быть не в ленте (открыто из журнала) — сверяем счётчик с сервером
      api()
        .listEvents({ status: 'new', limit: 1 })
        .then((page) => liveActions.setUnreviewed(page.total))
        .catch(() => undefined);
    },
  });
}

export function useStats(groupBy: StatsGroupBy, from?: number, to?: number) {
  const allowed = usePermission('analytics.view');
  return useQuery({
    queryKey: queryKeys.stats(groupBy, from, to),
    queryFn: () => api().getStats({ group_by: groupBy, from, to }),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
    enabled: allowed,
  });
}

/** Один запрос — один снимок SQLite для всех показателей выбранного периода. */
export function useAnalyticsStats(from: number | undefined, to: number, employeeId?: string) {
  const allowed = usePermission('analytics.view');
  return useQuery({
    queryKey: ['stats', 'analytics', from, to, employeeId],
    queryFn: ({ signal }) => api().getAnalytics({ from, to, employee_id: employeeId }, signal),
    staleTime: 30_000,
    enabled: allowed,
  });
}

export function useSaveCamera() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ camera, isNew }: { camera: CameraConfig; isNew: boolean }) =>
      isNew ? api().createCamera(camera) : api().updateCamera(camera.camera_id, camera),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.cameras }),
  });
}

export function useDeleteCamera() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api().deleteCamera(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.cameras }),
  });
}
