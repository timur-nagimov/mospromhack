import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo } from 'react';
import type { CameraConfig, EventPatch, EventQuery, GuardEvent, StatsGroupBy } from '../domain/types';
import { registerPluginTitles } from '../domain/violations';
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

export function useCameras() {
  return useQuery({
    queryKey: queryKeys.cameras,
    queryFn: () => api().listCameras(),
    staleTime: 30_000,
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

export function useRuleTypes() {
  return useQuery({
    queryKey: queryKeys.ruleTypes,
    queryFn: async () => {
      const items = await api().listRuleTypes();
      registerPluginTitles(items);
      return items;
    },
    staleTime: Infinity,
  });
}

export function useEvents(query: EventQuery, options: { enabled?: boolean; refetchInterval?: number } = {}) {
  return useQuery({
    queryKey: queryKeys.events(query),
    queryFn: ({ signal }) => api().listEvents(query, signal),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    ...options,
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
    },
  });
}

export function useStats(groupBy: StatsGroupBy, from?: number, to?: number) {
  return useQuery({
    queryKey: queryKeys.stats(groupBy, from, to),
    queryFn: () => api().getStats({ group_by: groupBy, from, to }),
    placeholderData: keepPreviousData,
    staleTime: 30_000,
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
