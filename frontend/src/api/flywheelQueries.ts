// TanStack Query для маховика данных (/api/flywheel/*). Ключи — под общим префиксом
// ['flywheel'], чтобы WS-сообщение flywheel_sample могло разом освежить счётчики и очередь.

import { keepPreviousData, useMutation, useQuery, useQueryClient, type QueryClient } from '@tanstack/react-query';
import type {
  DatasetBuildRequest,
  FlywheelJob,
  FlywheelSample,
  FlywheelSampleBrief,
  FlywheelSummary,
  JobKind,
  SamplePage,
  SamplePatch,
  SampleQuery,
} from '../domain/flywheel';
import { isHoldout, jobPhase } from '../domain/flywheel';
import { nowSec, startOfDay } from '../domain/format';
import { api } from './index';
import { ApiError } from './types';

export const flywheelKeys = {
  all: ['flywheel'] as const,
  summary: ['flywheel', 'summary'] as const,
  samplesAll: ['flywheel', 'samples'] as const,
  samples: (q: SampleQuery) => ['flywheel', 'samples', q] as const,
  sample: (id: string) => ['flywheel', 'sample', id] as const,
  byEvent: (eventId: string) => ['flywheel', 'by-event', eventId] as const,
  datasets: ['flywheel', 'datasets'] as const,
  models: ['flywheel', 'models'] as const,
  jobs: ['flywheel', 'jobs'] as const,
};

/** 404 — на сервере нет API маховика (бэкенд старее контракта) или нет такого сэмпла */
export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

/** 403 — задачи train и deploy запрещены без GUARD_FLYWHEEL_ALLOW_TRAIN=1 */
export function isForbidden(error: unknown): boolean {
  return error instanceof ApiError && error.status === 403;
}

/** Повторять запрос при сбоях сети, но не при 4xx: ответ не изменится */
const retryUnlessClientError = (count: number, error: unknown): boolean =>
  !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2;

export function useFlywheelSummary() {
  return useQuery({
    queryKey: flywheelKeys.summary,
    queryFn: () => api().flywheel.summary(),
    staleTime: 5_000,
    // запасной путь, если WS оборвался; без API маховика не опрашиваем
    refetchInterval: (query) => (isNotFound(query.state.error) ? false : 20_000),
    retry: retryUnlessClientError,
  });
}

export function useFlywheelSamples(query: SampleQuery) {
  return useQuery({
    queryKey: flywheelKeys.samples(query),
    queryFn: ({ signal }) => api().flywheel.listSamples(query, signal),
    placeholderData: keepPreviousData,
    staleTime: 10_000,
    retry: retryUnlessClientError,
  });
}

export function useFlywheelSample(id: string | null, initial?: FlywheelSample) {
  return useQuery({
    queryKey: flywheelKeys.sample(id ?? ''),
    queryFn: () => api().flywheel.getSample(id as string),
    enabled: Boolean(id),
    initialData: initial,
    staleTime: 5_000,
    retry: retryUnlessClientError,
  });
}

/**
 * Сэмпл маховика по событию: GET /api/flywheel/samples?event_id=…
 * Ответ сверяем по event_id: сервер без этого фильтра вернул бы чужие сэмплы.
 */
export function useSampleForEvent(eventId: string | null) {
  return useQuery({
    queryKey: flywheelKeys.byEvent(eventId ?? ''),
    queryFn: async ({ signal }) => {
      const page = await api().flywheel.listSamples({ event_id: eventId as string, limit: 5 }, signal);
      return page.items.find((s) => s.event_id === eventId) ?? null;
    },
    enabled: Boolean(eventId),
    retry: false,
    staleTime: 30_000,
  });
}

/** Обновить сэмпл во всех закэшированных страницах очереди */
function replaceInPages(qc: QueryClient, sample: FlywheelSample): void {
  qc.setQueryData(flywheelKeys.sample(sample.id), sample);
  qc.setQueriesData<SamplePage>({ queryKey: flywheelKeys.samplesAll }, (page) =>
    page ? { ...page, items: page.items.map((s) => (s.id === sample.id ? sample : s)) } : page,
  );
  if (sample.event_id) qc.setQueryData(flywheelKeys.byEvent(sample.event_id), sample);
}

export function usePatchSample() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, patch }: { id: string; patch: SamplePatch }) => api().flywheel.patchSample(id, patch),
    onSuccess: (sample) => {
      replaceInPages(qc, sample);
      void qc.invalidateQueries({ queryKey: flywheelKeys.summary });
      void qc.invalidateQueries({ queryKey: flywheelKeys.samplesAll });
    },
  });
}

export function useCaptureFrame() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (cameraId: string) => api().flywheel.capture(cameraId),
    onSuccess: (sample) => {
      qc.setQueryData(flywheelKeys.sample(sample.id), sample);
      void qc.invalidateQueries({ queryKey: flywheelKeys.summary });
      void qc.invalidateQueries({ queryKey: flywheelKeys.samplesAll });
    },
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: flywheelKeys.datasets,
    queryFn: () => api().flywheel.listDatasets(),
    staleTime: 15_000,
    retry: retryUnlessClientError,
  });
}

export function useBuildDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (request: DatasetBuildRequest) => api().flywheel.buildDataset(request),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: flywheelKeys.datasets });
      void qc.invalidateQueries({ queryKey: flywheelKeys.summary });
      // у сэмплов меняется dataset_version
      void qc.invalidateQueries({ queryKey: flywheelKeys.samplesAll });
    },
  });
}

export function useModelCards() {
  return useQuery({
    queryKey: flywheelKeys.models,
    queryFn: () => api().flywheel.listModels(),
    staleTime: 30_000,
    retry: retryUnlessClientError,
  });
}

export function useJobs() {
  return useQuery({
    queryKey: flywheelKeys.jobs,
    queryFn: () => api().flywheel.listJobs(),
    staleTime: 3_000,
    // пока задача идёт — опрос раз в 2 с, иначе редко
    refetchInterval: (query) => {
      if (isNotFound(query.state.error)) return false;
      const jobs = query.state.data as FlywheelJob[] | undefined;
      return jobs?.some((j) => jobPhase(j.status) === 'active') ? 2_000 : 30_000;
    },
    retry: retryUnlessClientError,
  });
}

export function useStartJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ kind, args }: { kind: JobKind; args?: Record<string, unknown> }) => api().flywheel.startJob(kind, args),
    onSuccess: (job) => {
      qc.setQueryData<FlywheelJob[]>(flywheelKeys.jobs, (list) => [job, ...(list ?? []).filter((j) => j.id !== job.id)]);
      void qc.invalidateQueries({ queryKey: flywheelKeys.jobs });
    },
  });
}

// ---------------------------------------------------------------- живые обновления

function bumpSummary(s: FlywheelSummary, sample: FlywheelSampleBrief): FlywheelSummary {
  const source = sample.source ?? 'unknown';
  const status = sample.status ?? 'new';
  const createdAt = sample.created_at ?? nowSec();
  return {
    ...s,
    total: (s.total ?? 0) + 1,
    today: (s.today ?? 0) + (createdAt >= startOfDay(nowSec()) ? 1 : 0),
    by_source: { ...s.by_source, [source]: (s.by_source?.[source] ?? 0) + 1 },
    by_status: { ...s.by_status, [status]: (s.by_status?.[status] ?? 0) + 1 },
    holdout: (s.holdout ?? 0) + (sample.holdout != null && isHoldout({ holdout: sample.holdout }) ? 1 : 0),
  };
}

let liveTimer: number | null = null;

/**
 * WS flywheel_sample: счётчики растут сразу, очередь и сводка сверяются с сервером
 * одной пачкой (серия сэмплов схлопывается в один запрос). Неактивные запросы
 * только помечаются устаревшими — на других экранах сети нет.
 */
export function applyLiveSample(qc: QueryClient, sample: FlywheelSampleBrief): void {
  qc.setQueryData<FlywheelSummary>(flywheelKeys.summary, (s) => (s ? bumpSummary(s, sample) : s));
  if (sample.event_id) void qc.invalidateQueries({ queryKey: flywheelKeys.byEvent(sample.event_id) });
  if (liveTimer != null) return;
  liveTimer = window.setTimeout(() => {
    liveTimer = null;
    void qc.invalidateQueries({ queryKey: flywheelKeys.samplesAll });
    void qc.invalidateQueries({ queryKey: flywheelKeys.summary });
  }, 1200);
}
