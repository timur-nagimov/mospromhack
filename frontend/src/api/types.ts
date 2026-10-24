import type {
  AnalyticsResponse,
  CameraConfig,
  EventPage,
  EventPatch,
  EventQuery,
  GuardEvent,
  Health,
  MetricsSnapshot,
  RuleTypeInfo,
  SourceConfig,
  SourceProbeResult,
  StatsGroupBy,
  StatsResponse,
  WsMessage,
} from '../domain/types';
import type {
  DatasetBuildRequest,
  DatasetVersion,
  FlywheelJob,
  FlywheelSample,
  FlywheelSummary,
  JobKind,
  ModelCard,
  SamplePage,
  SamplePatch,
  SampleQuery,
} from '../domain/flywheel';
import type { Employee, EmployeeInput, EmployeeReport } from '../domain/employees';

export type ConnectionStatus = 'connecting' | 'open' | 'closed';

export interface StatsQuery {
  group_by: StatsGroupBy;
  type?: string;
  camera_id?: string;
  zone_id?: string;
  status?: string;
  severity?: string;
  from?: number;
  to?: number;
}

export interface AnalyticsQuery {
  from?: number;
  to?: number;
  employee_id?: string;
}

export interface EmployeesApi {
  list(): Promise<Employee[]>;
  create(body: EmployeeInput): Promise<Employee>;
  forEvent(eventId: string): Promise<Employee[]>;
  assign(eventId: string, employeeIds: string[]): Promise<Employee[]>;
  report(employeeId: string, period: AnalyticsQuery, signal?: AbortSignal): Promise<EmployeeReport>;
  exportReport(employeeId: string, period: AnalyticsQuery): Promise<string>;
}

/** Ссылки на медиа. В мок-режиме кадры рисуются на клиенте (data: URL) */
export interface MediaUrls {
  streamUrl(cameraId: string): string;
  frameUrl(camera: CameraConfig): string;
  snapshotUrl(event: GuardEvent): string;
  /** null — клипа нет (мок-режим показывает симуляцию) */
  clipUrl(event: GuardEvent): string | null;
}

export interface Subscription {
  onMessage(message: WsMessage): void;
  onStatus?(status: ConnectionStatus): void;
}

/** Маховик данных: /api/flywheel/* (docs/FLYWHEEL.md, раздел «API») */
export interface FlywheelApi {
  summary(): Promise<FlywheelSummary>;
  listSamples(query: SampleQuery, signal?: AbortSignal): Promise<SamplePage>;
  getSample(id: string): Promise<FlywheelSample>;
  patchSample(id: string, patch: SamplePatch): Promise<FlywheelSample>;
  /** Текущий кадр камеры в выборку (source = operator) */
  capture(cameraId: string): Promise<FlywheelSample>;
  /** Экспорт синхронный: ответ — описание новой версии */
  buildDataset(request: DatasetBuildRequest): Promise<DatasetVersion>;
  listDatasets(): Promise<DatasetVersion[]>;
  /** Карточки ml/model_cards/, новые сверху */
  listModels(): Promise<ModelCard[]>;
  /** train и deploy без GUARD_FLYWHEEL_ALLOW_TRAIN=1 отвечают 403 */
  startJob(kind: JobKind, args?: Record<string, unknown>): Promise<FlywheelJob>;
  listJobs(): Promise<FlywheelJob[]>;
  getJob(id: string): Promise<FlywheelJob>;
  /** Сырой кадр сэмпла; thumb — подсказка мок-режиму рисовать кадр меньше */
  frameUrl(sample: FlywheelSample, size?: 'thumb' | 'full'): string;
}

/** Единый интерфейс: HTTP-реализация и мок взаимозаменяемы */
export interface GuardApi {
  readonly mode: 'http' | 'mock';
  health(): Promise<Health>;
  listCameras(): Promise<CameraConfig[]>;
  createCamera(camera: CameraConfig): Promise<CameraConfig>;
  updateCamera(id: string, camera: CameraConfig): Promise<CameraConfig>;
  deleteCamera(id: string): Promise<void>;
  /** POST /api/cameras/probe: открыть поток и дождаться кадра (docs/STREAMS.md) */
  probeSource(source: SourceConfig, timeoutS?: number): Promise<SourceProbeResult>;
  listRuleTypes(): Promise<RuleTypeInfo[]>;
  listEvents(query: EventQuery, signal?: AbortSignal): Promise<EventPage>;
  getEvent(id: string): Promise<GuardEvent>;
  patchEvent(id: string, patch: EventPatch): Promise<GuardEvent>;
  getStats(query: StatsQuery): Promise<StatsResponse>;
  getAnalytics(query: AnalyticsQuery, signal?: AbortSignal): Promise<AnalyticsResponse>;
  getMetrics(): Promise<MetricsSnapshot>;
  /** Подписка на WS /ws/events; возвращает функцию отписки */
  subscribe(subscription: Subscription): () => void;
  media: MediaUrls;
  flywheel: FlywheelApi;
  employees: EmployeesApi;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
