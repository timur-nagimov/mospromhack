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
import type { AuditPage, AuditQuery, AuthConfig, Me, RolesCatalog, UserCreate, UserPatch, UserRecord } from '../domain/auth';

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

/** Вход и сессия: /api/auth/* (docs/ROLES.md, раздел «API») */
export interface AuthApi {
  /** Без входа; бэкенд без /api/auth отвечает 404 — значит, вход выключен */
  config(): Promise<AuthConfig>;
  /** Неверный логин или пароль — 401 без уточнения */
  login(username: string, password: string): Promise<Me>;
  logout(): Promise<void>;
  /** Без сессии — 401 */
  me(): Promise<Me>;
  roles(): Promise<RolesCatalog>;
}

/** Учётные записи: /api/users* (право users.manage) */
export interface UsersApi {
  list(): Promise<UserRecord[]>;
  create(user: UserCreate): Promise<UserRecord>;
  /** Последнего активного администратора нельзя понизить или отключить — 409 */
  update(username: string, patch: UserPatch): Promise<UserRecord>;
}

/** Журнал действий: GET /api/audit (право audit.view) */
export interface AuditApi {
  list(query: AuditQuery, signal?: AbortSignal): Promise<AuditPage>;
}

/** Единый интерфейс: HTTP-реализация и мок взаимозаменяемы */
export interface GuardApi {
  readonly mode: 'http' | 'mock';
  health(): Promise<Health>;
  listCameras(): Promise<CameraConfig[]>;
  createCamera(camera: CameraConfig): Promise<CameraConfig>;
  updateCamera(id: string, camera: CameraConfig): Promise<CameraConfig>;
  deleteCamera(id: string): Promise<void>;
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
  auth: AuthApi;
  users: UsersApi;
  audit: AuditApi;
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
    /** 403 проверки прав: недостающее право из ответа {detail: "forbidden", permission} */
    public readonly permission?: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}
