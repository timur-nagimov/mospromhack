import type {
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

export type ConnectionStatus = 'connecting' | 'open' | 'closed';

export interface StatsQuery {
  group_by: StatsGroupBy;
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
  getMetrics(): Promise<MetricsSnapshot>;
  /** Подписка на WS /ws/events; возвращает функцию отписки */
  subscribe(subscription: Subscription): () => void;
  media: MediaUrls;
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
