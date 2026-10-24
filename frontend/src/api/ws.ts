import type { FlywheelSampleBrief } from '../domain/flywheel';
import type { WsMessage } from '../domain/types';
import type { ConnectionStatus, Subscription } from './types';

const MIN_DELAY_MS = 1000;
const MAX_DELAY_MS = 15000;

/**
 * Проверка и нормализация сообщения. Сообщение маховика по контракту приходит
 * как {type: "flywheel_sample", sample}, остальные — с ключом kind; приводим к kind.
 * Неизвестные сообщения отбрасываются: подписчики их не видят.
 */
export function parseWsMessage(value: unknown): WsMessage | null {
  if (!value || typeof value !== 'object') return null;
  const msg = value as { kind?: unknown; type?: unknown; sample?: unknown };
  if (msg.kind === 'event' || msg.kind === 'event_update' || msg.kind === 'metrics') return value as WsMessage;
  if (msg.kind === 'flywheel_sample' || msg.type === 'flywheel_sample') {
    const sample = msg.sample as { id?: unknown } | null | undefined;
    if (!sample || typeof sample !== 'object' || typeof sample.id !== 'string') return null;
    return { kind: 'flywheel_sample', sample: sample as FlywheelSampleBrief };
  }
  return null;
}

/**
 * Одно WS-соединение на приложение с переподключением (экспоненциальная
 * задержка с джиттером) и рассылкой сообщений подписчикам.
 */
export class EventSocket {
  private socket: WebSocket | null = null;
  private subscribers = new Set<Subscription>();
  private status: ConnectionStatus = 'closed';
  private attempt = 0;
  private reconnectTimer: number | null = null;

  constructor(private readonly url: string) {}

  subscribe(subscription: Subscription): () => void {
    this.subscribers.add(subscription);
    subscription.onStatus?.(this.status);
    if (this.subscribers.size === 1) this.connect();
    return () => {
      this.subscribers.delete(subscription);
      if (this.subscribers.size === 0) this.disconnect();
    };
  }

  private setStatus(status: ConnectionStatus): void {
    this.status = status;
    for (const s of this.subscribers) s.onStatus?.(status);
  }

  private connect(): void {
    this.clearTimer();
    this.setStatus('connecting');
    let socket: WebSocket;
    try {
      socket = new WebSocket(this.url);
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;

    socket.onopen = () => {
      this.attempt = 0;
      this.setStatus('open');
    };
    socket.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      let parsed: unknown;
      try {
        parsed = JSON.parse(ev.data);
      } catch {
        return;
      }
      const message = parseWsMessage(parsed);
      if (!message) return;
      for (const s of this.subscribers) s.onMessage(message);
    };
    socket.onclose = () => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.setStatus('closed');
      if (this.subscribers.size > 0) this.scheduleReconnect();
    };
    socket.onerror = () => {
      // onclose придёт следом — переподключение там
    };
  }

  private scheduleReconnect(): void {
    const exp = Math.min(MAX_DELAY_MS, MIN_DELAY_MS * 2 ** this.attempt);
    const delay = exp / 2 + Math.random() * (exp / 2);
    this.attempt += 1;
    this.reconnectTimer = window.setTimeout(() => this.connect(), delay);
  }

  private clearTimer(): void {
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private disconnect(): void {
    this.clearTimer();
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onclose = null;
      socket.close();
    }
    this.setStatus('closed');
  }
}
