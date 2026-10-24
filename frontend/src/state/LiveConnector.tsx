import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';
import { api } from '../api';
import { applyLiveSample } from '../api/flywheelQueries';
import { queryKeys } from '../api/queries';
import { isCriticalEvent } from '../domain/violations';
import { playAlarm, playChime, unlockAudioOnGesture } from '../hooks/alarmSound';
import { liveActions, liveStore } from './live';
import { settingsStore } from './settings';

const RESYNC_MS = 30_000;
const METRICS_POLL_MS = 5_000;

/**
 * Единственная точка подключения к WS /ws/events: наполняет liveStore,
 * синхронизирует кэш запросов, играет звук для критических тревог.
 */
export function LiveConnector() {
  const qc = useQueryClient();
  const invalidateTimer = useRef<number | null>(null);
  const resyncTimer = useRef<number | null>(null);

  useEffect(() => unlockAudioOnGesture(), []);

  useEffect(() => {
    let cancelled = false;
    let wasDisconnected = false;
    let everOpened = false;

    const scheduleInvalidate = () => {
      if (invalidateTimer.current != null) return;
      invalidateTimer.current = window.setTimeout(() => {
        invalidateTimer.current = null;
        void qc.invalidateQueries({ queryKey: queryKeys.eventsAll });
        void qc.invalidateQueries({ queryKey: queryKeys.statsAll });
      }, 800);
    };

    const loadInitial = async () => {
      try {
        const [latest, unreviewed] = await Promise.all([
          api().listEvents({ limit: 60 }),
          api().listEvents({ status: 'new', limit: 100 }),
        ]);
        if (cancelled) return;
        liveActions.seedFeed([...latest.items, ...unreviewed.items]);
        liveActions.setUnreviewed(unreviewed.total);
      } catch {
        // сервер недоступен — лента наполнится из WS после переподключения
      }
    };

    const resyncUnreviewed = async () => {
      try {
        const page = await api().listEvents({ status: 'new', limit: 1 });
        if (!cancelled) liveActions.setUnreviewed(page.total);
      } catch {
        // повторим по таймеру
      }
    };

    // Сверка счётчика с сервером с небольшой задержкой (схлопывает серию обновлений)
    const scheduleResync = () => {
      if (resyncTimer.current != null) return;
      resyncTimer.current = window.setTimeout(() => {
        resyncTimer.current = null;
        void resyncUnreviewed();
      }, 1000);
    };

    void loadInitial();

    const unsubscribe = api().subscribe({
      onStatus: (status) => {
        liveActions.setConnection(status);
        if (status === 'closed' && everOpened) wasDisconnected = true;
        if (status === 'open') {
          everOpened = true;
          if (wasDisconnected) {
            // события, пришедшие во время обрыва, догружаем из журнала
            wasDisconnected = false;
            void loadInitial();
            scheduleInvalidate();
          } else {
            void resyncUnreviewed();
          }
        }
      },
      onMessage: (msg) => {
        if (msg.kind === 'metrics') {
          liveActions.setMetrics(msg.data);
          return;
        }
        if (msg.kind === 'flywheel_sample') {
          // новый сэмпл маховика: счётчики и очередь страницы «Данные»
          if (liveActions.addFlywheelSample(msg.sample)) applyLiveSample(qc, msg.sample);
          return;
        }
        if (msg.kind === 'event') {
          const isNew = liveActions.addEvent(msg.event);
          if (isNew && settingsStore.get().sound) {
            if (isCriticalEvent(msg.event)) playAlarm();
            else playChime();
          }
        } else if (!liveActions.upsertEvent(msg.event)) {
          scheduleResync();
        }
        qc.setQueryData(queryKeys.event(msg.event.id), msg.event);
        scheduleInvalidate();
      },
    });

    const resync = window.setInterval(() => void resyncUnreviewed(), RESYNC_MS);

    // Если WS недоступен, метрики берём опросом /api/metrics
    const metricsPoll = window.setInterval(async () => {
      if (liveStore.get().connection === 'open') return;
      try {
        const m = await api().getMetrics();
        if (!cancelled) liveActions.setMetrics(m);
      } catch {
        // бэкенд недоступен
      }
    }, METRICS_POLL_MS);

    return () => {
      cancelled = true;
      unsubscribe();
      window.clearInterval(resync);
      window.clearInterval(metricsPoll);
      if (invalidateTimer.current != null) window.clearTimeout(invalidateTimer.current);
      if (resyncTimer.current != null) window.clearTimeout(resyncTimer.current);
      invalidateTimer.current = null;
      resyncTimer.current = null;
    };
  }, [qc]);

  return null;
}
