import { useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';

/** Карточка события открыта, если в адресе есть ?event=<id> — ссылку можно переслать */
export function useEventModal() {
  const [params, setParams] = useSearchParams();
  const eventId = params.get('event');

  const openEvent = useCallback(
    (id: string) => {
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set('event', id);
          return next;
        },
        { replace: Boolean(eventId) },
      );
    },
    [setParams, eventId],
  );

  const closeEvent = useCallback(() => {
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete('event');
        return next;
      },
      { replace: true },
    );
  }, [setParams]);

  return { eventId, openEvent, closeEvent };
}
