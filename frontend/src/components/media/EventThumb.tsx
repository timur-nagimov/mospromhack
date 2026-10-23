import { ImageOff } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { api } from '../../api';
import type { GuardEvent } from '../../domain/types';

/** Превью кадра события; ссылка вычисляется, только когда превью попало в видимую область */
export function EventThumb({ event, width = 72 }: { event: GuardEvent; width?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [visible, setVisible] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true);
          io.disconnect();
        }
      },
      { rootMargin: '200px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <span ref={ref} className="thumb-slot" style={{ width }}>
      {visible && !failed ? (
        <img
          className="thumb"
          style={{ width }}
          src={api().media.snapshotUrl(event)}
          alt=""
          loading="lazy"
          decoding="async"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="thumb thumb--empty" style={{ width }}>
          {failed && <ImageOff size={14} aria-hidden="true" />}
        </span>
      )}
    </span>
  );
}
