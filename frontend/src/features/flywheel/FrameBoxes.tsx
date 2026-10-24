import { ImageOff } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { classColor, classLabel } from '../../domain/flywheel';
import type { BBox } from '../../domain/types';

export interface FrameBox {
  /** l0, l1 … — метки; d0, d1 … — детекции */
  key: string;
  cls: string;
  bbox: BBox;
  kind: 'label' | 'detection';
  conf?: number | null;
}

/** Рамка в пределах кадра; перевёрнутые координаты упорядочиваем */
function clampBox(b: BBox, w: number, h: number): BBox {
  const x1 = Math.max(0, Math.min(b[0], b[2]));
  const y1 = Math.max(0, Math.min(b[1], b[3]));
  const x2 = Math.min(w, Math.max(b[0], b[2]));
  const y2 = Math.min(h, Math.max(b[1], b[3]));
  return [x1, y1, x2, y2];
}

/**
 * Кадр и рамки в одной системе координат: viewBox — пиксели сохранённого кадра,
 * поэтому рамки из API (bbox в пикселях кадра) масштабируются вместе с картинкой
 * при любом размере контейнера. Толщина линий не зависит от масштаба.
 */
export function FrameBoxes({
  src,
  width,
  height,
  boxes,
  highlight = null,
  captions = true,
  onHover,
  alt,
}: {
  src: string | null;
  width: number;
  height: number;
  boxes: FrameBox[];
  highlight?: string | null;
  captions?: boolean;
  onHover?: (key: string | null) => void;
  alt: string;
}) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [src]);
  const w = width > 0 ? width : 960;
  const h = height > 0 ? height : 540;
  const font = Math.max(w, h) * 0.0165;

  return (
    <svg className="fw-frame" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label={alt}>
      <rect width={w} height={h} fill="#050607" />
      {src && !failed && <image href={src} width={w} height={h} preserveAspectRatio="none" onError={() => setFailed(true)} />}
      {(failed || !src) && (
        <foreignObject width={w} height={h}>
          <div className="fw-frame__missing">
            {failed && <ImageOff size={22} aria-hidden="true" />}
            {failed ? 'Кадр недоступен' : ''}
          </div>
        </foreignObject>
      )}
      {boxes.map((b) => {
        const [x1, y1, x2, y2] = clampBox(b.bbox, w, h);
        const color = classColor(b.cls);
        const active = highlight === b.key;
        const text = `${classLabel(b.cls)}${b.conf != null ? ` ${b.conf.toFixed(2)}` : ''}`;
        const tw = text.length * font * 0.62 + font * 0.8;
        const th = font * 1.45;
        const above = y1 - th >= 0;
        return (
          <g
            key={b.key}
            className={`fw-box${active ? ' is-active' : ''}${b.kind === 'detection' ? ' fw-box--det' : ''}`}
            onMouseEnter={onHover ? () => onHover(b.key) : undefined}
            onMouseLeave={onHover ? () => onHover(null) : undefined}
          >
            <rect
              x={x1}
              y={y1}
              width={Math.max(1, x2 - x1)}
              height={Math.max(1, y2 - y1)}
              fill={active ? color : 'transparent'}
              fillOpacity={active ? 0.16 : 0}
              stroke={color}
              strokeWidth={active ? 3.5 : 2}
              strokeDasharray={b.kind === 'detection' ? '7 5' : undefined}
              vectorEffect="non-scaling-stroke"
            />
            {captions && (
              <g transform={`translate(${x1} ${above ? y1 - th : y1})`} pointerEvents="none">
                <rect width={tw} height={th} fill={b.kind === 'detection' ? 'rgba(5,6,7,0.78)' : color} rx={font * 0.2} />
                <text
                  x={font * 0.4}
                  y={th * 0.72}
                  fontSize={font}
                  fontFamily="'JetBrains Mono Variable', ui-monospace, monospace"
                  fontWeight={600}
                  fill={b.kind === 'detection' ? color : '#0b0c0e'}
                >
                  {text}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}

type FrameBoxesProps = Parameters<typeof FrameBoxes>[0];

/**
 * Превью кадра с рамками: адрес картинки вычисляется и запрашивается, только когда
 * превью видно (в мок-режиме вычисление — это отрисовка сцены на canvas).
 */
export function LazyFrame({ getSrc, ...props }: Omit<FrameBoxesProps, 'src'> & { getSrc: () => string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [visible, setVisible] = useState(false);

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
      { rootMargin: '240px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <span ref={ref} className="fw-thumb">
      <FrameBoxes {...props} src={visible ? getSrc() : null} />
    </span>
  );
}
