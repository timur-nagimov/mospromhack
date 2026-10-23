import type { SVGProps } from 'react';

// Иконки, которых нет в lucide, — в той же сетке 24×24 и толщине штриха

export interface IconProps extends Omit<SVGProps<SVGSVGElement>, 'ref'> {
  size?: number | string;
  strokeWidth?: number | string;
}

function base({ size = 24, strokeWidth = 2, ...rest }: IconProps): SVGProps<SVGSVGElement> {
  return {
    xmlns: 'http://www.w3.org/2000/svg',
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth,
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    ...rest,
  };
}

/** Человек лежит — падение и неподвижность */
export function PersonFallenIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <circle cx="5" cy="13.5" r="2" />
      <path d="M8 15h8l4 -3" />
      <path d="M16 15l4 2.5" />
      <path d="M11 15l2 -4" />
      <path d="M2 20h20" />
    </svg>
  );
}

/** Лестница — работа на высоте */
export function LadderIcon(props: IconProps) {
  return (
    <svg {...base(props)}>
      <path d="M7 2v20" />
      <path d="M17 2v20" />
      <path d="M7 6h10" />
      <path d="M7 11h10" />
      <path d="M7 16h10" />
    </svg>
  );
}

/** Логотип: буква «З» и её отражение по обе стороны «стекла» */
export function MirrorMark({ size = 40 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden="true">
      <defs>
        <pattern id="mm-hazard" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="3.5" height="7" fill="#f5c400" />
          <rect x="3.5" width="3.5" height="7" fill="#101114" />
        </pattern>
      </defs>
      <rect x="0.5" y="0.5" width="39" height="39" rx="7" fill="url(#mm-hazard)" />
      <rect x="4" y="4" width="32" height="32" rx="5" fill="#0d0e10" />
      <text
        x="13.4"
        y="27.5"
        textAnchor="middle"
        fontFamily="'Tektur Variable', sans-serif"
        fontWeight="700"
        fontSize="19"
        fill="#f5c400"
      >
        З
      </text>
      <g transform="translate(53.2 0) scale(-1 1)">
        <text
          x="26.6"
          y="27.5"
          textAnchor="middle"
          fontFamily="'Tektur Variable', sans-serif"
          fontWeight="700"
          fontSize="19"
          fill="#f5c400"
          opacity="0.45"
        >
          З
        </text>
      </g>
      <rect x="19.5" y="9" width="1.2" height="22" fill="#f5c400" />
    </svg>
  );
}
