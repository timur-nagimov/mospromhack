import { Ban, Check, Forward } from 'lucide-react';
import type { EventStatus, Severity } from '../../domain/types';
import { SEVERITY_LABELS, STATUS_LABELS } from '../../domain/labels';
import { getViolationMeta } from '../../domain/violations';

export const SEVERITY_COLOR: Record<Severity, string> = {
  critical: 'var(--sev-critical)',
  high: 'var(--sev-high)',
  medium: 'var(--sev-medium)',
  low: 'var(--sev-low)',
};

export function SeverityBadge({ severity, solid = false }: { severity: Severity; solid?: boolean }) {
  return (
    <span className={`badge badge--${severity}${solid && severity === 'critical' ? ' badge--solid-critical' : ''}`}>
      {!(solid && severity === 'critical') && <span className="badge__mark" aria-hidden="true" />}
      {SEVERITY_LABELS[severity]}
    </span>
  );
}

const STATUS_ICON: Partial<Record<EventStatus, typeof Check>> = {
  confirmed: Check,
  forwarded: Forward,
  false: Ban,
};

export function StatusBadge({ status }: { status: EventStatus }) {
  const Icon = STATUS_ICON[status];
  return (
    <span className={`badge badge--${status}`}>
      {Icon ? <Icon size={12} strokeWidth={2.6} aria-hidden="true" /> : <span className="badge__mark" aria-hidden="true" />}
      {STATUS_LABELS[status]}
    </span>
  );
}

/** Табличка типа нарушения: цвет — по важности события */
export function ViolationIcon({
  type,
  severity,
  size = 'md',
}: {
  type: string;
  severity: Severity;
  size?: 'sm' | 'md' | 'lg';
}) {
  const meta = getViolationMeta(type);
  const Icon = meta.icon;
  const px = size === 'lg' ? 24 : size === 'sm' ? 14 : 17;
  return (
    <span
      className={`vicon${size === 'md' ? '' : ` vicon--${size}`}`}
      style={{ ['--c' as string]: SEVERITY_COLOR[severity] }}
      title={meta.title}
    >
      <Icon size={px} strokeWidth={2} aria-hidden="true" />
    </span>
  );
}
