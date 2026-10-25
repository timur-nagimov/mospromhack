// Мелкие элементы ролевой модели: аватар, уровень допуска, бейдж роли

import { initials, roleLevel, roleTitle } from '../../domain/auth';

/** Квадратный «пропуск» с инициалами */
export function Avatar({ name, size = 'md' }: { name: string; size?: 'sm' | 'md' | 'lg' }) {
  return (
    <span className={`avatar avatar--${size}`} aria-hidden="true">
      {initials(name)}
    </span>
  );
}

/** Уровень допуска I–IV: четыре деления, заполнено по уровню роли */
export function AccessLevel({ role, label = false }: { role: string; label?: boolean }) {
  const level = roleLevel(role);
  if (!level) return null;
  return (
    <span className="access-level" title={`Уровень допуска ${['I', 'II', 'III', 'IV'][level - 1]} из IV`}>
      <span className="access-level__bars" aria-hidden="true">
        {[1, 2, 3, 4].map((i) => (
          <i key={i} className={i <= level ? 'is-on' : undefined} />
        ))}
      </span>
      {label && <span className="access-level__text mono">{['I', 'II', 'III', 'IV'][level - 1]}</span>}
    </span>
  );
}

export function RoleBadge({ role, title }: { role: string; title?: string }) {
  return (
    <span className="badge role-badge">
      <AccessLevel role={role} />
      {title || roleTitle(role)}
    </span>
  );
}
