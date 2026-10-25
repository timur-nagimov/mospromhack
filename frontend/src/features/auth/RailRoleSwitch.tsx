import { ChevronDown } from 'lucide-react';
import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { sectionFor } from '../../components/layout/sections';
import { isRoleId, ROLE_META, ROLE_PERMISSIONS, roleColor, roleTitle } from '../../domain/auth';
import { isRole, ROLES } from '../../domain/roles';
import { useAuth } from '../../state/auth';
import { selectRole, useRole } from '../../state/role';
import { confirmLeave } from '../../state/unsaved';
import { useAuthControls } from './AuthProvider';
import { AccessLevel } from './parts';

/**
 * Вход выключен (GUARD_AUTH=0): локальный выбор роли коллеги — права интерфейса
 * и заголовок X-Guard-Role для справочника сотрудников.
 */
export function LocalRoleSelector() {
  const role = useRole();
  const authOn = useAuth((s) => Boolean(s.config?.auth_enabled));
  const { pathname } = useLocation();
  if (authOn) return null;
  const section = sectionFor(pathname);
  return (
    <label className="role-selector" title={`Роль: ${ROLES[role].title}`} style={{ ['--role' as string]: roleColor(role) }}>
      <span className="role-selector__caption" aria-hidden="true">
        Роль <ChevronDown size={12} />
      </span>
      <span className="role-selector__value" aria-hidden="true">
        {ROLES[role].shortTitle}
      </span>
      <select
        aria-label="Роль"
        value={role}
        onChange={(event) => {
          const next = event.currentTarget.value;
          if (!isRole(next) || next === role) return;
          // раздел закроется для новой роли — сначала спросить про несохранённые правки
          const keeps = !section || ROLE_PERMISSIONS[next].includes(section.permission);
          if (keeps || confirmLeave()) selectRole(next);
        }}
      >
        {Object.entries(ROLES).map(([id, { title }]) => (
          <option key={id} value={id}>
            {title}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Демо-вход: смена роли в один клик прямо в колонне навигации */
export function DemoRoleSwitch() {
  const config = useAuth((s) => s.config);
  const username = useAuth((s) => s.user?.username);
  const { switchUser } = useAuthControls();
  const [busy, setBusy] = useState<string | null>(null);
  if (!config?.auth_enabled || !config.demo_login || config.demo_users.length === 0) return null;
  return (
    <div className="demo-switch" role="group" aria-label="Демо: сменить роль">
      <span className="demo-switch__caption">Демо-роль</span>
      {config.demo_users.map((u) => {
        const current = u.username === username;
        return (
          <button
            key={u.username}
            type="button"
            className={`demo-switch__btn${current ? ' is-current' : ''}`}
            style={{ ['--role' as string]: roleColor(u.role) }}
            disabled={current || busy != null}
            aria-pressed={current}
            title={current ? `Вы вошли как ${roleTitle(u.role)}` : `Войти как ${roleTitle(u.role)}`}
            onClick={async () => {
              if (!confirmLeave()) return;
              setBusy(u.username);
              await switchUser(u.username);
              setBusy(null);
            }}
          >
            <AccessLevel role={u.role} />
            <span>{busy === u.username ? 'Вход…' : isRoleId(u.role) ? ROLE_META[u.role].short : u.role}</span>
          </button>
        );
      })}
    </div>
  );
}
