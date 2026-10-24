import { Check, ChevronDown, Lock, LogOut, Repeat2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useRolesCatalog } from '../../api/authQueries';
import { SECTIONS } from '../../components/layout/sections';
import { ALL_PERMISSIONS, isRoleId, needRoleText, ROLE_META, ROLE_ORDER, roleColor, roleTitle } from '../../domain/auth';
import { useAuth } from '../../state/auth';
import { selectRole } from '../../state/role';
import { useAuthControls } from './AuthProvider';
import { ROLE_ACTIONS } from './lock';
import { AccessLevel, Avatar } from './parts';
import './auth.css';

/**
 * Бейдж роли в шапке: цвет и уровень допуска. По клику — карточка «Что может эта роль»:
 * разделы и действия, доступные и недоступные (с подсказкой, какой роли они открыты),
 * смена роли в один клик (демо-вход или локальный выбор при выключенном входе) и «Выйти».
 */
export function UserMenu() {
  const user = useAuth((s) => s.user);
  const config = useAuth((s) => s.config);
  const permissions = useAuth((s) => s.permissions);
  const { logout, switchUser } = useAuthControls();
  const { catalog } = useRolesCatalog();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);

  if (!user || !config) return null;

  const authOn = config.auth_enabled;
  const color = roleColor(user.role);
  const title = user.role_title || roleTitle(user.role);
  const granted = ALL_PERMISSIONS.filter((p) => permissions.has(p)).length;
  const sections = SECTIONS.filter((s) => !(config.legacy && (s.permission === 'users.manage' || s.permission === 'audit.view')));
  // смена роли: демо-учётки при входе, локальный выбор при выключенном входе
  const switchTargets = authOn
    ? config.demo_login
      ? config.demo_users.map((u) => ({ key: u.username, role: u.role, current: u.username === user.username }))
      : []
    : ROLE_ORDER.map((role) => ({ key: role, role, current: role === user.role }));

  const run = async (key: string, action: () => Promise<unknown> | void) => {
    setBusy(key);
    try {
      await action();
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="user-menu" ref={rootRef}>
      <button
        type="button"
        className={`role-chip${open ? ' is-open' : ''}`}
        style={{ ['--role' as string]: color }}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title="Что может эта роль"
      >
        <span className="role-chip__stripe" aria-hidden="true" />
        {authOn && <Avatar name={user.display_name} size="sm" />}
        <span className="role-chip__text">
          <strong>{authOn ? user.display_name : title}</strong>
          <small>
            <AccessLevel role={user.role} />
            {authOn ? title : 'вход выключен'}
          </small>
        </span>
        <ChevronDown size={14} className="role-chip__chevron" aria-hidden="true" />
      </button>

      {open && (
        <div className="role-card" role="dialog" aria-label={`Что может роль «${title}»`} style={{ ['--role' as string]: color }}>
          <header className="role-card__head">
            <span className="role-card__badge">
              <AccessLevel role={user.role} label />
              {title}
            </span>
            <div className="role-card__who">
              {authOn ? (
                <>
                  <strong>{user.display_name}</strong>
                  <span className="mono muted">{user.username}</span>
                </>
              ) : (
                <span className="muted">Вход выключен (GUARD_AUTH=0): роль выбирается здесь, сервер отвечает как администратору</span>
              )}
            </div>
            <span className="role-card__count">
              <b className="num">{granted}</b> из {ALL_PERMISSIONS.length} прав
            </span>
          </header>
          {isRoleId(user.role) && <p className="role-card__about">{ROLE_META[user.role].description}</p>}

          <div className="role-card__cols">
            <section>
              <span className="eyebrow">Разделы</span>
              <ul className="cap-list">
                {sections.map((s) => {
                  const on = permissions.has(s.permission);
                  const Icon = s.icon;
                  return (
                    <li key={s.path} className={on ? 'is-on' : 'is-off'} title={on ? undefined : needRoleText(s.permission, catalog.roles)}>
                      <Icon size={14} aria-hidden="true" />
                      <span className="cap-list__text">
                        {s.title}
                        {!on && <small>{needRoleText(s.permission, catalog.roles)}</small>}
                      </span>
                      {on ? <Check size={14} className="cap-list__mark" aria-label="доступно" /> : <Lock size={13} className="cap-list__mark" aria-label="недоступно" />}
                    </li>
                  );
                })}
              </ul>
            </section>
            <section>
              <span className="eyebrow">Действия</span>
              <ul className="cap-list">
                {ROLE_ACTIONS.map((a) => {
                  const on = permissions.has(a.permission);
                  return (
                    <li key={a.permission} className={on ? 'is-on' : 'is-off'}>
                      {on ? <Check size={14} className="cap-list__lead" aria-label="можно" /> : <Lock size={13} className="cap-list__lead" aria-label="нельзя" />}
                      <span className="cap-list__text">
                        {a.text}
                        {!on && <small>{needRoleText(a.permission, catalog.roles)}</small>}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </section>
          </div>

          {switchTargets.length > 0 && (
            <div className="role-card__switch">
              <span className="eyebrow">
                <Repeat2 size={12} aria-hidden="true" /> Сменить роль{authOn ? ' (демо-вход)' : ''}
              </span>
              <div className="role-switch">
                {switchTargets.map((t) => (
                  <button
                    key={t.key}
                    type="button"
                    className={`role-switch__btn${t.current ? ' is-current' : ''}`}
                    style={{ ['--role' as string]: roleColor(t.role) }}
                    disabled={t.current || busy != null}
                    aria-pressed={t.current}
                    onClick={() =>
                      void run(t.key, async () => {
                        if (authOn) {
                          if (await switchUser(t.key)) setOpen(false);
                        } else {
                          selectRole(t.role as Parameters<typeof selectRole>[0]);
                        }
                      })
                    }
                  >
                    <AccessLevel role={t.role} />
                    {busy === t.key ? 'Вход…' : isRoleId(t.role) ? ROLE_META[t.role].short : t.role}
                  </button>
                ))}
              </div>
            </div>
          )}

          {authOn && (
            <button type="button" className="role-card__logout" disabled={busy != null} onClick={() => void run('logout', logout)}>
              <LogOut size={15} aria-hidden="true" />
              {busy === 'logout' ? 'Выход…' : 'Выйти'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
