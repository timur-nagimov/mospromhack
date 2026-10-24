import { Check, ChevronDown, LogOut, Repeat2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ALL_PERMISSIONS, roleTitle } from '../../domain/auth';
import { useAuth } from '../../state/auth';
import { useAuthControls } from './AuthProvider';
import { AccessLevel, Avatar } from './parts';
import './auth.css';

/**
 * Меню пользователя в шапке: имя, роль, «Выйти»; при демо-входе — «Сменить роль».
 * При выключенном входе (GUARD_AUTH=0) меню нет — всё как до ролей.
 */
export function UserMenu() {
  const user = useAuth((s) => s.user);
  const config = useAuth((s) => s.config);
  const permissions = useAuth((s) => s.permissions);
  const { logout, switchUser } = useAuthControls();
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

  if (!config?.auth_enabled || !user) return null;

  const demo = config.demo_login ? config.demo_users : [];
  const granted = ALL_PERMISSIONS.filter((p) => permissions.has(p)).length;

  const run = async (key: string, action: () => Promise<unknown>) => {
    setBusy(key);
    try {
      await action();
    } finally {
      setBusy(null);
      setOpen(false);
    }
  };

  return (
    <div className="user-menu" ref={rootRef}>
      <button
        type="button"
        className={`user-chip${open ? ' is-open' : ''}`}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title={`${user.display_name} · ${user.role_title}`}
      >
        <Avatar name={user.display_name} size="sm" />
        <span className="user-chip__text">
          <strong>{user.display_name}</strong>
          <small>{user.role_title || roleTitle(user.role)}</small>
        </span>
        <ChevronDown size={14} className="user-chip__chevron" aria-hidden="true" />
      </button>

      {open && (
        <div className="user-pop" role="menu" aria-label="Пользователь">
          <header className="user-pop__head">
            <Avatar name={user.display_name} size="lg" />
            <div className="user-pop__who">
              <strong>{user.display_name}</strong>
              <span className="mono muted">{user.username}</span>
              <span className="user-pop__role">
                <AccessLevel role={user.role} label />
                {user.role_title || roleTitle(user.role)}
              </span>
            </div>
          </header>
          <p className="user-pop__perms">
            Прав у роли: <b className="num">{granted}</b> из <span className="num">{ALL_PERMISSIONS.length}</span>
          </p>

          {demo.length > 0 && (
            <div className="user-pop__section">
              <span className="eyebrow">
                <Repeat2 size={12} aria-hidden="true" /> Сменить роль
              </span>
              <ul>
                {demo.map((u) => {
                  const current = u.username === user.username;
                  return (
                    <li key={u.username}>
                      <button
                        type="button"
                        role="menuitem"
                        className={`user-pop__item${current ? ' is-current' : ''}`}
                        disabled={current || busy != null}
                        onClick={() => void run(u.username, () => switchUser(u.username))}
                      >
                        <AccessLevel role={u.role} />
                        <span className="user-pop__item-text">
                          {busy === u.username ? 'Вход…' : roleTitle(u.role)}
                          <small className="mono">{u.username}</small>
                        </span>
                        {current && <Check size={14} aria-label="текущая роль" />}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          <button
            type="button"
            role="menuitem"
            className="user-pop__item user-pop__logout"
            disabled={busy != null}
            onClick={() => void run('logout', logout)}
          >
            <LogOut size={15} aria-hidden="true" />
            {busy === 'logout' ? 'Выход…' : 'Выйти'}
          </button>
        </div>
      )}
    </div>
  );
}
