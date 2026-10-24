import { LockKeyhole, LogIn } from 'lucide-react';
import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useRolesCatalog } from '../../api/authQueries';
import { sectionFor, startPath } from '../../components/layout/sections';
import { PERMISSION_META, roleTitle, type Permission } from '../../domain/auth';
import { useAuth, useCan } from '../../state/auth';
import { useAuthControls } from './AuthProvider';
import { AccessLevel } from './parts';
import './auth.css';

/**
 * «Нет доступа»: закрытая зона с названием нужного права, ролью пользователя и
 * ролями, у которых право есть. При демо-входе — переход в подходящую роль.
 */
export function NoAccess({ permission, area, embedded = false }: { permission: Permission; area?: string; embedded?: boolean }) {
  const { pathname } = useLocation();
  const user = useAuth((s) => s.user);
  const config = useAuth((s) => s.config);
  const can = useCan();
  const { switchUser } = useAuthControls();
  const { catalog } = useRolesCatalog();
  const [switching, setSwitching] = useState<string | null>(null);

  const title = area ?? sectionFor(pathname)?.title ?? 'Раздел';
  const meta = PERMISSION_META[permission];
  const holders = catalog.roles.filter((r) => r.permissions.includes(permission));
  const start = startPath(can);
  const demo = config?.demo_login
    ? config.demo_users.filter((u) => u.username !== user?.username && holders.some((r) => r.id === u.role))
    : [];

  const card = (
    <section className="no-access reveal" role="alert" aria-labelledby="no-access-title">
      <span className="no-access__tape hazard" aria-hidden="true" />
      <div className="no-access__body">
        <span className="no-access__icon" aria-hidden="true">
          <LockKeyhole size={26} strokeWidth={1.8} />
        </span>
        <span className="eyebrow">Закрытая зона · {title}</span>
        <h2 id="no-access-title" className="display">
          Нет доступа
        </h2>
        <p className="no-access__lead">Раздел открыт ролям с правом</p>
        <div className="perm-tag">
          <code className="mono">{permission}</code>
          <span>{meta?.title ?? permission}</span>
        </div>
        {meta?.description && <p className="no-access__desc muted">{meta.description}</p>}
        <dl className="no-access__facts">
          <div>
            <dt>Ваша роль</dt>
            <dd>
              {user && <AccessLevel role={user.role} />}
              {user?.role_title || roleTitle(user?.role)}
            </dd>
          </div>
          <div>
            <dt>Право есть у ролей</dt>
            <dd>{holders.length ? holders.map((r) => r.title).join(', ') : '—'}</dd>
          </div>
        </dl>
        <div className="no-access__actions">
          {start && start !== pathname && (
            <Link className="btn btn--primary" to={start}>
              На стартовую страницу
            </Link>
          )}
          {demo.map((u) => (
            <button
              key={u.username}
              type="button"
              className="btn"
              disabled={switching != null}
              onClick={async () => {
                setSwitching(u.username);
                await switchUser(u.username);
                setSwitching(null);
              }}
            >
              <LogIn size={15} /> {switching === u.username ? 'Вход…' : `Войти как ${roleTitle(u.role)}`}
            </button>
          ))}
        </div>
      </div>
    </section>
  );

  return embedded ? card : <div className="page no-access-page">{card}</div>;
}
