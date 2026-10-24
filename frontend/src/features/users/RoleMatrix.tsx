import { Check } from 'lucide-react';
import { useMemo } from 'react';
import { isRoleId, PERMISSION_GROUPS, PERMISSION_META, ROLE_META, type Permission, type PermissionInfo, type RolesCatalog } from '../../domain/auth';
import { AccessLevel } from '../auth/parts';

interface Group {
  title: string;
  permissions: PermissionInfo[];
}

/** Строки матрицы по участкам ответственности; права, которых нет в словаре, — в «Прочие» */
function groupPermissions(catalog: RolesCatalog): Group[] {
  const byId = new Map(catalog.permissions.map((p) => [p.id, p]));
  const used = new Set<string>();
  const groups: Group[] = PERMISSION_GROUPS.map((g) => ({
    title: g.title,
    permissions: g.permissions.filter((id) => byId.has(id)).map((id) => {
      used.add(id);
      return byId.get(id) as PermissionInfo;
    }),
  })).filter((g) => g.permissions.length > 0);
  const rest = catalog.permissions.filter((p) => !used.has(p.id));
  if (rest.length) groups.push({ title: 'Прочие', permissions: rest });
  return groups;
}

/**
 * Матрица «роли × права» для экспертов: роли по возрастанию допуска, права —
 * по участкам ответственности, у каждого права — что оно открывает.
 */
export function RoleMatrix({ catalog, fromServer, currentRole }: { catalog: RolesCatalog; fromServer: boolean; currentRole?: string }) {
  const groups = useMemo(() => groupPermissions(catalog), [catalog]);
  const roles = catalog.roles;
  const total = catalog.permissions.length;

  return (
    <section className="panel role-matrix reveal" style={{ ['--i' as string]: 1 }} aria-labelledby="role-matrix-title">
      <div className="panel__head role-matrix__head">
        <div>
          <h3 id="role-matrix-title" className="panel__title">
            Роли и права
          </h3>
          <p className="panel__sub">
            Каждый видит то, что нужно для его работы. Интерфейс прячет недоступное, решает сервер: запрос без права получает 403
          </p>
        </div>
        <span className="role-matrix__source" title={fromServer ? 'Матрица с сервера: guard/auth/roles.py' : 'Сервер не ответил — матрица по docs/ROLES.md'}>
          источник: <span className="mono">{fromServer ? 'GET /api/auth/roles' : 'docs/ROLES.md'}</span>
        </span>
      </div>
      <div className="role-matrix__scroll">
        <table className="role-matrix__table">
          <colgroup>
            <col className="role-matrix__col-perm" />
            {roles.map((r) => (
              <col key={r.id} />
            ))}
          </colgroup>
          <thead>
            <tr>
              <th scope="col" className="role-matrix__corner">
                <span className="eyebrow">Право</span>
              </th>
              {roles.map((r) => (
                <th key={r.id} scope="col" className={r.id === currentRole ? 'is-current' : undefined}>
                  <div className="role-head">
                    <AccessLevel role={r.id} label />
                    <strong>{r.title}</strong>
                    <small>{isRoleId(r.id) ? ROLE_META[r.id].who : r.description}</small>
                    {r.id === currentRole && <span className="role-head__you">ваша роль</span>}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          {groups.map((g) => (
            <tbody key={g.title}>
              <tr className="role-matrix__group">
                <th scope="rowgroup" colSpan={roles.length + 1}>
                  {g.title}
                </th>
              </tr>
              {g.permissions.map((p) => (
                <tr key={p.id}>
                  <th scope="row">
                    <div className="perm-cell">
                      <span className="perm-cell__title">
                        {p.title}
                        <code className="mono">{p.id}</code>
                        {PERMISSION_META[p.id as Permission]?.personalData && (
                          <span className="pd-tag" title="Изображения людей — персональные данные (152-ФЗ)">
                            ПДн
                          </span>
                        )}
                      </span>
                      {p.description && <span className="perm-cell__desc">{p.description}</span>}
                    </div>
                  </th>
                  {roles.map((r) => {
                    const on = r.permissions.includes(p.id);
                    return (
                      <td key={r.id} className={`${on ? 'is-on' : 'is-off'}${r.id === currentRole ? ' is-current' : ''}`}>
                        {on ? (
                          <span className="grant" role="img" aria-label={`${r.title}: есть`}>
                            <Check size={14} strokeWidth={3} />
                          </span>
                        ) : (
                          <span className="deny" role="img" aria-label={`${r.title}: нет`} />
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          ))}
          <tfoot>
            <tr>
              <th scope="row">Всего прав</th>
              {roles.map((r) => (
                <td key={r.id} className={r.id === currentRole ? 'is-current' : undefined}>
                  <b className="num">{r.permissions.length}</b>
                  <span className="muted"> из {total}</span>
                </td>
              ))}
            </tr>
          </tfoot>
        </table>
      </div>
    </section>
  );
}
