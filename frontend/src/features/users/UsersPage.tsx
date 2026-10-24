import { KeyRound, UserCheck, UserPlus, UsersRound, UserX } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useRolesCatalog, useUpdateUser, useUsers } from '../../api/authQueries';
import { toast } from '../../components/ui/toast';
import { roleLevel, roleTitle, type UserRecord } from '../../domain/auth';
import { formatDate, formatSmartDateTime, plural } from '../../domain/format';
import { useAuth } from '../../state/auth';
import { AccessLevel, Avatar } from '../auth/parts';
import { RoleMatrix } from './RoleMatrix';
import { CreateUserDialog, PasswordDialog } from './UserDialogs';
import './users.css';

type DialogState = { kind: 'create' } | { kind: 'password'; user: UserRecord } | null;

const SELF_HINT = 'Свою учётную запись меняет другой администратор';

/** «Пользователи» (users.manage): учётные записи и матрица ролей и прав */
export function UsersPage() {
  const { data: users = [], isLoading, isError, error } = useUsers();
  const me = useAuth((s) => s.user);
  const { catalog, fromServer } = useRolesCatalog();
  const update = useUpdateUser();
  const [dialog, setDialog] = useState<DialogState>(null);

  const sorted = useMemo(
    () =>
      [...users].sort(
        (a, b) =>
          Number(b.active) - Number(a.active) || roleLevel(b.role) - roleLevel(a.role) || a.display_name.localeCompare(b.display_name, 'ru'),
      ),
    [users],
  );
  const active = users.filter((u) => u.active).length;
  // роль из каталога сервера; неизвестную роль пользователя тоже показываем в списке
  const roleOptions = (current: string) =>
    catalog.roles.some((r) => r.id === current) ? catalog.roles : [...catalog.roles, { id: current, title: roleTitle(current), description: '', permissions: [] }];

  const changeRole = (user: UserRecord, role: string) => {
    if (role === user.role) return;
    update.mutate(
      { username: user.username, patch: { role } },
      {
        onSuccess: (saved) => toast(`${saved.display_name}: роль «${roleTitle(saved.role)}»`),
        onError: (err) => toast(`Роль не изменена: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  const toggleActive = (user: UserRecord) => {
    if (user.active && !window.confirm(`Отключить ${user.display_name} (${user.username})? Войти будет нельзя, открытые сессии закроются.`)) return;
    update.mutate(
      { username: user.username, patch: { active: !user.active } },
      {
        onSuccess: (saved) => toast(saved.active ? `${saved.display_name} снова может войти` : `${saved.display_name} отключён`),
        onError: (err) => toast(`Не изменено: ${(err as Error).message}`, 'bad'),
      },
    );
  };

  return (
    <div className="page users">
      <section className="panel users__accounts reveal" aria-labelledby="users-title">
        <div className="panel__head">
          <div>
            <h3 id="users-title" className="panel__title">
              Учётные записи
            </h3>
            <p className="panel__sub">
              {users.length} {plural(users.length, ['учётная запись', 'учётные записи', 'учётных записей'])} · {active}{' '}
              {plural(active, ['активна', 'активны', 'активны'])} · смена роли действует сразу, смена пароля и отключение закрывают сессии
            </p>
          </div>
          <button type="button" className="btn btn--primary btn--sm" onClick={() => setDialog({ kind: 'create' })}>
            <UserPlus size={15} /> Добавить пользователя
          </button>
        </div>

        {isError ? (
          <div className="panel__body">
            <div className="error-box">Список пользователей не загрузился: {(error as Error).message}</div>
          </div>
        ) : isLoading ? (
          <div className="empty">
            <span className="spinner" />
          </div>
        ) : users.length === 0 ? (
          <div className="empty">
            <UsersRound size={28} />
            <span className="empty__title">Учётных записей нет</span>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="table users__table">
              <thead>
                <tr>
                  <th>Пользователь</th>
                  <th>Роль</th>
                  <th>Статус</th>
                  <th>Последний вход</th>
                  <th>Создан</th>
                  <th className="right">Действия</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((u) => {
                  const self = u.username === me?.username;
                  return (
                    <tr key={u.username} className={u.active ? undefined : 'is-off'}>
                      <td>
                        <div className="user-cell">
                          <Avatar name={u.display_name} />
                          <div className="user-cell__text">
                            <strong>
                              {u.display_name}
                              {self && <span className="badge badge--plain user-cell__you">это вы</span>}
                            </strong>
                            <span className="mono muted">{u.username}</span>
                          </div>
                        </div>
                      </td>
                      <td>
                        <div className="role-select" title={self ? SELF_HINT : undefined}>
                          <AccessLevel role={u.role} />
                          <select
                            className="select select--sm"
                            value={u.role}
                            disabled={self || update.isPending}
                            onChange={(e) => changeRole(u, e.target.value)}
                            aria-label={`Роль: ${u.display_name}`}
                          >
                            {roleOptions(u.role).map((r) => (
                              <option key={r.id} value={r.id}>
                                {r.title}
                              </option>
                            ))}
                          </select>
                        </div>
                      </td>
                      <td>
                        {u.active ? (
                          <span className="badge badge--confirmed">
                            <span className="badge__mark" aria-hidden="true" />
                            Активна
                          </span>
                        ) : (
                          <span className="badge badge--false">
                            <span className="badge__mark" aria-hidden="true" />
                            Отключена
                          </span>
                        )}
                      </td>
                      <td>{u.last_login_at ? formatSmartDateTime(u.last_login_at) : <span className="muted">не входил</span>}</td>
                      <td className="dim">{u.created_at ? formatDate(u.created_at) : '—'}</td>
                      <td className="right">
                        <div className="row-actions" title={self ? SELF_HINT : undefined}>
                          <button type="button" className="btn btn--sm" disabled={self} onClick={() => setDialog({ kind: 'password', user: u })}>
                            <KeyRound size={14} /> Сбросить пароль
                          </button>
                          <button
                            type="button"
                            className={`btn btn--sm${u.active ? ' btn--danger' : ''}`}
                            disabled={self || update.isPending}
                            onClick={() => toggleActive(u)}
                          >
                            {u.active ? <UserX size={14} /> : <UserCheck size={14} />}
                            {u.active ? 'Отключить' : 'Включить'}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <RoleMatrix catalog={catalog} fromServer={fromServer} currentRole={me?.role} />

      {dialog?.kind === 'create' && <CreateUserDialog roles={catalog.roles} onClose={() => setDialog(null)} />}
      {dialog?.kind === 'password' && <PasswordDialog user={dialog.user} onClose={() => setDialog(null)} />}
    </div>
  );
}
