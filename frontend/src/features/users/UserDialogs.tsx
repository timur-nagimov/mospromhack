import { Dices, Eye, EyeOff, KeyRound, UserPlus, X } from 'lucide-react';
import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { useCreateUser, useUpdateUser } from '../../api/authQueries';
import { toast } from '../../components/ui/toast';
import { isRoleId, ROLE_META, roleTitle, type RoleInfo, type UserRecord } from '../../domain/auth';

const USERNAME_RE = /^[a-z0-9][a-z0-9._-]{1,31}$/;
const MIN_PASSWORD = 8;
/** Без похожих символов (0/O, 1/l/I) — пароль диктуют по телефону */
const PASSWORD_ALPHABET = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';

export function generatePassword(length = 12): string {
  const bytes = new Uint32Array(length);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => PASSWORD_ALPHABET[b % PASSWORD_ALPHABET.length]).join('');
}

function Dialog({ title, icon, onClose, children }: { title: string; icon: ReactNode; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal modal--sm user-dialog" role="dialog" aria-modal="true" aria-labelledby="user-dialog-title">
        <header className="user-dialog__head">
          <span className="user-dialog__icon" aria-hidden="true">
            {icon}
          </span>
          <h2 id="user-dialog-title">{title}</h2>
          <button type="button" className="btn btn--ghost btn--icon" onClick={onClose} aria-label="Закрыть (Esc)">
            <X size={18} />
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

function PasswordInput({ value, onChange, autoFocus }: { value: string; onChange: (v: string) => void; autoFocus?: boolean }) {
  const [visible, setVisible] = useState(false);
  return (
    <div className="pwd-row">
      <span className="pwd-field">
        <input
          className="input mono"
          type={visible ? 'text' : 'password'}
          autoComplete="new-password"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoFocus={autoFocus}
          aria-invalid={value.length > 0 && value.length < MIN_PASSWORD}
        />
        <button
          type="button"
          className="btn btn--ghost btn--sm btn--icon"
          onClick={() => setVisible((v) => !v)}
          aria-label={visible ? 'Скрыть пароль' : 'Показать пароль'}
        >
          {visible ? <EyeOff size={15} /> : <Eye size={15} />}
        </button>
      </span>
      <button
        type="button"
        className="btn"
        onClick={() => {
          onChange(generatePassword());
          setVisible(true);
        }}
        title="Случайный пароль из 12 символов"
      >
        <Dices size={15} /> Сгенерировать
      </button>
    </div>
  );
}

export function CreateUserDialog({ roles, onClose }: { roles: RoleInfo[]; onClose: () => void }) {
  const create = useCreateUser();
  const [username, setUsername] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [role, setRole] = useState(roles.find((r) => r.id === 'worker')?.id ?? roles[0]?.id ?? 'worker');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const info = roles.find((r) => r.id === role);

  const problem = !USERNAME_RE.test(username)
    ? 'Логин: латиница в нижнем регистре, цифры, «.», «-», «_», от 2 до 32 символов'
    : !displayName.trim()
      ? 'Укажите имя'
      : password.length < MIN_PASSWORD
        ? `Пароль не короче ${MIN_PASSWORD} символов`
        : null;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    create.mutate(
      { username, display_name: displayName.trim(), role, password },
      {
        onSuccess: (user) => {
          toast(`Пользователь ${user.display_name} создан: ${roleTitle(user.role)}`);
          onClose();
        },
        onError: (err) => setError((err as Error).message),
      },
    );
  };

  return (
    <Dialog title="Новый пользователь" icon={<UserPlus size={18} />} onClose={onClose}>
      <form className="user-dialog__body" onSubmit={submit} noValidate>
        <div className="user-dialog__grid">
          <label className="field">
            <span className="field__label">Логин</span>
            <input
              className="input mono"
              value={username}
              onChange={(e) => setUsername(e.target.value.trim().toLowerCase())}
              placeholder="ivanov"
              autoComplete="off"
              autoFocus
              aria-invalid={username.length > 0 && !USERNAME_RE.test(username)}
            />
          </label>
          <label className="field">
            <span className="field__label">Имя и фамилия</span>
            <input className="input" value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="Иван Иванов" />
          </label>
        </div>
        <label className="field">
          <span className="field__label">Роль</span>
          <select className="select" value={role} onChange={(e) => setRole(e.target.value)}>
            {roles.map((r) => (
              <option key={r.id} value={r.id}>
                {r.title}
              </option>
            ))}
          </select>
          <span className="field__hint">{info?.description || (isRoleId(role) ? ROLE_META[role].description : '')}</span>
        </label>
        <label className="field">
          <span className="field__label">Пароль</span>
          <PasswordInput value={password} onChange={setPassword} />
          <span className="field__hint">Передайте пароль сотруднику лично; хранится только хэш (scrypt)</span>
        </label>
        {error && (
          <div className="error-box" role="alert">
            {error}
          </div>
        )}
        <footer className="user-dialog__foot">
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            Отмена
          </button>
          <button type="submit" className="btn btn--primary" disabled={create.isPending}>
            <UserPlus size={15} /> {create.isPending ? 'Создаём…' : 'Создать'}
          </button>
        </footer>
      </form>
    </Dialog>
  );
}

export function PasswordDialog({ user, onClose }: { user: UserRecord; onClose: () => void }) {
  const update = useUpdateUser();
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (password.length < MIN_PASSWORD) {
      setError(`Пароль не короче ${MIN_PASSWORD} символов`);
      return;
    }
    setError(null);
    update.mutate(
      { username: user.username, patch: { password } },
      {
        onSuccess: () => {
          toast(`Пароль ${user.display_name} изменён, его сессии закрыты`);
          onClose();
        },
        onError: (err) => setError((err as Error).message),
      },
    );
  };

  return (
    <Dialog title="Сброс пароля" icon={<KeyRound size={18} />} onClose={onClose}>
      <form className="user-dialog__body" onSubmit={submit} noValidate>
        <p className="user-dialog__lead">
          Новый пароль для <b>{user.display_name}</b> <span className="mono muted">{user.username}</span>. Открытые сессии пользователя
          закроются — войти можно будет только с новым паролем.
        </p>
        <label className="field">
          <span className="field__label">Новый пароль</span>
          <PasswordInput value={password} onChange={setPassword} autoFocus />
        </label>
        {error && (
          <div className="error-box" role="alert">
            {error}
          </div>
        )}
        <footer className="user-dialog__foot">
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            Отмена
          </button>
          <button type="submit" className="btn btn--primary" disabled={update.isPending}>
            <KeyRound size={15} /> {update.isPending ? 'Сохраняем…' : 'Сменить пароль'}
          </button>
        </footer>
      </form>
    </Dialog>
  );
}
