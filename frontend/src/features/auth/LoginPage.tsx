import { Eye, EyeOff, FileClock, LogIn, ServerCog, ShieldCheck, Split, TriangleAlert } from 'lucide-react';
import { useState, type FormEvent } from 'react';
import { api } from '../../api';
import { ApiError } from '../../api/types';
import { MirrorMark } from '../../components/ui/customIcons';
import { isRoleId, ROLE_META, roleTitle } from '../../domain/auth';
import { useAuth } from '../../state/auth';
import { useAuthControls } from './AuthProvider';
import { AccessLevel } from './parts';
import './auth.css';

const PRINCIPLES = [
  { icon: ShieldCheck, title: 'Минимум прав', text: 'Видео и снимки с людьми — только тем, кто реагирует на нарушение' },
  { icon: Split, title: 'Разделение обязанностей', text: 'Зоны и правила — инженер по ОТ, камеры и учётные записи — администратор' },
  { icon: ServerCog, title: 'Проверка на сервере', text: 'Интерфейс прячет лишнее, решает API: запрос без права получит 403' },
  { icon: FileClock, title: 'Журнал действий', text: 'Вход, правки камер, зон, правил, событий и учёток: кто, когда, откуда' },
];

function loginError(err: unknown): string {
  if (err instanceof ApiError && err.status === 401) return 'Неверный логин или пароль';
  if (err instanceof ApiError && err.status === 429) return 'Слишком много попыток — подождите минуту';
  if (err instanceof ApiError && err.status === 0) return 'Сервер недоступен — проверьте подключение';
  return err instanceof Error ? err.message : String(err);
}

export function LoginPage() {
  const config = useAuth((s) => s.config);
  const reason = useAuth((s) => s.reason);
  const { login } = useAuthControls();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // После входа маршрут /login сам уводит на ?next= или стартовую страницу роли (App.tsx)
  const enter = async (user: string, pass: string) => {
    if (pending) return;
    setPending(user);
    setError(null);
    try {
      await login(user, pass);
    } catch (err) {
      setError(loginError(err));
      setPending(null);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    const user = username.trim().toLowerCase();
    if (!user || !password) {
      setError('Введите логин и пароль');
      return;
    }
    void enter(user, password);
  };

  const demoUsers = config?.demo_login ? config.demo_users : [];

  return (
    <div className="login">
      <div className="login__frame reveal">
        <aside className="login__brand">
          <div className="login__mark">
            <MirrorMark size={58} />
            <div>
              <span className="eyebrow">Видеоаналитика охраны труда</span>
              <h1 className="display">Зазеркалье</h1>
            </div>
          </div>
          <p className="login__tagline">
            Пульт службы охраны труда: камеры и тревоги в реальном времени, журнал нарушений, аналитика. Инференс на ускорителе LinQ.
          </p>
          <ul className="login__principles">
            {PRINCIPLES.map(({ icon: Icon, title, text }) => (
              <li key={title}>
                <Icon size={17} strokeWidth={1.8} aria-hidden="true" />
                <div>
                  <b>{title}</b>
                  <span>{text}</span>
                </div>
              </li>
            ))}
          </ul>
          {api().mode === 'mock' && (
            <span className="pill pill--demo login__mock" title="Данные генерируются в браузере (?mock=1)">
              <span className="pill__stripe hazard" />
              Демо-данные
            </span>
          )}
        </aside>

        <section className="login__panel" aria-labelledby="login-title">
          <span className="login__tape hazard" aria-hidden="true" />
          <header className="login__head">
            <span className="eyebrow">Проходная</span>
            <h2 id="login-title" className="display">
              Вход в систему
            </h2>
          </header>

          {reason === 'expired' && <p className="login__note">Сессия завершена — войдите снова</p>}

          <form className="login__form" onSubmit={onSubmit} noValidate>
            <label className="field">
              <span className="field__label">Логин</span>
              <input
                className="input"
                name="username"
                autoComplete="username"
                autoCapitalize="none"
                spellCheck={false}
                autoFocus
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="field__label">Пароль</span>
              <span className="pwd-field">
                <input
                  className="input"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
                <button
                  type="button"
                  className="btn btn--ghost btn--sm btn--icon"
                  onClick={() => setShowPassword((v) => !v)}
                  aria-label={showPassword ? 'Скрыть пароль' : 'Показать пароль'}
                  title={showPassword ? 'Скрыть пароль' : 'Показать пароль'}
                >
                  {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                </button>
              </span>
            </label>
            {error && (
              <div className="error-box" role="alert">
                {error}
              </div>
            )}
            <button type="submit" className="btn btn--primary login__submit" disabled={pending != null}>
              <LogIn size={16} /> {pending && pending === username.trim().toLowerCase() ? 'Вход…' : 'Войти'}
            </button>
          </form>

          {demoUsers.length > 0 && (
            <section className="login__demo" aria-label="Демо-вход">
              <div className="login__demo-head">
                <span className="eyebrow">Демо-вход</span>
                <span className="muted">пароль совпадает с логином</span>
              </div>
              <div className="role-passes">
                {demoUsers.map((u) => (
                  <button
                    key={u.username}
                    type="button"
                    className="role-pass"
                    disabled={pending != null}
                    onClick={() => void enter(u.username, u.username)}
                  >
                    <span className="role-pass__top">
                      <AccessLevel role={u.role} label />
                      <span className="mono">{u.username}</span>
                    </span>
                    <strong>{pending === u.username ? 'Вход…' : `Войти как ${roleTitle(u.role)}`}</strong>
                    <small>{isRoleId(u.role) ? ROLE_META[u.role].summary : u.display_name}</small>
                  </button>
                ))}
              </div>
              <p className="login__warn">
                <TriangleAlert size={14} aria-hidden="true" />
                <span>
                  Для реального объекта выключите демо-вход (<span className="mono">GUARD_DEMO_LOGIN=0</span>) и смените пароли
                </span>
              </p>
            </section>
          )}
        </section>
      </div>
    </div>
  );
}
