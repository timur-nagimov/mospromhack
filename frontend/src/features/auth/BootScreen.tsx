import { PlugZap, RefreshCw } from 'lucide-react';
import { MirrorMark } from '../../components/ui/customIcons';
import { useAuth } from '../../state/auth';
import { useAuthControls } from './AuthProvider';
import './auth.css';

/** Пока не ясно, кто вошёл (/api/auth/config и /me), или сервер не ответил */
export function BootScreen() {
  const status = useAuth((s) => s.status);
  const error = useAuth((s) => s.error);
  const { retry } = useAuthControls();

  if (status !== 'error') {
    return (
      <div className="boot" aria-busy="true">
        <MirrorMark size={48} />
        <span className="spinner" aria-label="Загрузка" />
      </div>
    );
  }

  return (
    <div className="boot" role="alert">
      <MirrorMark size={48} />
      <div className="boot__error">
        <PlugZap size={20} aria-hidden="true" />
        <div>
          <strong>Нет связи с сервером</strong>
          <p>
            {error ?? 'Сервер не ответил'}. Повторяем каждые 5 с. Посмотреть интерфейс без сервера можно на демо-данных:{' '}
            <a href="?mock=1">?mock=1</a>
          </p>
        </div>
        <button type="button" className="btn btn--sm" onClick={retry}>
          <RefreshCw size={14} /> Повторить
        </button>
      </div>
    </div>
  );
}
