import { Bell, BellOff, ChartColumn, Cctv, Gauge, LayoutGrid, ScrollText, Siren, X } from 'lucide-react';
import { useEffect, type ReactNode } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { api } from '../../api';
import { useCameraMap } from '../../api/queries';
import { formatClock, formatLongDate, formatTime } from '../../domain/format';
import { backendLabel } from '../../domain/labels';
import { getViolationMeta } from '../../domain/violations';
import { useEventModal } from '../../features/event/useEventModal';
import { isAudioBlocked } from '../../hooks/alarmSound';
import { useNow } from '../../hooks/useNow';
import { liveActions, useLive } from '../../state/live';
import { toggleSound, useSettings } from '../../state/settings';
import { confirmLeave } from '../../state/unsaved';
import { MirrorMark } from '../ui/customIcons';
import './shell.css';

const NAV = [
  { to: '/', label: 'Мониторинг', icon: LayoutGrid, end: true },
  { to: '/journal', label: 'Журнал', icon: ScrollText },
  { to: '/analytics', label: 'Аналитика', icon: ChartColumn },
  { to: '/cameras', label: 'Камеры и правила', icon: Cctv },
  { to: '/performance', label: 'Производи­тельность', icon: Gauge },
];

const TITLES: Record<string, { title: string; sub: string }> = {
  '/': { title: 'Мониторинг', sub: 'Камеры и лента тревог в реальном времени' },
  '/journal': { title: 'Журнал событий', sub: 'Доказательная база службы охраны труда' },
  '/analytics': { title: 'Аналитика', sub: 'Нарушения по типам, участкам и времени' },
  '/cameras': { title: 'Камеры и правила', sub: 'Зоны, расписания и правила без кода' },
  '/performance': { title: 'Производительность', sub: 'FPS, задержки по этапам и загрузка ускорителя' },
};

function NavRail() {
  const unreviewed = useLive((s) => s.unreviewed);
  const hasCritical = useLive((s) => s.alarm != null);
  return (
    <nav className="rail" aria-label="Разделы">
      <div className="rail__brand" title="Зазеркалье — видеоаналитика ТБ">
        <MirrorMark size={42} />
      </div>
      <ul className="rail__nav">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={end}
              className="rail__link"
              onClick={(e) => {
                if (!confirmLeave()) e.preventDefault();
              }}
            >
              <span className="rail__icon">
                <Icon size={20} strokeWidth={1.8} aria-hidden="true" />
                {to === '/' && unreviewed != null && unreviewed > 0 && (
                  <span className={`rail__badge${hasCritical ? ' rail__badge--critical' : ''}`}>
                    {unreviewed > 99 ? '99+' : unreviewed}
                  </span>
                )}
              </span>
              <span className="rail__label">{label}</span>
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="rail__foot">
        <span className="rail__ver mono">v0.1</span>
      </div>
    </nav>
  );
}

function ConnectionPill() {
  const connection = useLive((s) => s.connection);
  const label = connection === 'open' ? 'Поток событий' : connection === 'connecting' ? 'Подключение…' : 'Нет связи';
  const cls = connection === 'open' ? '' : connection === 'connecting' ? ' live-dot--warn' : ' live-dot--bad';
  return (
    <span className="pill" title="WebSocket /ws/events">
      <span className={`live-dot${cls}`} />
      {label}
    </span>
  );
}

function BackendPill() {
  const metrics = useLive((s) => s.metrics);
  const util = metrics?.inference?.utilization;
  return (
    <span className="pill" title="Активный бэкенд инференса">
      <span className="pill__key">Инференс</span>
      <strong>{backendLabel(metrics?.backend)}</strong>
      {typeof util === 'number' && <span className="pill__meter" style={{ ['--v' as string]: util }} aria-label={`загрузка ${Math.round(util * 100)} %`} />}
    </span>
  );
}

function Clock() {
  const now = useNow();
  return (
    <div className="clock" aria-label="Текущее время">
      <span className="clock__time mono">{formatTime(now)}</span>
      <span className="clock__date">{formatLongDate(now)}</span>
    </div>
  );
}

/** Браузер не даёт играть звук до первого клика — показываем это явно */
function SoundToggle({ sound }: { sound: boolean }) {
  useNow(); // перерисовка раз в секунду: состояние AudioContext не реактивно
  const blocked = sound && isAudioBlocked();
  const label = !sound ? 'Звук выкл.' : blocked ? 'Включить звук' : 'Звук вкл.';
  return (
    <button
      type="button"
      className={`btn btn--sm sound-toggle${sound && !blocked ? ' is-on' : ''}${blocked ? ' is-blocked' : ''}`}
      // при блокировке клик сам «размораживает» звук (глобальный обработчик жеста)
      onClick={blocked ? undefined : toggleSound}
      aria-pressed={sound}
      title={
        blocked
          ? 'Браузер ждёт первого клика, чтобы разрешить звук тревог'
          : sound
            ? 'Звук критических тревог включён'
            : 'Звук выключен'
      }
    >
      {sound ? <Bell size={15} aria-hidden="true" /> : <BellOff size={15} aria-hidden="true" />}
      {label}
    </button>
  );
}

/** Число неразобранных в заголовке вкладки — видно, даже когда пульт свёрнут */
function useDocumentTitle() {
  const unreviewed = useLive((s) => s.unreviewed);
  useEffect(() => {
    document.title = unreviewed ? `(${unreviewed}) Зазеркалье · Видеоаналитика ТБ` : 'Зазеркалье · Видеоаналитика ТБ';
  }, [unreviewed]);
}

function TopBar() {
  useDocumentTitle();
  const { pathname } = useLocation();
  const sound = useSettings((s) => s.sound);
  const meta = TITLES[`/${pathname.split('/')[1] ?? ''}`.replace(/\/$/, '') || '/'] ?? TITLES['/'];
  return (
    <header className="topbar">
      <div className="topbar__title">
        <h1 className="display">{meta.title}</h1>
        <p>{meta.sub}</p>
      </div>
      <div className="topbar__status">
        {api().mode === 'mock' && (
          <span className="pill pill--demo" title="Данные генерируются в браузере (VITE_MOCK=1 или ?mock=1)">
            <span className="pill__stripe hazard" />
            Демо-данные
          </span>
        )}
        <BackendPill />
        <ConnectionPill />
        <SoundToggle sound={sound} />
        <Clock />
      </div>
    </header>
  );
}

/** Глобальный баннер критической тревоги — виден на любом экране */
function AlarmBanner() {
  const alarm = useLive((s) => s.alarm);
  const now = useNow();
  const cameras = useCameraMap();
  const { openEvent } = useEventModal();
  if (!alarm) return null;
  const meta = getViolationMeta(alarm.type);
  const camera = cameras.get(alarm.camera_id);
  return (
    <div className="alarm" role="alert">
      <span className="alarm__stripe hazard--red" aria-hidden="true" />
      <Siren className="alarm__icon" size={22} aria-hidden="true" />
      <div className="alarm__text">
        <span className="alarm__kicker">Критическая тревога</span>
        <strong>{meta.title}</strong>
        <span className="alarm__where">
          {camera?.name ?? alarm.camera_id} · {alarm.site} · {formatTime(alarm.confirmed_at)}
        </span>
      </div>
      <span className="alarm__timer mono" title="Прошло с момента оповещения">
        {formatClock(now - alarm.confirmed_at)}
      </span>
      <button
        type="button"
        className="btn btn--primary"
        onClick={() => {
          openEvent(alarm.id);
          liveActions.dismissAlarm();
        }}
      >
        Открыть карточку
      </button>
      <button type="button" className="btn btn--ghost btn--icon" onClick={liveActions.dismissAlarm} aria-label="Скрыть баннер">
        <X size={18} />
      </button>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="app">
      <NavRail />
      <div className="main">
        <TopBar />
        <div className="main__body">
          <AlarmBanner />
          {children}
        </div>
      </div>
    </div>
  );
}
