import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HashRouter, Navigate, Route, Routes, useLocation, useSearchParams } from 'react-router-dom';
import { useRuleTypes } from './api/queries';
import { ApiError } from './api/types';
import { AppShell } from './components/layout/AppShell';
import { safeNextPath, startPath } from './components/layout/sections';
import { ToastHost } from './components/ui/toast';
import { AnalyticsPage } from './features/analytics/AnalyticsPage';
import { AuditPage } from './features/audit/AuditPage';
import { AuthProvider } from './features/auth/AuthProvider';
import { BootScreen } from './features/auth/BootScreen';
import { RequirePermission } from './features/auth/Can';
import { LoginPage } from './features/auth/LoginPage';
import { NoAccess } from './features/auth/NoAccess';
import { CamerasPage } from './features/cameras/CamerasPage';
import { EventModalHost } from './features/event/EventModal';
import { EmployeesPage } from './features/employees/EmployeesPage';
import { DataPage } from './features/flywheel/DataPage';
import { JournalPage } from './features/journal/JournalPage';
import { MonitoringPage } from './features/monitoring/MonitoringPage';
import { PerformancePage } from './features/performance/PerformancePage';
import { UsersPage } from './features/users/UsersPage';
import { useAuth, useCan } from './state/auth';
import { LiveConnector } from './state/LiveConnector';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 4xx не повторяем: 401 и 403 не исправятся сами, а лишние 403 — лишние уведомления
      retry: (count, error) => !(error instanceof ApiError && error.status >= 400 && error.status < 500) && count < 2,
      retryDelay: (attempt) => Math.min(8000, 800 * 2 ** attempt),
      refetchOnWindowFocus: false,
    },
  },
});

/** Названия типов из плагинов нужны везде — грузим заранее (при праве cameras.view) */
function Prefetch() {
  useRuleTypes();
  return null;
}

/** Без входа любой адрес ведёт на /login; адрес запоминается в ?next= (кроме явного выхода) */
function ToLogin() {
  const { pathname, search } = useLocation();
  const reason = useAuth((s) => s.reason);
  const target = `${pathname}${search}`;
  const next = reason !== 'logout' && pathname !== '/' ? `?next=${encodeURIComponent(target)}` : '';
  return <Navigate to={`/login${next}`} replace />;
}

/** Вошли, а адрес — /login: туда, откуда отправили на вход, иначе на стартовую страницу роли */
function AfterLogin() {
  const [params] = useSearchParams();
  const can = useCan();
  return <Navigate to={safeNextPath(params.get('next')) ?? startPath(can) ?? '/'} replace />;
}

/** «/» — Мониторинг; без video.view — стартовая страница роли (обычно Аналитика) */
function Home() {
  const can = useCan();
  if (can('video.view')) return <MonitoringPage />;
  const start = startPath(can);
  return start && start !== '/' ? <Navigate to={start} replace /> : <NoAccess permission="video.view" />;
}

function Workspace() {
  return (
    <>
      <LiveConnector />
      <Prefetch />
      <AppShell>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<AfterLogin />} />
          <Route
            path="/journal"
            element={
              <RequirePermission perm="events.view">
                <JournalPage />
              </RequirePermission>
            }
          />
          <Route
            path="/analytics"
            element={
              <RequirePermission perm="analytics.view">
                <AnalyticsPage />
              </RequirePermission>
            }
          />
          <Route
            path="/cameras"
            element={
              <RequirePermission perm="cameras.view">
                <CamerasPage />
              </RequirePermission>
            }
          />
          <Route
            path="/cameras/:cameraId"
            element={
              <RequirePermission perm="cameras.view">
                <CamerasPage />
              </RequirePermission>
            }
          />
          <Route
            path="/performance"
            element={
              <RequirePermission perm="system.view">
                <PerformancePage />
              </RequirePermission>
            }
          />
          <Route
            path="/data"
            element={
              <RequirePermission perm="flywheel.review">
                <DataPage />
              </RequirePermission>
            }
          />
          <Route
            path="/employees"
            element={
              <RequirePermission perm="employees.view">
                <EmployeesPage />
              </RequirePermission>
            }
          />
          <Route
            path="/users"
            element={
              <RequirePermission perm="users.manage">
                <UsersPage />
              </RequirePermission>
            }
          />
          <Route
            path="/audit"
            element={
              <RequirePermission perm="audit.view">
                <AuditPage />
              </RequirePermission>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
      <EventModalHost />
    </>
  );
}

function AppRoutes() {
  const status = useAuth((s) => s.status);
  // смена пользователя — новое дерево: WS переподключается с новой сессией, состояние страниц сбрасывается
  const username = useAuth((s) => s.user?.username ?? '');

  if (status === 'loading' || status === 'error') return <BootScreen />;
  if (status === 'anonymous') {
    return (
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="*" element={<ToLogin />} />
      </Routes>
    );
  }
  return <Workspace key={username} />;
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* HashRouter: статику может раздавать любой сервер без SPA-fallback */}
      <HashRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
        <ToastHost />
      </HashRouter>
    </QueryClientProvider>
  );
}
