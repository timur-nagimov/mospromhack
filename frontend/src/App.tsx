import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom';
import { useRuleTypes } from './api/queries';
import { AppShell } from './components/layout/AppShell';
import { ToastHost } from './components/ui/toast';
import { canViewSection, roleHome, type SectionPermission } from './domain/roles';
import { AnalyticsPage } from './features/analytics/AnalyticsPage';
import { CamerasPage } from './features/cameras/CamerasPage';
import { EventModalHost } from './features/event/EventModal';
import { DataPage } from './features/flywheel/DataPage';
import { JournalPage } from './features/journal/JournalPage';
import { MonitoringPage } from './features/monitoring/MonitoringPage';
import { PerformancePage } from './features/performance/PerformancePage';
import { EmployeesPage } from './features/employees/EmployeesPage';
import { LiveConnector } from './state/LiveConnector';
import { useRole } from './state/role';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      retryDelay: (attempt) => Math.min(8000, 800 * 2 ** attempt),
      refetchOnWindowFocus: false,
    },
  },
});

/** Названия типов из плагинов нужны везде — грузим заранее */
function Prefetch() {
  useRuleTypes();
  return null;
}

function RolePage({ permission, children }: { permission: SectionPermission; children: ReactNode }) {
  const role = useRole();
  return canViewSection(role, permission) ? children : <Navigate to={roleHome(role)} replace />;
}

export function App() {
  const role = useRole();
  return (
    <QueryClientProvider client={queryClient}>
      {/* HashRouter: статику может раздавать любой сервер без SPA-fallback */}
      <HashRouter>
        <LiveConnector />
        <Prefetch />
        <AppShell>
          <Routes>
            <Route path="/" element={<RolePage permission="video.view"><MonitoringPage /></RolePage>} />
            <Route path="/journal" element={<RolePage permission="events.view"><JournalPage /></RolePage>} />
            <Route path="/analytics" element={<RolePage permission="analytics.view"><AnalyticsPage /></RolePage>} />
            <Route path="/cameras" element={<RolePage permission="cameras.view"><CamerasPage /></RolePage>} />
            <Route path="/cameras/:cameraId" element={<RolePage permission="cameras.view"><CamerasPage /></RolePage>} />
            <Route path="/performance" element={<RolePage permission="system.view"><PerformancePage /></RolePage>} />
            <Route path="/data" element={<RolePage permission="flywheel.review"><DataPage /></RolePage>} />
            <Route path="/employees" element={<RolePage permission="employees.view"><EmployeesPage /></RolePage>} />
            <Route path="*" element={<Navigate to={roleHome(role)} replace />} />
          </Routes>
        </AppShell>
        <EventModalHost />
        <ToastHost />
      </HashRouter>
    </QueryClientProvider>
  );
}
