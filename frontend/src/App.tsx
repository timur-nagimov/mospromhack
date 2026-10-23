import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom';
import { useRuleTypes } from './api/queries';
import { AppShell } from './components/layout/AppShell';
import { ToastHost } from './components/ui/toast';
import { AnalyticsPage } from './features/analytics/AnalyticsPage';
import { CamerasPage } from './features/cameras/CamerasPage';
import { EventModalHost } from './features/event/EventModal';
import { JournalPage } from './features/journal/JournalPage';
import { MonitoringPage } from './features/monitoring/MonitoringPage';
import { PerformancePage } from './features/performance/PerformancePage';
import { LiveConnector } from './state/LiveConnector';

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

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* HashRouter: статику может раздавать любой сервер без SPA-fallback */}
      <HashRouter>
        <LiveConnector />
        <Prefetch />
        <AppShell>
          <Routes>
            <Route path="/" element={<MonitoringPage />} />
            <Route path="/journal" element={<JournalPage />} />
            <Route path="/analytics" element={<AnalyticsPage />} />
            <Route path="/cameras" element={<CamerasPage />} />
            <Route path="/cameras/:cameraId" element={<CamerasPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </AppShell>
        <EventModalHost />
        <ToastHost />
      </HashRouter>
    </QueryClientProvider>
  );
}
