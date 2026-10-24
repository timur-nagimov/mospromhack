import '@fontsource-variable/onest';
import '@fontsource-variable/tektur';
import '@fontsource-variable/jetbrains-mono';
import './styles/global.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { initApi } from './api';
import { App } from './App';

/** Адрес /login без hash (из документации или редиректа сервера) → #/login: маршруты в hash */
function normalizeLoginPath() {
  const { pathname, search, hash } = window.location;
  if (hash || !/\/login\/?$/.test(pathname)) return;
  window.history.replaceState(null, '', `${pathname.replace(/login\/?$/, '')}${search}#/login`);
}

async function bootstrap() {
  normalizeLoginPath();
  await initApi();
  const root = document.getElementById('root');
  if (!root) throw new Error('#root не найден');
  createRoot(root).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

void bootstrap();
