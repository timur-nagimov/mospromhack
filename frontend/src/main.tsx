import '@fontsource-variable/onest';
import '@fontsource-variable/tektur';
import '@fontsource-variable/jetbrains-mono';
import './styles/global.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { initApi } from './api';
import { App } from './App';

async function bootstrap() {
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
