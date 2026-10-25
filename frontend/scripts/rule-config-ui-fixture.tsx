// Test-only rendered harness. No production routes/configuration are changed.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { initApi } from '../src/api';
import { OPEN_ACCESS_USER, OPEN_CONFIG } from '../src/domain/auth';
import { CameraEditor } from '../src/features/cameras/CamerasPage';
import { SchemaForm } from '../src/features/cameras/SchemaForm';
import { authActions } from '../src/state/auth';

await initApi();
// Редактор проверяется без входа — как при GUARD_AUTH=0: все права администратора
authActions.signedIn(OPEN_CONFIG, OPEN_ACCESS_USER);
const input = (window as any).__fixture;
const queries = new QueryClient({ defaultOptions: { queries: { retry: false } } });
queries.setQueryData(['rule-types'], input.types ?? []);

function Fixture() {
  const [original, setOriginal] = useState(input.camera);
  const [value, setValue] = useState(input.value ?? {});
  (window as any).updateOriginal = setOriginal;
  (window as any).replaceFormValue = setValue;
  return input.schema ? <>
    <SchemaForm schema={input.schema} value={value} onChange={setValue} idPrefix="form" />
    <output id="draft-json">{JSON.stringify(value)}</output>
  </> : <CameraEditor original={original} isNew={false} existing={[original]} />;
}

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={queries}><MemoryRouter><Fixture /></MemoryRouter></QueryClientProvider>,
);
