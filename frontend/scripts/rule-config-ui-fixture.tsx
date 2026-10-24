// Test-only rendered harness. No production routes/configuration are changed.
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { initApi } from '../src/api';
import { CameraEditor } from '../src/features/cameras/CamerasPage';
import { SchemaForm } from '../src/features/cameras/SchemaForm';

await initApi();
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
