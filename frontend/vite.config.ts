import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// Адрес бэкенда для dev-прокси. Переопределяется переменной GUARD_API_URL
// (в окружении или в .env.local), например GUARD_API_URL=http://192.168.1.10:8000
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.GUARD_API_URL || 'http://localhost:8000';

  return {
    plugins: [react()],
    // Относительные пути к ассетам: dist можно раздавать с / или из подкаталога
    base: './',
    server: {
      port: 5173,
      strictPort: false,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
        '/ws': { target, changeOrigin: true, ws: true },
      },
    },
    preview: {
      port: 4173,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
        '/ws': { target, changeOrigin: true, ws: true },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      chunkSizeWarningLimit: 900,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ['react', 'react-dom', 'react-router-dom', '@tanstack/react-query'],
            charts: ['recharts'],
          },
        },
      },
    },
  };
});
