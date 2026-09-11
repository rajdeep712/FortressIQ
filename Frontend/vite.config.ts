import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      // Forward API calls to the FastAPI backend. Using a same-origin proxy
      // (frontend :3000 -> backend :8000) keeps the httpOnly auth cookies on
      // one origin, which is required for cookie-based auth.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
