import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// build goes straight into the Hono server's static folder
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: '../server/public', emptyOutDir: true },
  server: { proxy: { '/api': process.env.API_URL ?? 'http://localhost:3000' } },
});
