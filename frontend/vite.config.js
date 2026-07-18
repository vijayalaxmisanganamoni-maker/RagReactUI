import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// dev server proxies API calls to the FastAPI backend on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ask': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
