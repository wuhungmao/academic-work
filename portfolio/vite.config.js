import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // GitHub Pages serves the site under /academic-work/; Vercel serves it from the root.
  base: process.env.VERCEL ? '/' : '/academic-work/',
})
