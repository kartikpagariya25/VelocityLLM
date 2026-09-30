import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { host: true, port: 5173 },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three', '@react-three/fiber'],
          motion: ['gsap', '@gsap/react', 'lenis'],
        },
      },
    },
  },
})
