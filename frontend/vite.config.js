import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const siteUrl = (process.env.SITE_URL || '').replace(/\/$/, '')

const socialMeta = () => ({
  name: 'social-meta',
  transformIndexHtml: (html) =>
    html.replace(
      '<!--social-meta-->',
      siteUrl
        ? `<link rel="canonical" href="${siteUrl}/" />\n    <meta property="og:url" content="${siteUrl}/" />\n    <meta property="og:image" content="${siteUrl}/og.png" />\n    <meta name="twitter:image" content="${siteUrl}/og.png" />`
        : '',
    ),
})

export default defineConfig({
  plugins: [react(), socialMeta()],
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
