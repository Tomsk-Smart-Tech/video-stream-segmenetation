// client/vite.config.ts
import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'assets'
        },
        {
          src: 'node_modules/onnxruntime-web/dist/*.mjs',
          dest: 'assets'
        }
      ]
    })
  ],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  assetsInclude: ['**/*.wasm', '**/*.mjs'],
  
  appType: 'spa',
  
  // MIME типы
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      if (req.url?.endsWith('.mjs')) {
        res.setHeader('Content-Type', 'application/javascript');
      } else if (req.url?.endsWith('.wasm')) {
        res.setHeader('Content-Type', 'application/wasm');
      }
      next();
    });
  }
});