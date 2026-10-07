import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022' },
  // Transformers.js ships its own prebuilt ESM and wasm loader; pre-bundling breaks its asset URLs.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  server: { port: 5199, strictPort: true },
});
