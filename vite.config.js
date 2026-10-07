import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { defineConfig } from 'vite';

/** Emits sw.js with every built file in its precache list, so the app reloads offline. */
function serviceWorker() {
  return {
    name: 'gemma-service-worker',
    apply: 'build',
    generateBundle(_options, bundle) {
      const files = ['./', 'favicon.svg', ...Object.keys(bundle).filter((file) => !file.endsWith('.map'))];
      const unique = [...new Set(files)];
      const version = createHash('sha256').update(unique.join('\n')).digest('hex').slice(0, 12);
      const source = readFileSync(new URL('./src/sw.js', import.meta.url), 'utf8')
        .replace('__VERSION__', version)
        .replace('__PRECACHE__', JSON.stringify(unique));
      this.emitFile({ type: 'asset', fileName: 'sw.js', source });
    },
  };
}

export default defineConfig({
  base: './',
  worker: { format: 'es' },
  build: { target: 'es2022' },
  // Transformers.js ships its own prebuilt ESM and wasm loader; pre-bundling breaks its asset URLs.
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
  server: { port: 5199, strictPort: true },
  preview: { port: 5200, strictPort: true },
  plugins: [serviceWorker()],
});
