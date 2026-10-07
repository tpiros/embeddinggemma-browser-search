import { env } from '@huggingface/transformers';
import { runTextSmoke } from './text.js';

const log = (line) => self.postMessage({ type: 'log', line });

self.addEventListener('message', async ({ data }) => {
  const { test, device, media } = data;
  log(`transformers.js ${env.version}, device=${device}, test=${test}`);
  const started = performance.now();
  try {
    let pass;
    if (test === 'text') {
      pass = await runTextSmoke({ device, log });
    } else if (test === 'multimodal') {
      const { runMultimodalSmoke } = await import('./multimodal.js');
      pass = await runMultimodalSmoke({ device, log, media });
    } else {
      throw new Error(`Unknown test "${test}"`);
    }
    log(`${pass ? 'PASS' : 'FAIL'} in ${((performance.now() - started) / 1000).toFixed(1)}s`);
    self.postMessage({ type: 'done', pass });
  } catch (error) {
    log(`ERROR: ${error?.stack ?? error}`);
    self.postMessage({ type: 'done', pass: false, error: String(error?.message ?? error) });
  }
});
