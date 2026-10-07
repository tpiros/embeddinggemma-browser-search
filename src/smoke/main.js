const params = new URLSearchParams(location.search);
const test = params.get('test') ?? 'text';
const device = params.get('device') ?? 'webgpu';
const out = document.querySelector('#log');

const log = (line) => {
  out.textContent += `${line}\n`;
  console.info(line);
};

if (test === 'api') {
  // Runs on the page: the Embedder client owns its own worker.
  const { runApiSmoke } = await import('./api.js');
  const started = performance.now();
  try {
    const pass = await runApiSmoke({ device: params.get('device') ?? 'auto', log });
    log(`${pass ? 'PASS' : 'FAIL'} in ${((performance.now() - started) / 1000).toFixed(1)}s`);
  } catch (error) {
    log(`ERROR: ${error?.stack ?? error}`);
  }
} else {
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.addEventListener('message', ({ data }) => {
    if (data.type === 'log') log(data.line);
  });
  let media;
  if (test === 'multimodal') {
    // Workers cannot decode audio or video, so the page does it and sends raw data.
    const { decodeSampleMedia } = await import('./decode.js');
    log('decoding sample media on the main thread…');
    media = await decodeSampleMedia();
  }
  worker.postMessage({ test, device, media });
}
