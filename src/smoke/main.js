const params = new URLSearchParams(location.search);
const test = params.get('test') ?? 'text';
const device = params.get('device') ?? 'webgpu';
const out = document.querySelector('#log');

const log = (line) => {
  out.textContent += `${line}\n`;
  console.info(line);
};

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.addEventListener('message', ({ data }) => {
  if (data.type === 'log') log(data.line);
  if (data.type === 'done') window.smokeResult = data;
});

let media;
if (test === 'multimodal') {
  // Workers cannot decode audio or video, so the page does it and sends raw data.
  const { decodeSampleMedia } = await import('./decode.js');
  log('decoding sample media on the main thread…');
  media = await decodeSampleMedia();
}
worker.postMessage({ test, device, media });
