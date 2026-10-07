// Step 3: exercises the Embedder client + worker API from the page.
import { Embedder } from '../embed/client.js';
import { SAMPLES } from './decode.js';

const dot = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);
const blob = async (name) => (await fetch(`${SAMPLES}/${name}`)).blob();

export async function runApiSmoke({ device, log }) {
  const embedder = new Embedder();
  let lastPct = -10;
  embedder.addEventListener('progress', ({ detail }) => {
    if (detail.kind === 'download' && detail.progress - lastPct >= 10) {
      lastPct = detail.progress;
      log(`download ${detail.progress.toFixed(0)}%`);
    } else if (detail.kind !== 'download') {
      log(`event ${JSON.stringify(detail)}`);
    }
  });
  const t0 = performance.now();
  log(`load → ${JSON.stringify(await embedder.load({ vision: true, audio: true, device }))} in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const queries = ['cats sleeping on a couch', "a president's speech about serving your country", 'a turtle swimming in the ocean'];
  const q = (await embedder.embedText(queries, 'query')).map(([s]) => s.vector);

  const [cats, jfk, turtle] = await Promise.all(['cats.jpg', 'jfk.wav', 'sea-turtle.mp4'].map(blob));
  const checks = [];

  // Nine copies + one: verifies batching (9 per batch) and that items aren't merged.
  const images = await embedder.embedImages([...Array(9).fill(cats), cats]);
  checks.push(['10 images → 10 vectors', images.length === 10 && images.every((s) => s.length === 1)]);
  checks.push(['batched copies identical', Math.abs(dot(images[0][0].vector, images[9][0].vector) - 1) < 1e-3]);

  const [audio] = await embedder.embedAudio([jfk]);
  const [video] = await embedder.embedVideo([turtle]);
  const rows = { image: images[0][0].vector, audio: audio[0].vector, video: video[0].vector };
  for (const [name, v] of Object.entries(rows)) log(`${name.padEnd(6)} ${q.map((x) => dot(v, x).toFixed(3)).join('  ')}`);
  Object.values(rows).forEach((v, i) => {
    const scores = q.map((x) => dot(v, x));
    checks.push([`${Object.keys(rows)[i]} matches own query`, scores.indexOf(Math.max(...scores)) === i]);
  });

  // A long document is chunked; a short one isn't.
  const long = Array.from({ length: 120 }, (_, i) => `Sentence number ${i} talks about the red planet Mars and its dusty surface.`).join(' ');
  const docs = await embedder.embedText([{ title: 'Mars', text: long }, 'A short note about turtles.'], 'document');
  log(`document chunks: ${docs.map((d) => d.length).join(', ')}`);
  checks.push(['long doc chunked', docs[0].length > 1 && docs[1].length === 1]);
  checks.push(['norm ≈ 1', Math.abs(dot(q[0], q[0]) - 1) < 1e-3]);

  for (const [name, ok] of checks) log(`${ok ? '✓' : '✗'} ${name}`);
  return checks.every(([, ok]) => ok);
}
