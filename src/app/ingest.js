// Turns a dropped file (or a typed note) into a library item plus its embedded segments.
import { decodeAudio, decodeVideo, drawThumbnail, SAMPLE_RATE } from '../lib/media.js';
import { VIDEO_MAX_FRAMES, VIDEO_SEGMENT_SECONDS } from '../embed/config.js';
import { addItem } from './store.js';

const TEXT_EXTENSIONS = /\.(txt|md|markdown|csv|json|html?|xml|ya?ml|log|rtf)$/i;
const MEDIA_EXTENSIONS = {
  image: /\.(jpe?g|png|gif|webp|avif|bmp|heic)$/i,
  audio: /\.(wav|mp3|m4a|aac|ogg|oga|opus|flac|weba)$/i,
  video: /\.(mp4|m4v|webm|mov|mkv|ogv)$/i,
};

/** @returns {import('./store.js').Kind | null} */
export function detectKind(file) {
  const [major] = (file.type || '').split('/');
  if (major === 'image' || major === 'audio' || major === 'video') return major;
  if (major === 'text' || TEXT_EXTENSIONS.test(file.name)) return 'text';
  for (const [kind, pattern] of Object.entries(MEDIA_EXTENSIONS)) if (pattern.test(file.name)) return kind;
  return null;
}

const ENCODER_NEEDED = { image: 'vision', video: 'vision', audio: 'audio' };
const ENCODER_LABEL = { vision: 'image and video', audio: 'audio' };

/**
 * @param {File} file
 * @param {import('../embed/client.js').Embedder} embedder
 * @param {(stage: string) => void} onStage
 */
export async function ingestFile(file, embedder, onStage) {
  const kind = detectKind(file);
  if (!kind) throw new Error('Unsupported file type. Add images, audio, video or text files.');
  const encoder = ENCODER_NEEDED[kind];
  if (encoder && !embedder.info?.[encoder]) {
    throw new Error(`The ${ENCODER_LABEL[encoder]} encoder is off. Turn it on in Settings to add this file.`);
  }

  const base = {
    id: crypto.randomUUID(),
    kind,
    name: file.name,
    mime: file.type,
    size: file.size,
    addedAt: Date.now(),
  };

  onStage('Decoding');
  if (kind === 'text') {
    const text = (await file.text()).trim();
    if (!text) throw new Error('This file is empty.');
    return ingestNote({ title: file.name.replace(/\.[^.]+$/, ''), text, name: file.name, size: file.size }, embedder, onStage);
  }

  if (kind === 'image') {
    const bitmap = await createImageBitmap(file).catch(() => {
      throw new Error('This image format cannot be decoded by the browser.');
    });
    const { width, height } = bitmap;
    const preview = await drawThumbnail(bitmap, width, height, 480);
    bitmap.close();
    onStage('Embedding');
    const [segments] = await embedder.embedImages([file]);
    onStage('Saving');
    return save({ ...base, blob: file, preview, width, height }, segments, 'image');
  }

  if (kind === 'audio') {
    const pcm = await decodeAudio(file).catch(() => {
      throw new Error('This audio format cannot be decoded by the browser.');
    });
    const duration = pcm.length / SAMPLE_RATE;
    const preview = await drawWaveform(pcm);
    onStage(duration > 100 ? `Embedding ${Math.ceil(duration / 100)} chunks` : 'Embedding');
    const [segments] = await embedder.embedAudio([pcm]);
    onStage('Saving');
    return save({ ...base, blob: file, preview, duration }, segments, 'audio');
  }

  // Video: frames through the vision encoder, plus the soundtrack through the audio encoder when there is one.
  const video = await decodeVideo(file, { segmentSeconds: VIDEO_SEGMENT_SECONDS, maxFrames: VIDEO_MAX_FRAMES });
  const { duration, width, height, poster } = video;
  const soundtrack = embedder.info?.audio ? await decodeAudio(file).catch(() => null) : null;
  onStage(video.segments.length > 1 ? `Embedding ${video.segments.length} segments` : 'Embedding');
  const [frames] = await embedder.embedVideo([video]);
  const vectors = frames.map((s) => ({ ...s, modality: 'video' }));
  if (soundtrack && isAudible(soundtrack)) {
    onStage('Embedding soundtrack');
    const [sound] = await embedder.embedAudio([soundtrack]);
    vectors.push(...sound.map((s) => ({ ...s, modality: 'audio' })));
  }
  onStage('Saving');
  const item = { ...base, blob: file, preview: poster, duration, width, height };
  return { item, rows: await addItem(item, vectors) };
}

/**
 * @param {{ title?: string, text: string, name?: string, size?: number }} note
 * @param {import('../embed/client.js').Embedder} embedder
 */
export async function ingestNote({ title, text, name, size }, embedder, onStage = () => {}) {
  onStage('Embedding');
  const [segments] = await embedder.embedText([{ title, text }], 'document');
  onStage('Saving');
  const item = {
    id: crypto.randomUUID(),
    kind: 'text',
    name: name ?? (title || 'Note'),
    mime: 'text/plain',
    size: size ?? new Blob([text]).size,
    addedAt: Date.now(),
    title: title || '',
    text,
  };
  return save(item, segments, 'text');
}

async function save(item, segments, modality) {
  const rows = await addItem(item, segments.map((s) => ({ ...s, modality })));
  return { item, rows };
}

/** Skips silent soundtracks (screen recordings, muted clips) so they don't add noise vectors. */
function isAudible(pcm) {
  let sum = 0;
  const step = 16;
  for (let i = 0; i < pcm.length; i += step) sum += pcm[i] * pcm[i];
  return Math.sqrt(sum / Math.ceil(pcm.length / step)) > 0.005;
}

/** A bar waveform preview. Drawn in a mid grey that reads on light and dark themes. */
async function drawWaveform(pcm, width = 480, height = 200, bars = 96) {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  const per = Math.max(1, Math.floor(pcm.length / bars));
  const peaks = Array.from({ length: bars }, (_, b) => {
    let peak = 0;
    for (let i = b * per; i < Math.min(pcm.length, (b + 1) * per); i++) peak = Math.max(peak, Math.abs(pcm[i]));
    return peak;
  });
  const max = Math.max(...peaks, 1e-3);
  const gap = 2;
  const barWidth = (width - gap * (bars - 1)) / bars;
  ctx.fillStyle = '#8b97a8';
  peaks.forEach((peak, b) => {
    const h = Math.max(3, (peak / max) * (height - 16));
    ctx.beginPath();
    ctx.roundRect(b * (barWidth + gap), (height - h) / 2, barWidth, h, barWidth / 2);
    ctx.fill();
  });
  return canvas.convertToBlob({ type: 'image/png' });
}
