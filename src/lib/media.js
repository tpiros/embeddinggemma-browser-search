// Main-thread media decoding. Workers have neither AudioContext nor <video>, so the page decodes
// audio to mono 16 kHz PCM and samples video frames to RGBA, then transfers the raw data to the worker.

export const SAMPLE_RATE = 16_000;

/**
 * Decodes any browser-supported audio (or the audio track of a video) to mono 16 kHz.
 * @param {Blob} blob
 * @returns {Promise<Float32Array>}
 */
export async function decodeAudio(blob) {
  const bytes = await blob.arrayBuffer();
  // Decode at the file's own rate, then let OfflineAudioContext resample and downmix.
  const ctx = new AudioContext();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(bytes);
  } finally {
    await ctx.close();
  }
  return resampleToMono(decoded);
}

/**
 * @param {AudioBuffer} buffer
 * @returns {Promise<Float32Array>}
 */
export async function resampleToMono(buffer) {
  const length = Math.max(1, Math.ceil(buffer.duration * SAMPLE_RATE));
  const offline = new OfflineAudioContext(1, length, SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/**
 * Samples a video into segments of `segmentSeconds`, each with frames at `fps` subsampled to at most
 * `maxFrames`. Frames are scaled so the longer side is at most `maxSide` (the processor resizes anyway).
 * Also captures a small poster frame.
 * @param {Blob} blob
 * @param {{ fps?: number, segmentSeconds?: number, maxFrames?: number, maxSegments?: number, maxSide?: number }} [options]
 */
export async function decodeVideo(
  blob,
  { fps = 1, segmentSeconds = 60, maxFrames = 16, maxSegments = 20, maxSide = 768 } = {},
) {
  const url = URL.createObjectURL(blob);
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  try {
    await new Promise((resolve, reject) => {
      video.onloadeddata = resolve;
      video.onerror = () => reject(new Error(`Cannot decode video (${video.error?.message || 'unsupported format'})`));
      video.src = url;
    });
    const { duration, videoWidth, videoHeight } = video;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error('Video has no readable duration');

    const scale = Math.min(1, maxSide / Math.max(videoWidth, videoHeight));
    const width = Math.max(1, Math.round(videoWidth * scale));
    const height = Math.max(1, Math.round(videoHeight * scale));
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    // Even segments of roughly `segmentSeconds`, so 61 s is one segment rather than 60 s + a 1 s sliver.
    const segments = [];
    const count = Math.min(maxSegments, Math.max(1, Math.round(duration / segmentSeconds)));
    const length = duration / count;
    for (let s = 0; s < count; s++) {
      const start = s * length;
      const end = Math.min(duration, start + length);
      const frames = [];
      for (const t of sampleTimes(start, end, fps, maxFrames)) {
        await seek(video, t);
        ctx.drawImage(video, 0, 0, width, height);
        frames.push({ data: ctx.getImageData(0, 0, width, height).data, width, height, timestamp: t });
      }
      segments.push({ start, end, frames });
    }

    await seek(video, Math.min(1, duration / 2));
    const poster = await drawThumbnail(video, videoWidth, videoHeight);
    return { segments, duration, width: videoWidth, height: videoHeight, poster };
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

/** Times in [start, end) at `fps`, subsampled uniformly to `maxFrames` (as the processor's `np.linspace`). */
function sampleTimes(start, end, fps, maxFrames) {
  const all = [];
  for (let t = start; t < end; t += 1 / fps) all.push(t);
  if (all.length === 0) all.push(start);
  if (all.length <= maxFrames) return all;
  const step = (all.length - 1) / (maxFrames - 1);
  return Array.from({ length: maxFrames }, (_, i) => all[Math.round(i * step)]);
}

/**
 * Draws a source (image, bitmap, video) into a JPEG thumbnail whose longer side is `size`.
 * @returns {Promise<Blob>}
 */
export async function drawThumbnail(source, sourceWidth, sourceHeight, size = 320) {
  const scale = Math.min(1, size / Math.max(sourceWidth, sourceHeight));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(sourceWidth * scale)), Math.max(1, Math.round(sourceHeight * scale)));
  canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas.convertToBlob({ type: 'image/jpeg', quality: 0.82 });
}

function seek(video, time) {
  return new Promise((resolve) => {
    video.addEventListener('seeked', resolve, { once: true });
    video.currentTime = time;
  });
}
