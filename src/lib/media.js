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
 * Samples frames from a video at `fps`, keeping at most `maxFrames` spread uniformly over the clip.
 * Frames are scaled so the longer side is at most `maxSide` (the processor resizes anyway).
 * @param {Blob} blob
 * @param {{ fps?: number, maxFrames?: number, maxSide?: number }} [options]
 * @returns {Promise<{ frames: { data: Uint8ClampedArray, width: number, height: number, timestamp: number }[], duration: number, width: number, height: number }>}
 */
export async function decodeVideo(blob, { fps = 1, maxFrames = 16, maxSide = 768 } = {}) {
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

    const times = sampleTimes(duration, fps, maxFrames);
    const scale = Math.min(1, maxSide / Math.max(videoWidth, videoHeight));
    const width = Math.max(1, Math.round(videoWidth * scale));
    const height = Math.max(1, Math.round(videoHeight * scale));
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });

    const frames = [];
    for (const t of times) {
      await seek(video, t);
      ctx.drawImage(video, 0, 0, width, height);
      frames.push({ data: ctx.getImageData(0, 0, width, height).data, width, height, timestamp: t });
    }
    return { frames, duration, width: videoWidth, height: videoHeight };
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

/** Times at `fps`, subsampled uniformly to `maxFrames` (same rule as the processor's `np.linspace`). */
function sampleTimes(duration, fps, maxFrames) {
  const all = [];
  for (let t = 0; t < duration; t += 1 / fps) all.push(t);
  if (all.length === 0) all.push(0);
  if (all.length <= maxFrames) return all;
  const step = (all.length - 1) / (maxFrames - 1);
  return Array.from({ length: maxFrames }, (_, i) => all[Math.round(i * step)]);
}

function seek(video, time) {
  return new Promise((resolve) => {
    video.addEventListener('seeked', resolve, { once: true });
    video.currentTime = time;
  });
}
