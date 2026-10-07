// Runs EmbeddingGemma 2. Owns prompts, chunking and batching; the page only sends raw media.
import {
  AutoConfig,
  AutoModel,
  AutoProcessor,
  RawImage,
  RawVideo,
  RawVideoFrame,
  env,
} from '@huggingface/transformers';
import ortMjs from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url';
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import {
  AUDIO_CHUNK_SECONDS,
  BATCH_BUDGET,
  DIM,
  DTYPE,
  MODEL_ID,
  TEXT_CHUNK_TOKENS,
  TOKENS,
  prefixDocument,
  prefixQuery,
} from './config.js';

// Serve the ONNX Runtime wasm from our own origin so the app works offline and on any static host.
const onnxWasm = env.backends.onnx.wasm;
onnxWasm.wasmPaths = { mjs: new URL(ortMjs, self.location.href).href, wasm: new URL(ortWasm, self.location.href).href };
// Threads need cross-origin isolation; without it ORT warns and falls back anyway.
onnxWasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

/** @type {{ model: any, processor: any, device: string, vision: boolean, audio: boolean } | null} */
let state = null;
let queue = Promise.resolve();

const emit = (event) => self.postMessage({ type: 'event', event });

// ---------- loading ----------

async function webgpuProblem() {
  if (!('gpu' in navigator)) return 'WebGPU is not available in this browser.';
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter ? null : 'No WebGPU adapter was found.';
  } catch (error) {
    return `WebGPU adapter request failed: ${error.message}`;
  }
}

async function loadOn(device, { vision, audio }) {
  const config = await AutoConfig.from_pretrained(MODEL_ID);
  if (!vision) config.vision_config = null;
  if (!audio) config.audio_config = null;
  const { model: text, vision_encoder, audio_encoder } = DTYPE[device];
  const dtype = { model: text };
  if (vision) dtype.vision_encoder = vision_encoder;
  if (audio) dtype.audio_encoder = audio_encoder;

  const progress_callback = (info) => {
    if (info.status === 'progress_total') {
      emit({ kind: 'download', progress: info.progress, loaded: info.loaded, total: info.total });
    }
  };
  const [processor, model] = await Promise.all([
    AutoProcessor.from_pretrained(MODEL_ID, { progress_callback }),
    AutoModel.from_pretrained(MODEL_ID, { config, device, dtype, progress_callback }),
  ]);
  return { processor, model };
}

async function load({ vision = true, audio = true, device = 'auto' } = {}) {
  if (state && state.vision === vision && state.audio === audio && (device === 'auto' || device === state.device)) {
    return describe();
  }
  if (state) {
    await state.model.dispose();
    state = null;
  }

  let fallbackReason = null;
  let target = device === 'wasm' ? 'wasm' : 'webgpu';
  if (target === 'webgpu') {
    fallbackReason = await webgpuProblem();
    if (fallbackReason) target = 'wasm';
  }

  emit({ kind: 'status', message: `Loading model on ${target}…` });
  let loaded;
  try {
    loaded = await loadOn(target, { vision, audio });
  } catch (error) {
    if (target !== 'webgpu') throw error;
    fallbackReason = `WebGPU failed to initialise (${error.message}).`;
    target = 'wasm';
    emit({ kind: 'status', message: 'WebGPU failed, retrying on WASM…' });
    loaded = await loadOn(target, { vision, audio });
  }

  state = { ...loaded, device: target, vision, audio };
  emit({ kind: 'status', message: 'Warming up…' });
  await run([prefixQuery('warm up')]);
  return { ...describe(), fallbackReason };
}

const describe = () => ({ device: state.device, vision: state.vision, audio: state.audio });

function requireLoaded(encoder) {
  if (!state) throw new Error('Model is not loaded.');
  if (encoder && !state[encoder]) throw new Error(`The ${encoder} encoder is not loaded. Enable it in settings.`);
}

// ---------- inference ----------

/** Runs one batch through the processor and model; returns n×768 vectors. */
async function run(...inputs) {
  const processed = await state.processor(...inputs);
  const output = await state.model(processed);
  const embedding = output.sentence_embedding;
  const data = new Float32Array(embedding.data); // copy before releasing GPU-backed tensors
  for (const value of [...Object.values(output), ...Object.values(processed)]) value?.dispose?.();
  return Array.from({ length: data.length / DIM }, (_, i) => data.subarray(i * DIM, (i + 1) * DIM));
}

/**
 * Packs items into batches under the device's token budget. A batch pads to its longest item,
 * so its cost is count × max(cost).
 */
function pack(items, cost) {
  const batches = [];
  let batch = [];
  let max = 0;
  for (const item of items) {
    const c = cost(item);
    const nextMax = Math.max(max, c);
    if (batch.length && (batch.length + 1) * nextMax > BATCH_BUDGET[state.device]) {
      batches.push(batch);
      batch = [];
      max = 0;
    }
    batch.push(item);
    max = Math.max(max, c);
  }
  if (batch.length) batches.push(batch);
  return batches;
}

/** Embeds a flat list of jobs batch by batch, reporting progress, and returns vectors in job order. */
async function embedJobs(jobs, cost, runBatch, task) {
  const vectors = new Array(jobs.length);
  const indexed = jobs.map((job, index) => ({ job, index }));
  let done = 0;
  for (const batch of pack(indexed, ({ job }) => cost(job))) {
    const out = await runBatch(batch.map(({ job }) => job));
    batch.forEach(({ index }, i) => (vectors[index] = out[i]));
    done += batch.length;
    emit({ kind: 'embed', task, done, total: jobs.length });
  }
  return vectors;
}

/** Groups per-job vectors back into per-input segment lists. */
function regroup(jobs, vectors, count) {
  const out = Array.from({ length: count }, () => []);
  jobs.forEach((job, i) => out[job.input].push({ vector: vectors[i], start: job.start, end: job.end }));
  return out;
}

// ---------- text ----------

const countTokens = (text) => state.processor.tokenizer.encode(text).length;
const sentences = new Intl.Segmenter(undefined, { granularity: 'sentence' });

/** Splits long text into chunks of about TEXT_CHUNK_TOKENS tokens on sentence boundaries. */
function chunkText(text) {
  if (countTokens(text) <= TEXT_CHUNK_TOKENS) return [text];
  const chunks = [];
  let current = '';
  for (const { segment } of sentences.segment(text)) {
    const candidate = current + segment;
    if (current && countTokens(candidate) > TEXT_CHUNK_TOKENS) {
      chunks.push(current.trim());
      current = segment;
    } else {
      current = candidate;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  return chunks;
}

/**
 * @param {(string | { title?: string, text: string })[]} texts
 * @param {'query' | 'document'} kind
 */
async function embedText(texts, kind) {
  requireLoaded();
  const jobs = [];
  texts.forEach((item, input) => {
    const { title, text } = typeof item === 'string' ? { text: item } : item;
    if (kind === 'query') {
      const prompt = prefixQuery(text.trim());
      jobs.push({ input, prompt, tokens: countTokens(prompt) });
      return;
    }
    chunkText(text.trim()).forEach((chunk, part) => {
      const prompt = prefixDocument(chunk, title);
      jobs.push({ input, prompt, start: part, end: part + 1, tokens: countTokens(prompt) });
    });
  });
  const vectors = await embedJobs(jobs, (j) => j.tokens, (batch) => run(batch.map((j) => j.prompt)), 'text');
  return regroup(jobs, vectors, texts.length);
}

// ---------- images ----------

/** @param {Blob[]} blobs */
async function embedImages(blobs) {
  requireLoaded('vision');
  const jobs = blobs.map((blob, input) => ({ input, blob }));
  const vectors = await embedJobs(
    jobs,
    () => TOKENS.image,
    async (batch) => {
      const images = await Promise.all(batch.map((j) => RawImage.fromBlob(j.blob)));
      // One list per image, otherwise the processor merges them into a single input.
      return run(null, images.map((image) => [image]));
    },
    'image',
  );
  return regroup(jobs, vectors, blobs.length);
}

// ---------- audio ----------

const SAMPLE_RATE = 16_000;

/** @param {Float32Array[]} clips mono 16 kHz */
async function embedAudio(clips) {
  requireLoaded('audio');
  const jobs = [];
  const chunk = AUDIO_CHUNK_SECONDS * SAMPLE_RATE;
  clips.forEach((pcm, input) => {
    const count = Math.max(1, Math.ceil(pcm.length / chunk));
    // Even chunks, so a 101 s clip becomes 2 × 50.5 s instead of 100 s + 1 s.
    const size = Math.ceil(pcm.length / count);
    for (let i = 0; i < count; i++) {
      const samples = pcm.subarray(i * size, Math.min(pcm.length, (i + 1) * size));
      jobs.push({ input, samples, start: (i * size) / SAMPLE_RATE, end: (i * size + samples.length) / SAMPLE_RATE });
    }
  });
  const vectors = await embedJobs(
    jobs,
    (j) => Math.ceil((j.samples.length / SAMPLE_RATE) * TOKENS.audioPerSecond) + 2,
    (batch) => run(null, null, batch.map((j) => [j.samples])),
    'audio',
  );
  return regroup(jobs, vectors, clips.length);
}

// ---------- video ----------

/**
 * @param {{ segments: { start: number, end: number, frames: { data: Uint8ClampedArray, width: number, height: number, timestamp: number }[] }[] }[]} videos
 */
async function embedVideo(videos) {
  requireLoaded('vision');
  const jobs = videos.flatMap(({ segments }, input) => segments.map((segment) => ({ input, ...segment })));
  const vectors = await embedJobs(
    jobs,
    (j) => j.frames.length * TOKENS.videoFrame,
    (batch) => {
      const raw = batch.map(({ frames, start, end }) => {
        const rawFrames = frames.map(
          ({ data, width, height, timestamp }) => new RawVideoFrame(new RawImage(data, width, height, 4), timestamp - start),
        );
        return [new RawVideo(rawFrames, end - start)];
      });
      return run(null, null, null, raw);
    },
    'video',
  );
  return regroup(jobs, vectors, videos.length);
}

// ---------- RPC ----------

const methods = { load, embedText, embedImages, embedAudio, embedVideo };

self.addEventListener('message', ({ data: { id, method, args } }) => {
  // One request at a time: the GPU is the bottleneck and concurrent sessions just contend.
  queue = queue.then(async () => {
    try {
      const result = await methods[method](...args);
      const transfer = collectBuffers(result);
      self.postMessage({ id, result }, transfer);
    } catch (error) {
      self.postMessage({ id, error: error?.message ?? String(error) });
    }
  });
});

/** Distinct buffers of the result's vectors, so they move instead of being copied. */
function collectBuffers(result) {
  if (!Array.isArray(result)) return [];
  const buffers = new Set();
  for (const segments of result) for (const { vector } of segments) buffers.add(vector.buffer);
  return [...buffers];
}
