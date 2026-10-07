// Main-thread facade over the embedding worker. Decodes media here (workers can't), then hands the
// worker raw PCM / frames. Every method resolves to one segment list per input:
// [{ vector: Float32Array(768), start?, end? }, ...]
import { decodeAudio, decodeVideo } from '../lib/media.js';
import { VIDEO_MAX_FRAMES, VIDEO_SEGMENT_SECONDS } from './config.js';

export class Embedder extends EventTarget {
  #worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  #pending = new Map();
  #nextId = 0;
  /** @type {{ device: string, vision: boolean, audio: boolean } | null} */
  info = null;

  constructor() {
    super();
    this.#worker.addEventListener('message', ({ data }) => {
      if (data.type === 'event') {
        this.dispatchEvent(new CustomEvent('progress', { detail: data.event }));
        return;
      }
      const pending = this.#pending.get(data.id);
      this.#pending.delete(data.id);
      if (data.error) pending?.reject(new Error(data.error));
      else pending?.resolve(data.result);
    });
    this.#worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'Embedding worker crashed');
      for (const { reject } of this.#pending.values()) reject(error);
      this.#pending.clear();
    });
  }

  #call(method, args, transfer = []) {
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#worker.postMessage({ id, method, args }, transfer);
    });
  }

  /**
   * @param {{ vision?: boolean, audio?: boolean, device?: 'auto' | 'webgpu' | 'wasm' }} options
   * @returns {Promise<{ device: string, vision: boolean, audio: boolean, fallbackReason?: string | null }>}
   */
  async load(options) {
    this.info = await this.#call('load', [options]);
    return this.info;
  }

  /** @param {(string | { title?: string, text: string })[]} texts @param {'query' | 'document'} kind */
  embedText(texts, kind) {
    return this.#call('embedText', [texts, kind]);
  }

  /** Convenience: one query vector. */
  async embedQuery(text) {
    const [[segment]] = await this.embedText([text], 'query');
    return segment.vector;
  }

  /** @param {Blob[]} files */
  embedImages(files) {
    return this.#call('embedImages', [files]);
  }

  /** @param {(Blob | Float32Array)[]} inputs files, or mono 16 kHz PCM already decoded */
  async embedAudio(inputs) {
    const clips = await Promise.all(inputs.map((x) => (x instanceof Float32Array ? x : decodeAudio(x))));
    return this.#call('embedAudio', [clips], clips.map((c) => c.buffer));
  }

  /** @param {(Blob | Awaited<ReturnType<typeof decodeVideo>>)[]} inputs files, or already-decoded videos */
  async embedVideo(inputs) {
    const decoded = await Promise.all(
      inputs.map((x) =>
        x instanceof Blob ? decodeVideo(x, { segmentSeconds: VIDEO_SEGMENT_SECONDS, maxFrames: VIDEO_MAX_FRAMES }) : x,
      ),
    );
    const videos = decoded.map(({ segments }) => ({ segments }));
    const transfer = videos.flatMap(({ segments }) => segments.flatMap(({ frames }) => frames.map((f) => f.data.buffer)));
    return this.#call('embedVideo', [videos], transfer);
  }
}
