// Per-browser preferences. localStorage can be unavailable (private mode, blocked storage), so every
// access is guarded and the defaults always work.
import { DIMENSIONS, DOWNLOAD_MB, MODEL_ID } from '../embed/config.js';

const KEY = 'gemma-search:settings';
const DEFAULTS = { dim: 768, vision: true, audio: true, device: 'auto' };

export function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}');
    const merged = { ...DEFAULTS, ...saved };
    if (!DIMENSIONS.includes(merged.dim)) merged.dim = DEFAULTS.dim;
    if (!['auto', 'webgpu', 'wasm'].includes(merged.device)) merged.device = DEFAULTS.device;
    return merged;
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(settings) {
  try {
    localStorage.setItem(KEY, JSON.stringify(settings));
  } catch {
    // Settings just won't persist this session.
  }
}

export const downloadMB = ({ vision, audio }) =>
  DOWNLOAD_MB.text + (vision ? DOWNLOAD_MB.vision : 0) + (audio ? DOWNLOAD_MB.audio : 0);

/** Whether the model files for these encoders are already in Transformers.js's browser cache. */
export async function isModelCached({ vision, audio }) {
  try {
    if (!('caches' in self)) return false;
    const cache = await caches.open('transformers-cache');
    const urls = (await cache.keys()).map((r) => r.url).filter((u) => u.includes(MODEL_ID));
    const has = (file) => urls.some((u) => u.endsWith(`/onnx/${file}`));
    return (
      has('model_q4.onnx_data') &&
      (!vision || has('vision_encoder_q4.onnx_data')) &&
      (!audio || has('audio_encoder_quantized.onnx_data'))
    );
  } catch {
    return false;
  }
}
