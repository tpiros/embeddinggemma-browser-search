// Per-browser preferences. localStorage can be unavailable (private mode, blocked storage), so every
// access is guarded and the defaults always work.
import { DIMENSIONS, DOWNLOAD_MB, DTYPE, DTYPE_SUFFIX, MODEL_ID } from '../embed/config.js';

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

/** The backend a device setting will most likely use. The worker makes the final call. */
export const likelyDevice = (device) => (device === 'auto' ? ('gpu' in navigator ? 'webgpu' : 'wasm') : device);

export function downloadMB({ vision, audio, device }) {
  const mb = DOWNLOAD_MB[likelyDevice(device)];
  return mb.text + (vision ? mb.vision : 0) + (audio ? mb.audio : 0);
}

/** Whether the model files for these encoders and device are already in Transformers.js's browser cache. */
export async function isModelCached({ vision, audio, device }) {
  try {
    if (!('caches' in self)) return false;
    const cache = await caches.open('transformers-cache');
    const urls = (await cache.keys()).map((r) => r.url).filter((u) => u.includes(MODEL_ID));
    const dtype = DTYPE[likelyDevice(device)];
    const has = (name, type) => urls.some((u) => u.endsWith(`/onnx/${name}${DTYPE_SUFFIX[type]}.onnx_data`));
    return (
      has('model', dtype.model) &&
      (!vision || has('vision_encoder', dtype.vision_encoder)) &&
      (!audio || has('audio_encoder', dtype.audio_encoder))
    );
  } catch {
    return false;
  }
}
