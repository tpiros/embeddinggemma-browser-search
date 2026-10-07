// Facts from the model card that the worker and UI share.

export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
export const DIM = 768;
export const DIMENSIONS = [768, 512, 256, 128];

// WebGPU: q4, except audio, which suffers most from 4-bit and gets q8.
// WASM: every quantized text and vision graph (q4, q4f16, q8) uses GatherBlockQuantized, which the
// ONNX Runtime WASM build has no kernel for (see NOTES.md), so text and vision fall back to fp16.
export const DTYPE = {
  webgpu: { model: 'q4', vision_encoder: 'q4', audio_encoder: 'q8' },
  wasm: { model: 'fp16', vision_encoder: 'fp16', audio_encoder: 'q8' },
};

// Download sizes (MB) of the ONNX graph + weights for the dtypes above, from the repo's file list.
export const DOWNLOAD_MB = {
  webgpu: { text: 175, vision: 109, audio: 340 },
  wasm: { text: 543, vision: 336, audio: 340 },
};

// File names Transformers.js uses for each dtype.
export const DTYPE_SUFFIX = { q4: '_q4', q8: '_quantized', fp16: '_fp16' };

// Token costs per input, from the model card.
export const TOKENS = {
  image: 282, // 280 soft tokens + begin/end markers
  videoFrame: 142,
  audioPerSecond: 25,
};

// Most tokens per batch. WebGPU kernels hit a dispatch limit above ~2,700. WASM has no such limit but
// runs out of its 4 GB heap on large fp16 batches, so it embeds one image at a time.
export const BATCH_BUDGET = { webgpu: 2_600, wasm: 300 };

export const AUDIO_CHUNK_SECONDS = 100;
export const VIDEO_SEGMENT_SECONDS = 60;
export const VIDEO_MAX_FRAMES = 16;
export const TEXT_CHUNK_TOKENS = 512;

/** Modalities and the encoders they need. Video frames go through the vision encoder. */
export const ENCODER_FOR = { text: null, image: 'vision', video: 'vision', audio: 'audio' };

export const prefixQuery = (text) => `task: search result | query: ${text}`;
export const prefixDocument = (text, title) => `title: ${title?.trim() || 'none'} | text: ${text}`;
