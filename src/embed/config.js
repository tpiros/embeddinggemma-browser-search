// Facts from the model card that the worker and UI share.

export const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';
export const DIM = 768;
export const DIMENSIONS = [768, 512, 256, 128];

// Audio suffers most from 4-bit, so it gets q8.
export const DTYPE = { model: 'q4', vision_encoder: 'q4', audio_encoder: 'q8' };

// Download sizes (MB) of the ONNX graph + weights for the dtypes above, from the repo's file list.
export const DOWNLOAD_MB = { text: 175, vision: 109, audio: 340 };

// Token costs and the WebGPU batch ceiling. Kernels hit a dispatch limit above ~2,700 tokens.
export const TOKENS = {
  image: 282, // 280 soft tokens + begin/end markers
  videoFrame: 142,
  audioPerSecond: 25,
  batchBudget: 2_600,
};

export const AUDIO_CHUNK_SECONDS = 100;
export const VIDEO_SEGMENT_SECONDS = 60;
export const VIDEO_MAX_FRAMES = 16;
export const TEXT_CHUNK_TOKENS = 512;

/** Modalities and the encoders they need. Video frames go through the vision encoder. */
export const ENCODER_FOR = { text: null, image: 'vision', video: 'vision', audio: 'audio' };

export const prefixQuery = (text) => `task: search result | query: ${text}`;
export const prefixDocument = (text, title) => `title: ${title?.trim() || 'none'} | text: ${text}`;
