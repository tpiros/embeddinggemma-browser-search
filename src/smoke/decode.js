import { decodeAudio, decodeVideo } from '../lib/media.js';

export const SAMPLES = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main';

const fetchBlob = async (name) => {
  const response = await fetch(`${SAMPLES}/${name}`);
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return response.blob();
};

export async function decodeSampleMedia() {
  const [image, audioBlob, videoBlob] = await Promise.all(
    ['cats.jpg', 'jfk.wav', 'sea-turtle.mp4'].map(fetchBlob),
  );
  return {
    image,
    audio: await decodeAudio(audioBlob),
    // 1 fps as on the card, capped at 16 frames for the WebGPU dispatch limit.
    video: await decodeVideo(videoBlob, { fps: 1, maxFrames: 16 }),
  };
}
