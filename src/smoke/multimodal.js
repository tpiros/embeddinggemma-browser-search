// Step 2: the model card's image / audio / video example. Media arrives pre-decoded from the page.
import { AutoModel, AutoProcessor, RawImage, RawVideo, RawVideoFrame, cat, matmul } from '@huggingface/transformers';

const MODEL_ID = 'onnx-community/embeddinggemma-2-ONNX';

export async function runMultimodalSmoke({ device, log, media }) {
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  const model = await AutoModel.from_pretrained(MODEL_ID, {
    device,
    dtype: { model: 'q4', vision_encoder: 'q4', audio_encoder: 'q8' },
  });
  const embed = async (...inputs) => (await model(await processor(...inputs))).sentence_embedding;

  const queries = [
    'task: search result | query: cats sleeping on a couch',
    "task: search result | query: a president's speech about serving your country",
    'task: search result | query: a turtle swimming in the ocean',
  ];
  const queryEmbeddings = await embed(queries);

  const image = await RawImage.fromBlob(media.image);
  const frames = media.video.segments[0].frames.map(
    ({ data, width, height, timestamp }) => new RawVideoFrame(new RawImage(data, width, height, 4), timestamp),
  );
  const video = new RawVideo(frames, media.video.duration);
  log(`audio ${(media.audio.length / 16000).toFixed(1)}s, video ${media.video.duration.toFixed(1)}s → ${frames.length} frames`);

  const mediaEmbeddings = cat([
    await embed(null, image),
    await embed(null, null, media.audio),
    await embed(null, null, null, video),
  ]);

  const scores = (await matmul(mediaEmbeddings, queryEmbeddings.transpose(1, 0))).tolist();
  log('         cats    speech  turtle');
  ['image', 'audio', 'video'].forEach((name, i) => log(`${name.padEnd(6)} ${scores[i].map((x) => x.toFixed(3)).join('  ')}`));

  await model.dispose();
  // Each input should score highest against its own query.
  return scores.every((row, i) => row.indexOf(Math.max(...row)) === i);
}
