// Step 1: the model card's text search example, run inside a worker.
import { pipeline, matmul } from '@huggingface/transformers';

export async function runTextSmoke({ device, log }) {
  const extractor = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-2-ONNX', {
    device,
    dtype: 'q4',
  });

  const query = 'task: search result | query: Which planet is known as the Red Planet?';
  const documents = [
    "title: none | text: Venus is often called Earth's twin because of its similar size and proximity.",
    'title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet.',
    'title: none | text: Jupiter, the largest planet in our solar system, has a prominent red spot.',
    'title: none | text: Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
  ];

  const embeddings = await extractor([query, ...documents], { pooling: 'mean', normalize: true });
  const queryEmbedding = embeddings.slice([0, 1]);
  const documentEmbeddings = embeddings.slice([1, null]);
  const scores = (await matmul(queryEmbedding, documentEmbeddings.transpose(1, 0))).tolist()[0];

  const ranking = scores
    .map((score, i) => ({ score: Number(score.toFixed(3)), document: documents[i] }))
    .sort((a, b) => b.score - a.score);
  log(JSON.stringify(ranking, null, 2));

  const pass = ranking[0].document.includes('Mars');
  await extractor.dispose();
  return pass;
}
