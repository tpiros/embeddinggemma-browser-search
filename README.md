# Gemma Search

Search your images, audio, video and notes with [EmbeddingGemma 2](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX), entirely in the browser. One model puts every kind of file in the same 768-dimensional space. A typed query can find a photo, and a voice clip can find a video. Nothing leaves the device, and there is no backend.

The write-up, with interactive figures for embeddings, token budgets and Matryoshka truncation, is at [tpiros.dev/blog/embeddinggemma-2-in-the-browser](https://tpiros.dev/blog/embeddinggemma-2-in-the-browser).

## Requirements

- Node.js 20.19+ or 22.12+ to build (Vite 8's requirement).
- A browser with WebGPU for the fast path (Chrome or Edge 113+). Others fall back to WASM, which works but is much slower and downloads about 1.2 GB of fp16 weights instead of 624 MB. See [NOTES.md](NOTES.md).

## Run it

```sh
npm install
npm run dev      # http://127.0.0.1:5199
npm run build    # static site in dist/, works from any static host
npm run preview  # serve dist/ locally
```

The first visit asks you to download the model: about 625 MB with every encoder, or 175 MB for text only. It is cached in the browser, and after that the app works offline.

## How it's put together

| Path | Role |
| --- | --- |
| `src/embed/worker.js` | Loads the model (WebGPU, falling back to WASM), applies task prefixes, chunks long inputs and packs batches under the WebGPU token budget. |
| `src/embed/client.js` | `Embedder`: `load()`, `embedText(texts, 'query' \| 'document')`, `embedImages()`, `embedAudio()`, `embedVideo()`, plus `progress` events. |
| `src/lib/media.js` | Main-thread decoding: audio to mono 16 kHz, video to sampled frames and a poster. |
| `src/app/store.js` | IndexedDB: items (file, preview, metadata) and vectors (one row per segment). |
| `src/app/vector-index.js` | Brute-force dot product with Matryoshka truncation and re-normalisation. |
| `src/app/ingest.js` | File to item and vectors, with thumbnails and waveforms. |
| `src/main.js` | UI. |
| `src/sw.js` | App-shell service worker, with its file list generated at build time. |
| `smoke.html` | Dev-only checks: `?test=text`, `?test=multimodal`, `?test=api` (add `&device=wasm` to skip WebGPU). |

See [NOTES.md](NOTES.md) for measured results and where the build differs from the brief.
