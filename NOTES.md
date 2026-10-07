# Build notes

Versions: `@huggingface/transformers` 4.3.1, `onnxruntime-web` 1.31.0-dev.20260914-8d85527a0 (the version Transformers.js resolves, pinned so the bundled wasm matches), Vite 8.3, Chrome stable on macOS.

## What matched the model card

- `embedding_gemma2` is recognised by Transformers.js 4.3.1, so no workarounds were needed.
- Text example on WebGPU, q4: Mars 0.854, Saturn 0.783, Jupiter 0.752, Venus 0.684. These are the card's exact numbers.
- Multimodal example on WebGPU with `{ model: q4, vision_encoder: q4, audio_encoder: q8 }`:

  |       | cats  | speech | turtle |
  | ----- | ----- | ------ | ------ |
  | image | 0.740 | 0.462  | 0.508  |
  | audio | 0.504 | 0.767  | 0.478  |
  | video | 0.500 | 0.499  | 0.734  |

  The image row is identical to the card. Audio and video differ by under 0.01, because audio runs at q8 here (the card used q4) and the video frames come from our own decoder.

- Nested lists keep items separate: 10 copies of an image in one call give 10 identical vectors, split into a batch of 9 and a batch of 1.

## What didn't work as documented: q4 on WASM

The card's text example uses `device: "webgpu", // or "wasm"` with `dtype: "q4"`. On WASM (Transformers.js 4.3.1, onnxruntime-web 1.31.0-dev.20260914-8d85527a0, Chrome stable) session creation fails:

```
Can't create a session. ERROR_CODE: 9, ERROR_MESSAGE: Could not find an implementation for
GatherBlockQuantized(1) node with name '/model/embed_tokens/Gather_Quant'
```

q8 fails the same way (`node_embedding_Quant` in the vision encoder). Checking the graph files shows that every quantized text and vision variant (`_q4`, `_q4f16`, `_quantized`) contains `GatherBlockQuantized`. The fp32 and fp16 graphs don't, and no audio encoder variant does.

**What the app does:** WASM uses `{ model: fp16, vision_encoder: fp16, audio_encoder: q8 }`, which is about 1.2 GB against 624 MB for WebGPU. Batches are capped at 300 tokens, one image at a time, because a 9-image fp16 batch ran out of wasm memory (`OrtRun() ERROR_CODE: 6, std::bad_alloc`). With that, `smoke.html?test=api&device=wasm` passes, and its scores are within 0.01 of WebGPU:

|       | cats  | speech | turtle |
| ----- | ----- | ------ | ------ |
| image | 0.745 | 0.462  | 0.501  |
| audio | 0.499 | 0.774  | 0.480  |
| video | 0.498 | 0.491  | 0.728  |

It is slow: the full smoke run took 403 s on WASM against 26 s on WebGPU. Settings shows the larger WASM download size, and the setup screen shows it when WebGPU is missing.

## Deviations from the brief

- **JavaScript, not TypeScript.** The brief asked for strict TypeScript; the follow-up instruction was "use modern JS". The code is ES2022 modules with JSDoc types where they help. The "no `any`" acceptance criterion doesn't apply.
- **Media is decoded on the main thread.** Workers have neither `AudioContext` nor `<video>`, so `load_audio` / `load_video` can't run there. The page decodes audio to mono 16 kHz (`OfflineAudioContext`) and samples video frames to RGBA, then transfers raw buffers to the worker, which builds `RawImage` / `RawVideo`. The reference Space does the same thing. The `Embedder` client in `src/embed/client.js` still exposes `embedAudio(files)` / `embedVideo(files)` as the brief describes.
- **Video uses our own frame sampler** (`src/lib/media.js`) instead of `load_video`. It applies the same 1 fps rule and the card's `linspace` subsampling, but splits long videos into even segments of about 60 s with at most 16 frames each (16 × 142 = 2,272 tokens). `load_video` with `num_frames: 16` would cover a 10-minute video with 16 frames total.
- **Videos also embed their soundtrack** through the audio encoder when the audio encoder is loaded and the track isn't silent. That way a voice query or a spoken topic can find a video. Results say "Matched on the soundtrack" when that vector wins.
- **Long items have several vectors.** Audio over 100 s is split evenly (180 s gives 2 × 90 s), long notes are split at about 512 tokens on sentence boundaries, and video gets one vector per segment. Search scores an item by its best segment and shows the matching time range.

## WebGPU batch budget

On WebGPU the worker packs batches so that `count × max(item tokens) ≤ 2,600` (300 on WASM, see above). Padding makes a batch cost its longest item times the count. Costs come from the card: 282 per image (280 + markers), 142 per video frame, 25 per second of audio, and tokenizer counts for text. The 60 s video and the 3-minute audio file both ingest on WebGPU with no console errors.

## Offline

- ONNX Runtime's wasm is bundled from `node_modules` and served from our origin. By default Transformers.js points at jsDelivr.
- `dist/sw.js` is generated at build time with the hashed file list. It precaches the app shell; model weights stay in Transformers.js's own `transformers-cache`.
- Checked by replacing the worker's `env.fetch` with a function that throws and reports each call. A cached load made zero fetches and searches worked. A production build reloaded with DevTools set to offline also loaded and searched.

## Results on the sample set

| Query | Top result (768) | Top result (256) |
| --- | --- | --- |
| cats sleeping | cats.jpg 0.754 | cats.jpg 0.779 |
| a speech about serving your country | jfk.wav 0.755 | jfk.wav 0.780 |
| a turtle swimming in the ocean | sea-turtle.mp4 0.734 | sea-turtle.mp4 0.763 |
| piano music | piano.wav 0.699 | piano.wav 0.753 |
| a lake in the mountains | moraine-lake.png 0.708 | moraine-lake.png 0.728 |

The top results also hold at 128.

## Not verified here

- Voice search with a real microphone. The automated browser has no mic. The recorded clip goes through the same decode and embed path as dropped audio files, which is tested.
- Browsers other than Chrome.
