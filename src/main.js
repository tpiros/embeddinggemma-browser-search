import './style.css';
import { Embedder } from './embed/client.js';
import { DIMENSIONS, DOWNLOAD_MB } from './embed/config.js';
import * as store from './app/store.js';
import { VectorIndex } from './app/vector-index.js';
import { detectKind, ingestFile, ingestNote } from './app/ingest.js';
import { VoiceRecorder } from './app/recorder.js';
import { downloadMB, isModelCached, loadSettings, saveSettings } from './app/settings.js';
import { fetchSampleFiles, SAMPLE_NOTES } from './app/samples.js';

// ---------- tiny DOM helpers ----------

const $ = (selector) => document.querySelector(selector);

/** Creates an element. `props` keys starting with "on" become listeners; `class`, `data-*`, `aria-*` become attributes. */
function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else if (key === 'class' || key.includes('-') || key === 'for' || key === 'role') el.setAttribute(key, value);
    else el[key] = value;
  }
  el.append(...children.flat().filter((c) => c != null && c !== false));
  return el;
}

const KIND_LABEL = { text: 'Note', image: 'Image', audio: 'Audio', video: 'Video' };

const formatTime = (seconds) => {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
};

const formatBytes = (bytes) =>
  bytes < 1024 ** 2 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 ** 2).toFixed(1)} MB`;

// ---------- state ----------

const embedder = new Embedder();
const recorder = new VoiceRecorder();
const settings = loadSettings();

const state = {
  /** @type {Map<string, import('./app/store.js').Item>} */
  items: new Map(),
  index: new VectorIndex([], settings.dim),
  /** @type {null | { label: string, image?: string, vector: Float32Array, exclude?: string, elapsed?: number }} */
  query: null,
  filter: 'all',
  /** @type {Promise<void> | null} resolves when the model is ready */
  model: null,
  ready: false,
};

const previewUrls = new Map();
function previewUrl(item) {
  if (!item.preview) return null;
  if (!previewUrls.has(item.id)) previewUrls.set(item.id, URL.createObjectURL(item.preview));
  return previewUrls.get(item.id);
}

// ---------- status, notices, loader ----------

function setStatus(text, tone = 'busy') {
  $('#status-text').textContent = text;
  $('#status').dataset.state = tone;
}

function notify(message, tone = 'info') {
  $('#notice-text').textContent = message;
  $('#notice').dataset.tone = tone;
  $('#notice').hidden = false;
}
$('#notice-dismiss').addEventListener('click', () => ($('#notice').hidden = true));

let lastDownload = 0;
embedder.addEventListener('progress', ({ detail }) => {
  if (detail.kind === 'download') {
    // Only show the bar for real downloads; cached loads finish in a blink.
    const now = performance.now();
    if (now - lastDownload < 100 && detail.progress < 100) return;
    lastDownload = now;
    $('#loader').hidden = detail.progress >= 100;
    $('#loader-fill').style.width = `${detail.progress}%`;
    $('#loader-pct').textContent = `${formatBytes(detail.loaded)} of ${formatBytes(detail.total)}`;
    setStatus(`Downloading model ${Math.floor(detail.progress)}%`);
  } else if (detail.kind === 'status') {
    setStatus(detail.message.replace('…', ''));
  }
});

// ---------- model ----------

function encoderRows(container, values, onChange) {
  const total = h('div', { class: 'encoder-total' });
  const rows = [
    ['text', 'Text', 'Notes and typed searches', DOWNLOAD_MB.text, true],
    ['vision', 'Images and video', 'Photos, video frames, image search', DOWNLOAD_MB.vision, false],
    ['audio', 'Audio', 'Sound files, soundtracks, voice search', DOWNLOAD_MB.audio, false],
  ].map(([key, label, detail, mb, locked]) =>
    h(
      'label',
      { class: 'encoder', 'data-locked': locked ? '' : null },
      h('input', {
        type: 'checkbox',
        name: key,
        checked: locked || values[key],
        disabled: locked,
        onchange: (event) => {
          values[key] = event.target.checked;
          update();
          onChange?.();
        },
      }),
      h('span', {}, label, h('small', {}, detail)),
      h('span', { class: 'size' }, `${mb} MB`),
    ),
  );
  const update = () => (total.replaceChildren(h('span', {}, 'Download'), h('span', {}, `${downloadMB(values)} MB`)));
  update();
  container.replaceChildren(...rows, total);
  return update;
}

function showSetup() {
  const choice = { vision: settings.vision, audio: settings.audio };
  const button = $('#setup-download');
  const label = () => (button.textContent = `Download model (${downloadMB(choice)} MB)`);
  encoderRows($('#setup-encoders'), choice, label);
  label();
  button.onclick = () => {
    Object.assign(settings, choice);
    saveSettings(settings);
    $('#setup').hidden = true;
    loadModel();
  };
  $('#setup').hidden = false;
  setStatus('Model not downloaded', 'idle');
}

function loadModel() {
  state.ready = false;
  $('#loader-label').textContent = 'Downloading model';
  state.model = (async () => {
    try {
      const info = await embedder.load({ vision: settings.vision, audio: settings.audio, device: settings.device });
      $('#loader').hidden = true;
      if (info.fallbackReason) notify(`${info.fallbackReason} Running on WASM instead, which is slower.`);
      state.ready = true;
      setStatus(`Ready on ${info.device === 'webgpu' ? 'WebGPU' : 'WASM'}`, 'ready');
      updateQueryControls();
    } catch (error) {
      $('#loader').hidden = true;
      setStatus('Model failed to load', 'error');
      const offline = !navigator.onLine ? ' You appear to be offline, and the model is not cached yet.' : '';
      notify(`The model could not load: ${error.message}.${offline}`, 'error');
      showSetup();
      throw error;
    }
  })();
  state.model.catch(() => {}); // surfaced above; callers awaiting it handle their own failure
  return state.model;
}

/** Waits for the model, or explains why it can't be used yet. */
async function requireModel() {
  if (!state.model) {
    notify('Download the model first. It takes a minute and only happens once.');
    $('#setup').scrollIntoView({ behavior: 'smooth', block: 'center' });
    throw new Error('Model not loaded');
  }
  await state.model;
}

function updateQueryControls() {
  const info = embedder.info;
  $('#image-query-label').hidden = Boolean(info && !info.vision);
  $('#mic').hidden = Boolean(info && !info.audio) || !('mediaDevices' in navigator);
}

// ---------- rendering ----------

function kindOf(itemId) {
  return state.items.get(itemId)?.kind;
}

function matchLabel(item, segment, segmentCount) {
  if (!segment) return null;
  if (item.kind === 'video' && segment.modality === 'audio') return 'Matched on the soundtrack';
  if ((item.kind === 'audio' || item.kind === 'video') && segmentCount > 1 && segment.start != null) {
    return `Best match ${formatTime(segment.start)}–${formatTime(segment.end)}`;
  }
  if (item.kind === 'text' && segmentCount > 1 && segment.start != null) return `Best match in passage ${segment.start + 1}`;
  return null;
}

function thumbFor(item) {
  if (item.kind === 'text') {
    return h('div', { class: 'thumb text' }, h('p', {}, item.text));
  }
  const badge = item.duration ? h('span', { class: 'badge' }, formatTime(item.duration)) : null;
  return h('div', { class: `thumb ${item.kind}` }, h('img', { src: previewUrl(item), alt: '', loading: 'lazy' }), badge);
}

function card(item, result, rank) {
  const segmentCount = result ? state.index.rowsFor(item.id).filter((r) => r.modality === result.segment.modality).length : 0;
  const detail = result ? matchLabel(item, result.segment, segmentCount) : null;
  const meta = [KIND_LABEL[item.kind], item.duration ? formatTime(item.duration) : null, detail].filter(Boolean).join(', ');
  return h(
    'li',
    { class: `card ${item.kind}` },
    rank ? h('span', { class: 'rank', 'aria-label': `Rank ${rank}` }, String(rank)) : null,
    h(
      'button',
      { class: 'card-button', type: 'button', onclick: () => openViewer(item, result?.segment) },
      thumbFor(item),
      h('span', { class: 'card-title' }, item.title || item.name),
    ),
    h('div', { class: 'card-meta' }, h('i', { class: 'swatch' }), meta),
    result
      ? h(
          'div',
          { class: 'score', title: 'Cosine similarity' },
          h('div', { class: 'bar' }, h('div', { class: 'bar-fill', style: `width:${Math.max(0, result.score) * 100}%` })),
          h('output', {}, result.score.toFixed(3)),
        )
      : null,
  );
}

function render() {
  const grid = $('#grid');
  const kinds = state.filter === 'all' ? null : new Set([state.filter]);
  const total = state.items.size;
  $('#empty').hidden = total > 0;
  $('#add-files-label').hidden = $('#add-note').hidden = total === 0;

  if (state.query) {
    const started = performance.now();
    const results = state.index.search(state.query.vector, { k: 10, kinds, kindOf, exclude: state.query.exclude });
    const searchMs = performance.now() - started;
    $('#view-title').textContent = `Top matches for ${state.query.label}`;
    $('#view-meta').textContent =
      `${results.length} of ${total} items, ${state.index.dim} dimensions, ` +
      `${state.query.elapsed != null ? `embedded in ${Math.round(state.query.elapsed)} ms, ` : ''}ranked in ${searchMs.toFixed(1)} ms`;
    grid.className = 'grid ranked';
    grid.replaceChildren(...results.map((r, i) => card(state.items.get(r.itemId), r, i + 1)));
    return;
  }

  const items = [...state.items.values()]
    .filter((item) => !kinds || kinds.has(item.kind))
    .sort((a, b) => b.addedAt - a.addedAt);
  $('#view-title').textContent = 'Library';
  $('#view-meta').textContent = total ? `${items.length} ${items.length === 1 ? 'item' : 'items'}` : '';
  grid.className = 'grid';
  grid.replaceChildren(...items.map((item) => card(item)));
}

// ---------- search ----------

function setQueryChip({ label, image } = {}) {
  const chip = $('#query-chip');
  chip.hidden = !label;
  $('#query-chip-label').textContent = label ?? '';
  const img = $('#query-chip-img');
  img.hidden = !image;
  if (image) img.src = image;
  else img.removeAttribute('src');
  $('#query-input').placeholder = label ? '' : "Describe what you're looking for";
}

function clearQuery() {
  if (state.query?.image) URL.revokeObjectURL(state.query.image);
  state.query = null;
  setQueryChip();
  $('#query-input').value = '';
  render();
}

/** @param {{ label: string, chip?: string, image?: string, exclude?: string, embed: () => Promise<Float32Array> }} query */
async function runQuery({ label, chip, image, exclude, embed }) {
  try {
    await requireModel();
  } catch {
    if (image) URL.revokeObjectURL(image);
    return;
  }
  setStatus('Searching');
  const started = performance.now();
  try {
    const vector = await embed();
    if (state.query?.image && state.query.image !== image) URL.revokeObjectURL(state.query.image);
    state.query = { label, image, vector, exclude, elapsed: performance.now() - started };
    setQueryChip(chip ? { label: chip, image } : {});
    render();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (error) {
    notify(`Search failed: ${error.message}`, 'error');
  } finally {
    setStatus(`Ready on ${embedder.info?.device === 'webgpu' ? 'WebGPU' : 'WASM'}`, 'ready');
  }
}

$('#query-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = $('#query-input').value.trim();
  if (!text) {
    if (!state.query) return;
    return clearQuery();
  }
  runQuery({ label: `“${text}”`, embed: () => embedder.embedQuery(text) });
});

$('#query-chip-clear').addEventListener('click', clearQuery);

$('#image-query').addEventListener('change', (event) => {
  const [file] = event.target.files;
  event.target.value = '';
  if (!file) return;
  $('#query-input').value = '';
  runQuery({
    label: 'your image',
    chip: file.name,
    image: URL.createObjectURL(file),
    embed: async () => (await embedder.embedImages([file]))[0][0].vector,
  });
});

const MAX_RECORDING_S = 30;
let recordingTimer = 0;

async function stopRecording() {
  clearInterval(recordingTimer);
  const mic = $('#mic');
  mic.setAttribute('aria-pressed', 'false');
  $('#mic-timer').textContent = '';
  $('#mic-label').textContent = 'Search with your voice';
  const blob = await recorder.stop();
  $('#query-input').value = '';
  runQuery({
    label: 'your voice clip',
    chip: 'Voice clip',
    embed: async () => (await embedder.embedAudio([blob]))[0][0].vector,
  });
}

$('#mic').addEventListener('click', async () => {
  if (recorder.recording) return stopRecording();
  try {
    await requireModel();
    await recorder.start();
  } catch (error) {
    if (error.name === 'NotAllowedError') notify('Microphone access was blocked. Allow it in the browser’s site settings to search by voice.', 'error');
    else if (error.message !== 'Model not loaded') notify(`Could not start recording: ${error.message}`, 'error');
    return;
  }
  const mic = $('#mic');
  mic.setAttribute('aria-pressed', 'true');
  $('#mic-label').textContent = 'Stop recording and search';
  const started = Date.now();
  const tick = () => {
    const elapsed = (Date.now() - started) / 1000;
    $('#mic-timer').textContent = formatTime(elapsed);
    if (elapsed >= MAX_RECORDING_S) stopRecording();
  };
  tick();
  recordingTimer = setInterval(tick, 250);
});

for (const button of document.querySelectorAll('#filters button')) {
  button.addEventListener('click', () => {
    state.filter = button.dataset.kind;
    for (const b of document.querySelectorAll('#filters button')) b.setAttribute('aria-pressed', String(b === button));
    render();
  });
}

// ---------- ingest ----------

const jobs = [];
let draining = false;

function addJobs(entries) {
  $('#tray').hidden = false;
  for (const entry of entries) {
    const kind = entry.file ? detectKind(entry.file) ?? 'text' : 'text';
    const stateEl = h('span', { class: 'job-state' }, 'Waiting');
    const fill = h('div', { class: 'bar-fill' });
    const el = h(
      'li',
      { class: `job ${kind}`, 'data-state': 'waiting' },
      h('i', { class: 'swatch' }),
      h('span', { class: 'job-name' }, entry.file?.name ?? entry.note.title ?? 'Note'),
      stateEl,
      h('div', { class: 'bar' }, fill),
    );
    $('#tray-list').append(el);
    jobs.push({ ...entry, el, stateEl, fill });
  }
  updateTrayTitle();
  drain();
}

function updateTrayTitle() {
  const remaining = jobs.length;
  $('#tray-title').textContent = remaining ? `Adding ${remaining} ${remaining === 1 ? 'file' : 'files'}` : 'All files added';
}

async function drain() {
  if (draining) return;
  draining = true;
  try {
    await requireModel();
  } catch {
    for (const job of jobs.splice(0)) finishJob(job, 'error', 'Download the model first');
    draining = false;
    return;
  }
  while (jobs.length) {
    const job = jobs[0];
    job.el.dataset.state = 'working';
    const STAGE_PROGRESS = { Decoding: 15, Embedding: 45, 'Embedding soundtrack': 80, Saving: 95 };
    const onStage = (stage) => {
      job.stateEl.textContent = stage;
      job.fill.style.width = `${STAGE_PROGRESS[stage] ?? 50}%`;
    };
    // Multi-chunk embeds report per-batch progress from the worker.
    const onEmbed = ({ detail }) => {
      if (detail.kind === 'embed' && detail.total > 1) {
        job.fill.style.width = `${20 + (detail.done / detail.total) * 70}%`;
        job.stateEl.textContent = `Embedding ${detail.done} of ${detail.total}`;
      }
    };
    embedder.addEventListener('progress', onEmbed);
    try {
      const { item, rows } = job.file ? await ingestFile(job.file, embedder, onStage) : await ingestNote(job.note, embedder, onStage);
      state.items.set(item.id, item);
      state.index.add(rows);
      finishJob(job, 'done', `${rows.length} ${rows.length === 1 ? 'vector' : 'vectors'}`);
      render();
    } catch (error) {
      finishJob(job, 'error', error.message);
    } finally {
      embedder.removeEventListener('progress', onEmbed);
      jobs.shift();
      updateTrayTitle();
    }
  }
  draining = false;
}

function finishJob(job, result, text) {
  job.el.dataset.state = result;
  job.stateEl.textContent = text;
}

$('#tray-close').addEventListener('click', () => {
  $('#tray').hidden = true;
  if (!jobs.length) $('#tray-list').replaceChildren();
});

$('#add-files').addEventListener('change', (event) => {
  addJobs([...event.target.files].map((file) => ({ file })));
  event.target.value = '';
});

$('#load-samples').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  try {
    const files = await fetchSampleFiles((done, total) => (button.textContent = `Downloading samples ${done} of ${total}`));
    addJobs([...files.map((file) => ({ file })), ...SAMPLE_NOTES.map((note) => ({ note }))]);
  } catch (error) {
    notify(`${error.message} Check your connection and try again.`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Load sample files';
  }
});

// Drag and drop anywhere.
let dragDepth = 0;
const hasFiles = (event) => event.dataTransfer?.types.includes('Files');
window.addEventListener('dragenter', (event) => {
  if (!hasFiles(event)) return;
  dragDepth++;
  $('#drop-overlay').hidden = false;
});
window.addEventListener('dragleave', (event) => {
  if (!hasFiles(event)) return;
  if (--dragDepth <= 0) $('#drop-overlay').hidden = true;
});
window.addEventListener('dragover', (event) => hasFiles(event) && event.preventDefault());
window.addEventListener('drop', (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  $('#drop-overlay').hidden = true;
  const files = [...event.dataTransfer.files];
  if (files.length) addJobs(files.map((file) => ({ file })));
});

// Notes.
$('#add-note').addEventListener('click', () => {
  $('#note-form').reset();
  $('#note-dialog').showModal();
});
$('#note-dialog').addEventListener('close', () => {
  if ($('#note-dialog').returnValue !== 'save') return;
  const data = new FormData($('#note-form'));
  const text = String(data.get('text') ?? '').trim();
  if (text) addJobs([{ note: { title: String(data.get('title') ?? '').trim(), text } }]);
});

// ---------- viewer ----------

let viewing = null;

function openViewer(item, segment) {
  viewing = item;
  $('#viewer-title').textContent = item.title || item.name;
  const media = $('#viewer-media');
  const blobUrl = item.blob ? URL.createObjectURL(item.blob) : null;
  const start = segment?.start ?? 0;
  if (item.kind === 'image') {
    media.replaceChildren(h('img', { src: blobUrl, alt: item.name }));
  } else if (item.kind === 'video') {
    const video = h('video', { src: blobUrl, controls: true, playsInline: true, poster: previewUrl(item) });
    if (start) video.addEventListener('loadedmetadata', () => (video.currentTime = start), { once: true });
    media.replaceChildren(video);
  } else if (item.kind === 'audio') {
    const audio = h('audio', { src: blobUrl, controls: true });
    if (start) audio.addEventListener('loadedmetadata', () => (audio.currentTime = start), { once: true });
    media.replaceChildren(h('img', { class: 'waveform', src: previewUrl(item), alt: '' }), audio);
  } else {
    media.replaceChildren(h('div', { class: 'note-text' }, item.text));
  }
  const vectors = state.index.rowsFor(item.id).length;
  $('#viewer-meta').textContent = [
    KIND_LABEL[item.kind],
    item.width ? `${item.width}×${item.height}` : null,
    item.duration ? formatTime(item.duration) : null,
    item.kind === 'text' ? `${item.text.length.toLocaleString()} characters` : formatBytes(item.size),
    `${vectors} ${vectors === 1 ? 'vector' : 'vectors'}`,
    `added ${new Date(item.addedAt).toLocaleString()}`,
  ]
    .filter(Boolean)
    .join(', ');
  $('#viewer').showModal();
  $('#viewer').addEventListener('close', () => blobUrl && URL.revokeObjectURL(blobUrl), { once: true });
}

$('#viewer-close').addEventListener('click', () => $('#viewer').close());

$('#viewer-similar').addEventListener('click', () => {
  const item = viewing;
  $('#viewer').close();
  const [row] = state.index.rowsFor(item.id).filter((r) => r.modality === item.kind);
  if (!row) return;
  const image = item.kind === 'image' || item.kind === 'video' ? URL.createObjectURL(item.preview) : undefined;
  runQuery({ label: `“${item.title || item.name}”`, chip: item.title || item.name, image, exclude: item.id, embed: async () => row.vector });
});

$('#viewer-delete').addEventListener('click', async () => {
  const item = viewing;
  $('#viewer').close();
  await store.deleteItem(item.id);
  state.items.delete(item.id);
  state.index.removeItem(item.id);
  if (previewUrls.has(item.id)) URL.revokeObjectURL(previewUrls.get(item.id));
  previewUrls.delete(item.id);
  render();
});

// ---------- settings ----------

function radioGroup(container, name, options, current, onChange) {
  container.replaceChildren(
    ...options.map(({ value, label, disabled }) =>
      h(
        'label',
        {},
        h('input', {
          type: 'radio',
          name,
          value,
          checked: String(value) === String(current),
          disabled,
          class: 'sr-only',
          onchange: () => onChange(value),
        }),
        label,
      ),
    ),
  );
}

function renderSettings() {
  radioGroup(
    $('#dim-options'),
    'dim',
    DIMENSIONS.map((d) => ({ value: d, label: String(d) })),
    settings.dim,
    (dim) => {
      settings.dim = dim;
      saveSettings(settings);
      // Stored vectors stay at 768; both sides are truncated and re-normalised on the fly.
      state.index.setDimension(dim);
      render();
    },
  );

  const pending = { vision: settings.vision, audio: settings.audio, device: settings.device };
  const showApply = () => {
    const info = embedder.info;
    const changed =
      !info ||
      info.vision !== pending.vision ||
      info.audio !== pending.audio ||
      (pending.device !== 'auto' && pending.device !== info.device);
    $('#apply-row').hidden = !changed || !state.model;
    $('#apply-text').textContent = `The model reloads with your changes. Download if not cached: up to ${downloadMB(pending)} MB.`;
  };
  encoderRows($('#settings-encoders'), pending, showApply);

  const webgpu = 'gpu' in navigator;
  radioGroup(
    $('#device-options'),
    'device',
    [
      { value: 'auto', label: 'Automatic' },
      { value: 'webgpu', label: 'WebGPU', disabled: !webgpu },
      { value: 'wasm', label: 'WASM' },
    ],
    settings.device,
    (device) => {
      pending.device = device;
      showApply();
    },
  );
  $('#device-hint').textContent = !webgpu
    ? 'This browser has no WebGPU, so the model runs on WASM (CPU).'
    : embedder.info
      ? `Running on ${embedder.info.device === 'webgpu' ? 'WebGPU' : 'WASM'}. WebGPU is much faster; WASM works everywhere.`
      : 'WebGPU is much faster; WASM works everywhere.';

  $('#apply-model').onclick = () => {
    Object.assign(settings, pending);
    saveSettings(settings);
    $('#settings').close();
    loadModel();
  };
  showApply();

  const vectors = state.index.size;
  $('#library-stats').textContent = `${state.items.size} items and ${vectors} vectors, stored in this browser only.`;
}

$('#open-settings').addEventListener('click', () => {
  renderSettings();
  $('#settings').showModal();
});

$('#clear-library').addEventListener('click', async () => {
  if (!confirm('Delete every file and note from this library? This cannot be undone.')) return;
  await store.clearLibrary();
  state.items.clear();
  state.index.clear();
  for (const url of previewUrls.values()) URL.revokeObjectURL(url);
  previewUrls.clear();
  clearQuery();
  renderSettings();
});

// Light-dismiss for dialogs: a click on the backdrop closes them.
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.addEventListener('click', (event) => event.target === dialog && dialog.close());
}

// ---------- boot ----------

async function boot() {
  setStatus('Opening library');
  const [items, vectors] = await Promise.all([store.allItems(), store.allVectors()]);
  for (const item of items) state.items.set(item.id, item);
  state.index = new VectorIndex(vectors, settings.dim);
  render();

  if (await isModelCached(settings)) loadModel();
  else showSetup();

  if (import.meta.env.PROD && 'serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(() => {
      // Offline reloads won't work, but the app itself is unaffected.
    });
  }
}

boot().catch((error) => {
  setStatus('Could not start', 'error');
  notify(`The library could not open: ${error.message}`, 'error');
});
