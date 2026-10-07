// Records a voice clip with MediaRecorder. The clip is decoded and resampled to 16 kHz mono later,
// by the same path as dropped audio files.

export class VoiceRecorder {
  /** @type {MediaRecorder | null} */
  #recorder = null;
  #chunks = [];
  #stopped = null;

  get recording() {
    return this.#recorder?.state === 'recording';
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.#chunks = [];
    this.#recorder = new MediaRecorder(stream);
    this.#recorder.addEventListener('dataavailable', ({ data }) => data.size && this.#chunks.push(data));
    this.#stopped = new Promise((resolve) =>
      this.#recorder.addEventListener('stop', () => {
        stream.getTracks().forEach((track) => track.stop());
        resolve(new Blob(this.#chunks, { type: this.#recorder.mimeType }));
      }),
    );
    this.#recorder.start();
  }

  /** @returns {Promise<Blob>} */
  async stop() {
    if (!this.#recorder) throw new Error('Not recording');
    if (this.#recorder.state !== 'inactive') this.#recorder.stop();
    const blob = await this.#stopped;
    this.#recorder = null;
    return blob;
  }
}
