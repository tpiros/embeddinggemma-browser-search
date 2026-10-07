// The sample set: media from the Transformers.js docs dataset, plus a few short notes.

const BASE = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main';

const FILES = [
  ['cats.jpg', 'image/jpeg'],
  ['corgi.jpg', 'image/jpeg'],
  ['tiger.jpg', 'image/jpeg'],
  ['football-match.jpg', 'image/jpeg'],
  ['airport.jpg', 'image/jpeg'],
  ['moraine-lake.png', 'image/png'],
  ['butterfly.jpg', 'image/jpeg'],
  ['savanna.jpg', 'image/jpeg'],
  ['jfk.wav', 'audio/wav'],
  ['dog_barking.wav', 'audio/wav'],
  ['cat_meow.wav', 'audio/wav'],
  ['piano.wav', 'audio/wav'],
  ['sea-turtle.mp4', 'video/mp4'],
  ['courtroom.mp4', 'video/mp4'],
];

export const SAMPLE_NOTES = [
  {
    title: 'Packing list',
    text: 'Snorkel, mask, reef-safe sunscreen, underwater camera housing and a dry bag for the boat trip to the turtle bay.',
  },
  {
    title: 'Vet appointment',
    text: 'Booked the corgi in for her yearly check-up and vaccinations on Thursday at 10am. Bring the vaccination card.',
  },
  {
    title: 'Hiking idea',
    text: 'Moraine Lake in Banff: go at sunrise to beat the crowds, then walk the Consolation Lakes trail.',
  },
];

/** @param {(done: number, total: number) => void} onProgress */
export async function fetchSampleFiles(onProgress) {
  let done = 0;
  return Promise.all(
    FILES.map(async ([name, type]) => {
      const response = await fetch(`${BASE}/${name}`);
      if (!response.ok) throw new Error(`Could not download ${name} (HTTP ${response.status}).`);
      const file = new File([await response.blob()], name, { type });
      onProgress(++done, FILES.length);
      return file;
    }),
  );
}
