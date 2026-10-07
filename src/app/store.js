// IndexedDB library: items (metadata, original file, preview) and vectors (one row per embedded segment).
import { openDB } from 'idb';

/**
 * @typedef {'text' | 'image' | 'audio' | 'video'} Kind
 * @typedef {{
 *   id: string, kind: Kind, name: string, mime: string, size: number, addedAt: number,
 *   blob?: Blob, preview?: Blob, title?: string, text?: string,
 *   duration?: number, width?: number, height?: number,
 * }} Item
 * @typedef {{ id?: number, itemId: string, modality: Kind, start?: number, end?: number, vector: Float32Array }} VectorRow
 */

const dbPromise = openDB('gemma-library', 1, {
  upgrade(db) {
    db.createObjectStore('items', { keyPath: 'id' });
    const vectors = db.createObjectStore('vectors', { keyPath: 'id', autoIncrement: true });
    vectors.createIndex('itemId', 'itemId');
  },
});

/** @param {Item} item @param {Omit<VectorRow, 'itemId'>[]} vectors */
export async function addItem(item, vectors) {
  const db = await dbPromise;
  const tx = db.transaction(['items', 'vectors'], 'readwrite');
  const rows = vectors.map((v) => ({ ...v, itemId: item.id }));
  await Promise.all([
    tx.objectStore('items').put(item),
    ...rows.map((row) => tx.objectStore('vectors').add(row)),
    tx.done,
  ]);
  return rows;
}

/** @returns {Promise<Item[]>} newest first */
export async function allItems() {
  const items = await (await dbPromise).getAll('items');
  return items.sort((a, b) => b.addedAt - a.addedAt);
}

/** @returns {Promise<VectorRow[]>} */
export async function allVectors() {
  return (await dbPromise).getAll('vectors');
}

export async function getItem(id) {
  return (await dbPromise).get('items', id);
}

export async function deleteItem(id) {
  const db = await dbPromise;
  const tx = db.transaction(['items', 'vectors'], 'readwrite');
  const keys = await tx.objectStore('vectors').index('itemId').getAllKeys(id);
  await Promise.all([tx.objectStore('items').delete(id), ...keys.map((k) => tx.objectStore('vectors').delete(k)), tx.done]);
}

export async function clearLibrary() {
  const db = await dbPromise;
  const tx = db.transaction(['items', 'vectors'], 'readwrite');
  await Promise.all([tx.objectStore('items').clear(), tx.objectStore('vectors').clear(), tx.done]);
}
