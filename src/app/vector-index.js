// Brute-force search over every stored vector, at a chosen Matryoshka dimension.
import { DIM } from '../embed/config.js';

/** Keeps the leading `dim` values and re-normalises to unit length (skipping this silently degrades ranking). */
export function truncate(vector, dim) {
  const out = vector.slice(0, dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) out[i] /= norm;
  return out;
}

export class VectorIndex {
  /** @type {import('./store.js').VectorRow[]} */
  #rows = [];
  #matrix = new Float32Array(0);
  dim = DIM;

  /** @param {import('./store.js').VectorRow[]} rows */
  constructor(rows = [], dim = DIM) {
    this.#rows = rows;
    this.setDimension(dim);
  }

  get size() {
    return this.#rows.length;
  }

  /** Rebuilds the packed, truncated matrix. Stored vectors always stay at 768. */
  setDimension(dim) {
    this.dim = dim;
    this.#matrix = new Float32Array(this.#rows.length * dim);
    this.#rows.forEach((row, i) => this.#matrix.set(truncate(row.vector, dim), i * dim));
  }

  add(rows) {
    this.#rows = [...this.#rows, ...rows];
    this.setDimension(this.dim);
  }

  /** The stored (full 768-d) rows of one item. */
  rowsFor(itemId) {
    return this.#rows.filter((row) => row.itemId === itemId);
  }

  removeItem(itemId) {
    this.#rows = this.#rows.filter((row) => row.itemId !== itemId);
    this.setDimension(this.dim);
  }

  clear() {
    this.#rows = [];
    this.setDimension(this.dim);
  }

  /**
   * Scores every segment, keeps each item's best segment, returns the top `k` items.
   * @param {Float32Array} query a full 768-d query vector
   * @param {{ k?: number, kinds?: Set<string>, kindOf?: (itemId: string) => string | undefined, exclude?: string }} [options]
   */
  search(query, { k = 10, kinds, kindOf, exclude } = {}) {
    const { dim } = this;
    const q = truncate(query, dim);
    const best = new Map();
    for (let r = 0; r < this.#rows.length; r++) {
      const row = this.#rows[r];
      if (row.itemId === exclude) continue;
      if (kinds && kindOf && !kinds.has(kindOf(row.itemId))) continue;
      let score = 0;
      const offset = r * dim;
      for (let i = 0; i < dim; i++) score += q[i] * this.#matrix[offset + i];
      const current = best.get(row.itemId);
      if (!current || score > current.score) best.set(row.itemId, { itemId: row.itemId, score, segment: row });
    }
    return [...best.values()].sort((a, b) => b.score - a.score).slice(0, k);
  }
}
