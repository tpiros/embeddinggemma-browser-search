// App-shell service worker. The build replaces the two placeholders below with the hashed file list.
// Model weights are not handled here: Transformers.js keeps them in its own Cache Storage entry.

const VERSION = '__VERSION__';
const PRECACHE = __PRECACHE__;
const CACHE = `gemma-search-shell-${VERSION}`;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(PRECACHE.map((path) => new URL(path, self.registration.scope))))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('gemma-search-shell-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;

  if (request.mode === 'navigate') {
    // Fresh HTML when online, the cached shell when not.
    event.respondWith(
      fetch(request).catch(async () => (await caches.match(request)) ?? caches.match(new URL('./', self.registration.scope))),
    );
    return;
  }
  // Hashed assets never change, so the cache wins.
  event.respondWith(caches.match(request).then((hit) => hit ?? fetch(request)));
});
