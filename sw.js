// Jen kvůli instalaci jako aplikace. Nic necachuje: dřívější verze (cache-first) držela v mobilu starou stránku.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(
  caches.keys().then((keys) => Promise.all(keys.map((k) => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener("fetch", () => {});
