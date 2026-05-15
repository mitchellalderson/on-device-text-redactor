const CACHE_VERSION = "v2";

const CACHE_NAME = `phi-firewall-models-${CACHE_VERSION}`;

export async function fetchWithCache(
  url: string,
  onStatus?: (msg: string) => void,
): Promise<ArrayBuffer> {
  await pruneOldCaches();

  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    onStatus?.("Loading model from cache...");
    return cached.arrayBuffer();
  }

  onStatus?.("Downloading model...");
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

  await cache.put(url, response.clone());
  return response.arrayBuffer();
}

async function pruneOldCaches(): Promise<void> {
  const names = await caches.keys();
  for (const name of names) {
    if (name.startsWith("phi-firewall-models-") && name !== CACHE_NAME) {
      await caches.delete(name);
    }
  }
}
