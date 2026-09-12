import assert from 'node:assert/strict';

const base = process.argv[2] || 'http://127.0.0.1:8792';
for (const path of ['/', '/issues/', '/issues/26-09-10-not-much/', '/tags/', '/projects/', '/issues/25-09-08-cog-smol/', '/frozen-issues/25-09-08-cog-smol.html', '/rss.xml', '/sitemap-index.xml', '/pagefind/pagefind.js']) {
  const response = await fetch(new URL(path, base));
  assert.equal(response.status, 200, `${path}: HTTP ${response.status}`);
  assert.ok((await response.text()).length > 100, `${path}: empty content`);
  console.log(`PASS ${path}`);
}
for (const path of ['/api/og?title=AI+News', '/api/og?type=issue&title=September+10+AI+News&description=The+**latest**+AI+news&companies=OpenAI&models=GPT-6&date=2026-09-10&secret=nocache', '/api/og-alt?title=AI+News']) {
  const response = await fetch(new URL(path, base));
  assert.equal(response.status, 200, path);
  assert.equal(response.headers.get('content-type'), 'image/png', path);
  assert.equal(response.headers.get('cache-control'), path.includes('nocache') ? 'no-cache, no-store' : 'public, max-age=86400', path);
  const png = Buffer.from(await response.arrayBuffer());
  assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', path);
  assert.equal(png.readUInt32BE(16), 1200, path);
  assert.equal(png.readUInt32BE(20), 630, path);
  console.log(`PASS ${path} (${png.length} bytes)`);
}
const issues = await (await fetch(new URL('/api/issues?page=2', base))).json();
assert.equal(issues.currentPage, 1, 'Preserve the statically generated issues API');
assert.equal(issues.issues.length, 30);
console.log('PASS /api/issues?page=2: static snapshot');
for (const path of ['/missing-cloudflare-check', '/_worker.js/index.js']) {
  const response = await fetch(new URL(path, base));
  assert.equal(response.status, 404, path);
  console.log(`PASS ${path}: 404`);
}
