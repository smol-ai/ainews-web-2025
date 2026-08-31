import assert from 'node:assert/strict';
import { test, type TestContext } from 'node:test';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { scrapeTarget } from './scrapeNitterList.js';
import { readJson } from '../shared/state.js';

const target = { pageType: 'list', identifier: '123' };
const window = { start: '2026-08-30T00:00:00Z', end: '2026-08-31T00:00:00Z' };
const tweet = (id: string) => `<div class="timeline-item"><a class="tweet-link" href="/alice/status/${id}"></a><div class="tweet-body"><div class="tweet-header"><a class="username">@alice</a></div><span class="tweet-date"><a title="Aug 30, 2026 12:00 PM UTC"></a></span><div class="tweet-content media-body">tweet ${id}</div></div></div>`;
const page = (ids: string[], cursor?: string) => `<div class="timeline">${ids.map(tweet).join('')}${cursor ? `<div class="show-more"><a href="?cursor=${cursor}">more</a></div>` : ''}</div>`;

function fixture(t: TestContext) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ainews-twitter-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const options = { ...window, mockHtmlDir: dir, checkpointPath: path.join(dir, 'checkpoint.json') };
  return { dir, options, write: (n: number, html: string) => fs.writeFileSync(path.join(dir, `page${n}.html`), html) };
}

test('failed page resumes from committed cursor, retains tweets and deduplicates overlap', async (t) => {
  const f = fixture(t);
  f.write(1, page(['1'], 'next'));
  const partial = await scrapeTarget(target, f.options);
  assert.equal(partial.diagnostics.wasPartial, true);
  assert.equal(partial.diagnostics.pagesFetched, 1);
  assert.deepEqual(partial.tweets.map((t) => t.id), ['1']);
  assert.equal(readJson<any>(f.options.checkpointPath).cursor, 'next');
  // Removing page 1 proves recovery fetches only the uncommitted second page.
  fs.unlinkSync(path.join(f.dir, 'page1.html'));
  f.write(2, page(['1', '2']));
  const resumed = await scrapeTarget(target, f.options);
  assert.equal(resumed.diagnostics.wasPartial, false);
  assert.equal(resumed.diagnostics.pagesFetched, 2);
  assert.deepEqual(resumed.tweets.map((t) => t.id), ['1', '2']);
  fs.unlinkSync(path.join(f.dir, 'page2.html'));
  const complete = await scrapeTarget(target, f.options);
  assert.equal(complete.diagnostics.wasPartial, false);
  assert.deepEqual(complete.tweets, JSON.parse(JSON.stringify(resumed.tweets)));
});

test('page and character limits remain partial and can be raised on resume', async (t) => {
  for (const limit of [{ maxPages: 1 }, { maxChars: 1 }]) {
    const f = fixture(t);
    f.write(1, page(['1'], 'next'));
    f.write(2, page(['2']));
    const partial = await scrapeTarget(target, { ...f.options, ...limit });
    assert.equal(partial.diagnostics.wasPartial, true);
    assert.equal(partial.diagnostics.pagesFetched, 1);
    const resumed = await scrapeTarget(target, { ...f.options, maxPages: 2, maxChars: 10000 });
    assert.equal(resumed.diagnostics.wasPartial, false);
    assert.equal(resumed.tweets.length, 2);
  }
});

test('checkpoint freezes target and window; omitted window resumes original dates', async (t) => {
  const f = fixture(t);
  f.write(1, page(['1']));
  await scrapeTarget(target, f.options);
  await assert.rejects(scrapeTarget({ ...target, identifier: '456' }, f.options), /does not match/);
  await assert.rejects(scrapeTarget(target, { ...f.options, start: '2026-08-29T00:00:00Z' }), /does not match/);
  await assert.rejects(scrapeTarget(target, { ...f.options, baseUrl: 'https://other.example' }), /does not match/);
  const { start, end, ...withoutWindow } = f.options;
  const result = await scrapeTarget(target, withoutWindow);
  assert.equal(result.diagnostics.startParam, new Date(start).toISOString());
  assert.equal(result.diagnostics.endParam, new Date(end).toISOString());
});

test('source credentials and query secrets are rejected before checkpoint creation', async (t) => {
  const f = fixture(t);
  for (const baseUrl of ['https://user:secret@example.com', 'https://example.com?token=secret', 'https://example.com#secret', 'file:///etc/passwd']) {
    await assert.rejects(scrapeTarget(target, { ...f.options, baseUrl }), /baseUrl must be/);
    assert.equal(fs.existsSync(f.options.checkpointPath), false);
  }
});

test('challenge, invalid timeline and malformed tweets do not advance the checkpoint', async (t) => {
  for (const html of ['<html>Verifying your browser</html>', '<div class="error-panel">unavailable</div>', '<div class="timeline"><div class="timeline-item"><div class="tweet-body">unparseable</div></div></div>']) {
    const f = fixture(t);
    f.write(1, html);
    const partial = await scrapeTarget(target, f.options);
    assert.equal(partial.diagnostics.wasPartial, true);
    assert.equal(partial.diagnostics.pagesFetched, 0);
    assert.equal(readJson<any>(f.options.checkpointPath).pageIndex, 1);
    f.write(1, page(['1']));
    const result = await scrapeTarget(target, f.options);
    assert.equal(result.diagnostics.wasPartial, false);
  }
});
