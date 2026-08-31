import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import axios, { AxiosError } from 'axios';
import { scrapeReddit } from '../reddit/scrape';
import { getRateLimiter } from '../reddit/api';

test('Reddit resumes after successful empty subreddit and retries blocked subreddit', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ainews-reddit-'));
  const oldSession = process.env.REDDIT_SESSION;
  process.env.REDDIT_SESSION = 'test-not-a-real-session';
  t.after(() => { fs.rmSync(dir, { recursive: true }); if (oldSession === undefined) delete process.env.REDDIT_SESSION; else process.env.REDDIT_SESSION = oldSession; });
  t.mock.method(getRateLimiter(), 'throttle', async () => {});
  const calls: string[] = [];
  let blocked = true;
  t.mock.method(axios, 'get', async (url: string) => {
    calls.push(url);
    if (url.includes('/second/') && blocked) throw new AxiosError('blocked', 'ERR_BAD_REQUEST', undefined, undefined, { status: 403 } as never);
    return { status: 200, data: { data: { children: [] } } };
  });
  await assert.rejects(scrapeReddit(dir, ['first', 'second'], 200), /blocked/);
  assert.ok(fs.existsSync(path.join(dir, 'checkpoints', 'first.json')));
  assert.equal(fs.existsSync(path.join(dir, 'checkpoints', 'second.json')), false);
  blocked = false;
  const raw = await scrapeReddit(dir, ['first', 'second'], 200);
  assert.equal(calls.filter(url => url.includes('/first/')).length, 1);
  assert.equal(calls.filter(url => url.includes('/second/')).length, 2);
  assert.equal(JSON.parse(fs.readFileSync(raw, 'utf8')).length, 2);
});
