import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const home = fileURLToPath(new URL('..', import.meta.url));
const runs = path.join(home, 'runs');
function cli(...args: string[]) {
  return spawnSync(process.execPath, ['--import', import.meta.resolve('tsx'), path.join(home, 'cli.ts'), ...args], {
    cwd: home, encoding: 'utf8', timeout: 15000,
    env: { ...process.env, OPENAI_API_KEY: '', ANTHROPIC_API_KEY: '' },
  });
}
const page = (id: string, cursor?: string) => `<div class="timeline"><div class="timeline-item"><a class="tweet-link" href="/alice/status/${id}"></a><div class="tweet-body"><span class="tweet-date"><a title="Aug 30, 2026 12:00 PM UTC"></a></span><div class="tweet-content media-body">tweet ${id}</div></div></div>${cursor ? `<div class="show-more"><a href="?cursor=${cursor}">more</a></div>` : ''}</div>`;

test('real CLI checkpoints incomplete scrape, resumes remaining page, and reuses completed run', (t) => {
  const fixtures = fs.mkdtempSync(path.join(os.tmpdir(), 'ainews-cli-'));
  t.after(() => fs.rmSync(fixtures, { recursive: true, force: true }));
  fs.writeFileSync(path.join(fixtures, 'page1.html'), page('1', 'second'));
  fs.writeFileSync(path.join(fixtures, 'page2.html'), page('2'));
  const first = cli('run', 'twitter', '--target', '123', '--start', '2026-08-30T00:00:00Z', '--end', '2026-08-31T00:00:00Z', '--scrape-only', '--mockHtmlDir', fixtures, '--maxPages', '1');
  const id = first.stdout.match(/^Run: ([a-zA-Z0-9_-]+)$/m)?.[1];
  assert.ok(id, `${first.stdout}\n${first.stderr}`);
  const directory = path.join(runs, id);
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const state = () => JSON.parse(fs.readFileSync(path.join(directory, 'run.json'), 'utf8'));
  assert.equal(first.status, 1, first.stderr);
  assert.equal(state().status, 'incomplete');
  assert.equal(state().jobs[0].scraped, undefined);
  assert.equal(fs.existsSync(path.join(directory, 'run.lock')), false);
  fs.unlinkSync(path.join(fixtures, 'page1.html'));
  const resume = cli('resume', id, '--maxPages', '2');
  assert.equal(resume.status, 0, `${resume.stdout}\n${resume.stderr}`);
  assert.equal(state().status, 'complete');
  assert.equal(state().jobs[0].scraped, true);
  const rawPath = path.join(directory, 'twitter', 'raw.json');
  const raw = fs.readFileSync(rawPath, 'utf8');
  assert.deepEqual(JSON.parse(raw).tweets.map((tweet: { id: string }) => tweet.id), ['1', '2']);
  fs.unlinkSync(path.join(fixtures, 'page2.html'));
  const complete = cli('resume', id);
  assert.equal(complete.status, 0, `${complete.stdout}\n${complete.stderr}`);
  assert.equal(fs.readFileSync(rawPath, 'utf8'), raw);
  assert.equal(fs.existsSync(path.join(directory, 'run.lock')), false);
});

test('malformed CLI options fail before creating runs', () => {
  const ids = () => fs.existsSync(runs) ? fs.readdirSync(runs).sort() : [];
  const before = ids();
  for (const args of [
    ['run', 'twitter', '--scrape-only', '--maxPages', '0'],
    ['run', 'twitter', '--scrape-only', '--maxPages', '1.5'],
    ['run', 'twitter', '--scrape-only', '--target', '../../oops'],
    ['run', 'twitter', '--scrape-only', '--baseUrl', 'https://user:password@example.com'],
    ['run', 'twitter', '--scrape-only', '--start', '2026-08-31', '--end', '2026-08-30'],
    ['run', 'twitter', '--scrape-only', '--start', 'nonsense'],
    ['run', 'twitter', '--scrape-only', '--end', 'nonsense'],
    ['resume', '../outside'],
    ['run', 'twitter', '--unknown'],
  ]) {
    const result = cli(...args);
    assert.equal(result.status, 1, `${args.join(' ')}: ${result.stdout}\n${result.stderr}`);
    assert.doesNotMatch(result.stdout, /^Run: /m);
    assert.deepEqual(ids(), before);
  }
});
