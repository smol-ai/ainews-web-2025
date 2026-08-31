import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { acquireLock, checkpoint, readJson, writeJson } from './state';

test('resume preserves successful work, retries failures and keeps empty results', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'scrapers-state-'));
  try {
    let calls = 0;
    const file = path.join(dir, 'step.json');
    await assert.rejects(checkpoint(file, async () => { calls++; throw new Error('provider failed'); }));
    assert.equal(readJson(file), undefined);
    assert.deepEqual(await checkpoint(file, async () => { calls++; return []; }), []);
    assert.deepEqual(await checkpoint(file, async () => { calls++; return ['unexpected']; }), []);
    assert.equal(calls, 2);
    assert.deepEqual(fs.readdirSync(dir), ['step.json']);
  } finally { fs.rmSync(dir, { recursive: true }); }
});

test('corrupt checkpoints fail closed; live runs cannot be resumed twice', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'scrapers-state-'));
  try {
    fs.writeFileSync(path.join(dir, 'bad.json'), '{');
    assert.throws(() => readJson(path.join(dir, 'bad.json')));
    const release = acquireLock(dir);
    assert.throws(() => acquireLock(dir), /active/);
    release();
    acquireLock(dir)();
    writeJson(path.join(dir, 'valid.json'), { complete: true });
    assert.deepEqual(readJson(path.join(dir, 'valid.json')), { complete: true });
  } finally { fs.rmSync(dir, { recursive: true }); }
});
