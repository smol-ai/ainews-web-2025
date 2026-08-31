import { test } from 'node:test';
import assert from 'node:assert/strict';
import axios from 'axios';
import sharp from 'sharp';
import { prepareImage } from '../reddit/image-input';

test('large images become bounded JPEG inputs; deleted images are explicitly unavailable', async t => {
  const data = await sharp({ create: { width: 2000, height: 1000, channels: 3, background: 'white' } }).png().toBuffer();
  const get = t.mock.method(axios, 'get', async () => ({ status: 200, data }));
  const prepared = await prepareImage('https://example.com/image.gif');
  assert.ok('image' in prepared);
  const metadata = await sharp(prepared.image).metadata();
  assert.equal(metadata.format, 'jpeg');
  assert.equal(metadata.width, 1536);
  get.mock.mockImplementation(async () => ({ status: 404, data: Buffer.alloc(0) }));
  const missing = await prepareImage('https://example.com/deleted.png');
  assert.ok('unavailable' in missing);
  assert.match(missing.unavailable, /404/);
});
