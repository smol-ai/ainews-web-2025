import { unstable_startWorker } from 'wrangler';

// Preview a completed build without watching the archive's thousands of files.
const portIndex = process.argv.indexOf('--port');
const port = portIndex === -1 ? 8792 : Number(process.argv[portIndex + 1]);
const worker = await unstable_startWorker({
  config: 'wrangler.jsonc',
  dev: { watch: false, remote: false, server: { port } },
});
console.log(`News preview ready at ${await worker.url}`);
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.once(signal, async () => { await worker.dispose(); process.exit(0); });
}
await worker.ready;
