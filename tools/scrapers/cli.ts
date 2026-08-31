import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { parseArgs } from 'node:util';
import { spawn } from 'node:child_process';
import dotenv from 'dotenv';
import { acquireLock, readJson, writeJson } from './shared/state';
import { redditPresets, scrapeReddit } from './reddit/scrape';
import { resolveWindow } from './twitter/time';

const home = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(home, '.env') });
type Job = { name: string; source: 'reddit' | 'twitter'; subreddits?: string[]; minimum?: number; raw?: string; scraped?: boolean; summarized?: boolean };
type Run = { version: 1; id: string; createdAt: string; status: string; summarize: boolean; options: Record<string, string>; jobs: Job[] };
const help = `AI News local scrapers

pnpm scrape run daily                  Reddit local + all, then Twitter + summaries
pnpm scrape run reddit-local           LocalLlama / localLLM + clustering
pnpm scrape run reddit-all             General AI subreddits + clustering
pnpm scrape run twitter --target @name List ID or profile (default: 1585430245762441216)
pnpm scrape run twitter --scrape-only  Save raw data without model calls
pnpm scrape resume RUN_ID              Retry unfinished work from checkpoints
pnpm scrape resume RUN_ID --summarize  Add summaries to a scrape-only run
pnpm scrape status [RUN_ID]            Inspect saved progress offline

Options for new runs:
  --start ISO|3d --end ISO             Twitter window (frozen when run starts)
  --baseUrl URL --maxPages N           Nitter instance and page limit
  --mockHtmlDir DIR                    Offline Twitter HTML fixtures
  --subreddits NAME,NAME --min-score N Override Reddit preset
  --json FILE                         Process existing raw JSON (single job only)
  --model MODEL --focusTopic TEXT --system FILE --maxChars N  Twitter summary options
  --clusters N --top-k N --max-per-cluster N --min-per-cluster N  Reddit options
  --scrape-only                       Skip AI processing

On resume, --maxPages N and --maxChars N may increase Twitter limits. Inputs otherwise stay frozen.
Runs and credentials are local and gitignored. Exit 1 means incomplete; resume the printed ID.
`;

function child(script: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, ['--import', import.meta.resolve('tsx'), path.join(home, script), ...args], { cwd, env: process.env, stdio: 'inherit' });
    const forward = (signal: NodeJS.Signals) => proc.kill(signal);
    const interrupt = () => forward('SIGINT');
    const terminate = () => forward('SIGTERM');
    process.on('SIGINT', interrupt); process.on('SIGTERM', terminate);
    proc.on('error', reject);
    proc.on('exit', (code, signal) => {
      process.off('SIGINT', interrupt); process.off('SIGTERM', terminate);
      code === 0 ? resolve() : reject(new Error(`Processing interrupted (${signal ?? code}); saved successful steps are reusable.`));
    });
  });
}

const stringOptions = ['target', 'start', 'end', 'baseUrl', 'maxPages', 'maxChars', 'mockHtmlDir', 'subreddits', 'min-score', 'json', 'model', 'focusTopic', 'system', 'clusters', 'top-k', 'max-per-cluster', 'min-per-cluster'];
function runPath(id: string): string {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) throw new Error('Invalid run ID; use an ID from pnpm scrape status.');
  return path.join(home, 'runs', id);
}

async function main() {
  const parsed = parseArgs({ allowPositionals: true, options: {
    ...Object.fromEntries(stringOptions.map(key => [key, { type: 'string' as const }])),
    'scrape-only': { type: 'boolean' }, summarize: { type: 'boolean' }, help: { type: 'boolean' },
  } });
  const values = parsed.values as Record<string, string | boolean | undefined>;
  const positionals = parsed.positionals;
  const [command, selected] = positionals;
  if (values.help || !command) { console.log(help); return; }
  if (positionals.length > 2) throw new Error('Too many arguments. Use --help.');
  if (command === 'status') {
    const directory = path.join(home, 'runs');
    if (selected) console.log(JSON.stringify(readJson(path.join(runPath(selected), 'run.json')) ?? { error: 'Run not found' }, null, 2));
    else for (const id of fs.existsSync(directory) ? fs.readdirSync(directory).sort() : []) {
      const run = readJson<Run>(path.join(runPath(id), 'run.json'));
      if (run) console.log(`${run.id} ${run.status} ${run.jobs.map(j => `${j.name}:${j.summarized ? 'done' : j.scraped ? 'scraped' : 'pending'}`).join(' ')}`);
    }
    return;
  }
  if (command !== 'run' && command !== 'resume') throw new Error('Expected run, resume, or status. Use --help.');
  for (const name of ['maxPages', 'maxChars', 'clusters', 'top-k', 'max-per-cluster', 'min-per-cluster', 'min-score']) {
    if (values[name] !== undefined && (!/^\d+$/.test(String(values[name])) || Number(values[name]) < (name === 'min-score' ? 0 : 1))) throw new Error(`--${name} requires a positive integer.`);
  }
  let run: Run;
  if (command === 'resume') {
    if (!selected) throw new Error('Provide a run ID from pnpm scrape status.');
    for (const key of Object.keys(values)) if (!['maxPages', 'maxChars', 'summarize'].includes(key)) throw new Error(`Cannot change --${key} on resume; start a new run.`);
    const saved = readJson<Run>(path.join(runPath(selected), 'run.json'));
    if (!saved || saved.version !== 1) throw new Error('Run missing or unsupported checkpoint version.');
    run = saved;
    if (values.summarize) run.summarize = true;
    if (values.maxPages) {
      if (Number(values.maxPages) < Number(run.options.maxPages)) throw new Error('Resume may only increase --maxPages.');
      run.options.maxPages = String(values.maxPages);
    }
    if (values.maxChars) {
      if (Number(values.maxChars) < Number(run.options.maxChars ?? '200000')) throw new Error('Resume may only increase --maxChars.');
      run.options.maxChars = String(values.maxChars);
    }
  } else {
    if (!selected || !['daily', 'reddit-local', 'reddit-all', 'twitter'].includes(selected)) throw new Error('Choose daily, reddit-local, reddit-all, or twitter.');
    if (values.summarize) throw new Error('New runs summarize by default; --summarize is for resume.');
    const options = Object.fromEntries(stringOptions.filter(key => typeof values[key] === 'string').map(key => [key, String(values[key])]));
    for (const key of ['json', 'system', 'mockHtmlDir']) if (options[key]) {
      options[key] = path.resolve(options[key]);
      if (!fs.existsSync(options[key])) throw new Error(`--${key} path does not exist.`);
    }
    const target = options.target ?? '1585430245762441216';
    if (!/^@?[a-zA-Z0-9_]+$/.test(target)) throw new Error('Target must be a list ID or Twitter handle.');
    if (options.baseUrl) {
      const url = new URL(options.baseUrl);
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) throw new Error('baseUrl must be HTTP(S) without credentials, query, or fragment.');
    }
    const { start, end } = resolveWindow(options.start, options.end);
    if (start >= end) throw new Error('start must be before end.');
    Object.assign(options, { target, start: start.toISOString(), end: end.toISOString(), maxPages: options.maxPages ?? '100' });
    const jobs: Job[] = [];
    for (const preset of ['local', 'all'] as const) if (selected === 'daily' || selected === `reddit-${preset}`) {
      const subreddits = options.subreddits ? [...new Set(options.subreddits.split(',').map(s => s.replace(/^\/?r\//, '').toLowerCase()))] : redditPresets[preset];
      if (subreddits.some(s => !/^[a-zA-Z0-9_]+$/.test(s))) throw new Error('Subreddits must be comma-separated names.');
      jobs.push({ name: `reddit-${preset}`, source: 'reddit', subreddits, minimum: Number(options['min-score'] ?? (preset === 'local' ? 200 : 350)) });
    }
    if (selected === 'daily' || selected === 'twitter') jobs.push({ name: 'twitter', source: 'twitter' });
    if (options.json && jobs.length !== 1) throw new Error('--json requires a single-source run.');
    run = { version: 1, id: `${new Date().toISOString().replace(/[:.]/g, '-')}-${randomUUID().slice(0, 8)}`, createdAt: new Date().toISOString(), status: 'pending', summarize: !values['scrape-only'], options, jobs };
  }
  const directory = runPath(run.id);
  const release = acquireLock(directory);
  const save = () => writeJson(path.join(directory, 'run.json'), run);
  try {
    if (command === 'run' && run.options.system) {
      const prompt = path.join(directory, 'system-prompt.txt');
      fs.copyFileSync(run.options.system, prompt, fs.constants.COPYFILE_EXCL);
      run.options.system = prompt;
    }
    run.status = 'running'; save();
    console.log(`Run: ${run.id}\nRecovery: pnpm scrape resume ${run.id}`);
    const errors: string[] = [];
    for (const job of run.jobs) {
      const jobDir = path.join(directory, job.name);
      fs.mkdirSync(jobDir, { recursive: true });
      try {
        if (!job.scraped) {
          if (run.options.json) {
            const snapshot = path.join(jobDir, 'raw.json');
            const input = readJson<unknown>(fs.existsSync(snapshot) ? snapshot : run.options.json);
            if (job.source === 'reddit' ? !Array.isArray(input) : !input || !Array.isArray((input as { tweets?: unknown }).tweets)) throw new Error('Input is not raw data for this source.');
            job.raw = path.join(jobDir, 'raw.json'); writeJson(job.raw, input);
          } else if (job.source === 'reddit') job.raw = await scrapeReddit(jobDir, job.subreddits!, job.minimum!);
          else {
            const { scrapeTarget } = await import('./twitter/scrapeNitterList');
            const target = run.options.target;
            const result = await scrapeTarget({ pageType: /^\d+$/.test(target) ? 'list' : 'profile', identifier: target.replace(/^@/, '') }, {
              start: new Date(run.options.start), end: new Date(run.options.end), baseUrl: run.options.baseUrl,
              maxPages: Number(run.options.maxPages), mockHtmlDir: run.options.mockHtmlDir,
              outputDir: jobDir, checkpointPath: path.join(jobDir, 'pages.json'),
            });
            job.raw = path.join(jobDir, 'raw.json'); writeJson(job.raw, result);
            if (result.diagnostics.wasPartial) throw new Error('Twitter coverage incomplete; resume (increase --maxPages if capped).');
          }
          job.scraped = true; save();
        }
        if (run.summarize && !job.summarized) {
          const keys = job.source === 'reddit' ? ['clusters', 'top-k', 'max-per-cluster', 'min-per-cluster'] : ['model', 'focusTopic', 'system', 'maxChars'];
          const args = keys.flatMap(key => run.options[key] ? [`--${key}`, run.options[key]] : []);
          await child(job.source === 'reddit' ? 'reddit/clusterWithLLM.ts' : 'twitter/processNitterListWithResponses.ts',
            job.source === 'reddit' ? ['--file', job.raw!, ...args] : ['--json', job.raw!, '--out', path.join(jobDir, 'reports'), ...args], jobDir);
          job.summarized = true; save();
        }
      } catch (error) { console.error(`[${job.name}] ${(error as Error).message}`); errors.push(job.name); }
    }
    run.status = errors.length ? 'incomplete' : 'complete'; save();
    if (errors.length) throw new Error(`Incomplete: ${errors.join(', ')}. Retry: pnpm scrape resume ${run.id}`);
    console.log(`Complete: ${directory}`);
  } finally { release(); }
}
main().catch(error => { console.error(error instanceof Error ? error.message : String(error)); process.exitCode = 1; });
