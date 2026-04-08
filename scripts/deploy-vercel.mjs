import { spawn } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const startedAt = Date.now();
const scope = process.env.VERCEL_SCOPE || 'smol-ai';
const project = process.env.VERCEL_PROJECT || 'ainews-web-2025';
const outputDir = '.vercel/output';
const vercelCommand = process.env.VERCEL_CLI || 'vercel';

function elapsed() {
  return `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
}

function run(command, args, options = {}) {
  const commandText = [command, ...args].join(' ');
  const stepStart = Date.now();
  console.log(`\n$ ${commandText}`);

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
      shell: false,
      ...options,
    });

    child.on('error', reject);
    child.on('close', (code) => {
      const seconds = ((Date.now() - stepStart) / 1000).toFixed(1);
      if (code === 0) {
        console.log(`Step complete in ${seconds}`);
        resolve();
      } else {
        reject(new Error(`${commandText} failed with exit code ${code} after ${seconds}`));
      }
    });
  });
}

async function directorySize(path) {
  const { readdir } = await import('node:fs/promises');
  let total = 0;
  const entries = await readdir(path, { withFileTypes: true });

  for (const entry of entries) {
    const entryPath = join(path, entry.name);
    if (entry.isDirectory()) {
      total += await directorySize(entryPath);
    } else if (entry.isFile()) {
      total += (await stat(entryPath)).size;
    }
  }

  return total;
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

async function verifyOutput() {
  if (!existsSync(outputDir)) {
    throw new Error(`${outputDir} does not exist. Run Vercel build first.`);
  }

  const configPath = join(outputDir, 'config.json');
  const config = JSON.parse(await readFile(configPath, 'utf-8'));
  const filesystemIndex = config.routes?.findIndex((route) => route.handle === 'filesystem') ?? -1;
  const first404Index = config.routes?.findIndex((route) => route.status === 404) ?? -1;

  if (filesystemIndex === -1) {
    throw new Error('Vercel output config has no filesystem route; static assets may 404.');
  }

  if (first404Index !== -1 && filesystemIndex > first404Index) {
    throw new Error('Vercel output config has filesystem after 404 route; static assets would 404.');
  }

  const size = await directorySize(outputDir);
  console.log(`Verified ${outputDir}: ${formatBytes(size)}, filesystem route index ${filesystemIndex}, first 404 index ${first404Index}`);
}

async function verifyScope() {
  const output = await new Promise((resolve, reject) => {
    const child = spawn(vercelCommand, ['project', 'ls', '--scope', scope], {
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk; });
    child.stderr.on('data', (chunk) => { stderr += chunk; });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve(`${stdout}\n${stderr}`);
      } else {
        reject(new Error(`Could not list Vercel projects for scope ${scope}: ${stderr || stdout}`));
      }
    });
  });

  if (!String(output).includes(project)) {
    throw new Error(`Project ${project} was not found under scope ${scope}. Refusing to deploy.`);
  }

  console.log(`Verified Vercel scope contains project ${scope}/${project}`);
}

try {
  console.log(`Deploying to Vercel project: ${scope}/${project}`);
  console.log(`Using Vercel CLI command: ${vercelCommand}`);
  await verifyScope();
  await run(vercelCommand, ['build', '--yes', '--scope', scope, '--target', 'production']);
  await verifyOutput();
  await run(vercelCommand, ['deploy', '--prebuilt', '--prod', '--scope', scope, '--target', 'production', '--archive=tgz']);
  console.log(`\nDeploy complete. Total runtime: ${elapsed()}`);
} catch (error) {
  console.error(`\nDeploy failed after ${elapsed()}`);
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
