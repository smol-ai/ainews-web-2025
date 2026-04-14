import { spawn } from 'node:child_process';
import { createWriteStream, existsSync } from 'node:fs';
import { copyFile, mkdir, readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const startedAt = Date.now();
const scope = process.env.VERCEL_SCOPE || 'smol-ai';
const project = process.env.VERCEL_PROJECT || 'ainews-web-2025';
const outputDir = '.vercel/output';
const vercelCommand = process.env.VERCEL_CLI || 'vercel';
const verboseDeploy = process.env.DEPLOY_VERBOSE === 'true';
const assetVersion = process.env.DEPLOY_ASSET_VERSION || String(startedAt);
const deployLogDir = '.vercel/deploy-logs';
const deployLogPath = join(deployLogDir, `deploy-${new Date(startedAt).toISOString().replace(/[:.]/g, '-')}.log`);

function elapsed() {
  return `${((Date.now() - startedAt) / 1000).toFixed(1)}s`;
}

function stripAnsi(value) {
  return value.replace(/\u001b\[[0-9;?]*[ -/]*[@-~]/g, '');
}

function shouldPrintLine(line) {
  const clean = stripAnsi(line).trim();
  if (!clean) return false;

  if (verboseDeploy) return true;
  if (/^\d{2}:\d{2}:\d{2}\s+├─ /.test(clean)) return false;
  if (/^\d{2}:\d{2}:\d{2}\s+└─ /.test(clean)) return false;

  return [
    /\b(error|failed|failure|warning|warn)\b/i,
    /^Detected `pnpm-lock\.yaml`/,
    /^Generated \d+ frozen issue pages/,
    /^\d{2}:\d{2}:\d{2} \[(content|types|check|build|pagefind|@astrojs\/vercel|@astrojs\/sitemap)\]/,
    /^\d{2}:\d{2}:\d{2} ✓ Completed in/,
    /^\d{2}:\d{2}:\d{2} ▶ /,
    /^- \d+ errors?/,
    /^- \d+ warnings?/,
    /^Step complete in/,
    /^Uploading \[/,
    /^Inspect:/,
    /^Production:/,
    /^Building:/,
    /^Completing/,
    /^Aliased:/,
    /^Deployment completed/,
    /^Build completed successfully/,
    /^Deploying /,
    /^Retrieving project/,
  ].some((pattern) => pattern.test(clean));
}

function run(command, args, options = {}) {
  const commandText = [command, ...args].join(' ');
  const stepStart = Date.now();
  console.log(`\n$ ${commandText}`);
  console.log(`Full log: ${deployLogPath}`);

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
      ...options,
    });
    const logStream = createWriteStream(deployLogPath, { flags: 'a' });
    const pending = { stdout: '', stderr: '' };

    logStream.write(`\n$ ${commandText}\n`);

    function handleOutput(source, chunk) {
      const text = chunk.toString();
      logStream.write(text);

      if (verboseDeploy) {
        process[source].write(text);
        return;
      }

      pending[source] += text.replace(/\r/g, '\n');
      const lines = pending[source].split('\n');
      pending[source] = lines.pop() ?? '';

      for (const line of lines) {
        if (shouldPrintLine(line)) {
          process[source].write(`${line}\n`);
        }
      }
    }

    child.stdout.on('data', (chunk) => handleOutput('stdout', chunk));
    child.stderr.on('data', (chunk) => handleOutput('stderr', chunk));

    child.on('error', reject);
    child.on('close', (code) => {
      for (const source of ['stdout', 'stderr']) {
        if (pending[source] && shouldPrintLine(pending[source])) {
          process[source].write(`${pending[source]}\n`);
        }
      }
      logStream.end();
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

async function fixVercelRoutes() {
  const configPath = join(outputDir, 'config.json');
  const config = JSON.parse(await readFile(configPath, 'utf-8'));
  const routes = config.routes ?? [];
  const filesystemRoute = routes.find((route) => route.handle === 'filesystem');
  const filesystemIndex = routes.findIndex((route) => route.handle === 'filesystem');
  const first404Index = routes.findIndex((route) => route.status === 404);

  if (!filesystemRoute) {
    throw new Error('Vercel output config has no filesystem route to repair.');
  }

  const staticRoutes = [
    { src: '^/_astro/(.*)$', dest: '/_astro/$1' },
    { src: '^/pagefind/(.*)$', dest: '/pagefind/$1' },
    { src: '^/favicon\\.ico$', dest: '/favicon.ico' },
  ];
  const isGeneratedStaticRoute = (route) => {
    return typeof route.src === 'string' && [
      '^/_astro/(.*)$',
      '^/pagefind/(.*)$',
      '^/favicon\\.ico$',
    ].includes(route.src);
  };

  config.routes = [
    ...staticRoutes,
    filesystemRoute,
    ...routes.filter((route) => route.handle !== 'filesystem' && !isGeneratedStaticRoute(route)),
  ];
  await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`);
  console.log(`Moved Vercel static/filesystem routes before error/404 routes: filesystem ${filesystemIndex} -> 3, first 404 was ${first404Index}.`);
}

async function findFiles(directory, predicate) {
  const results = [];
  const entries = await readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      results.push(...await findFiles(entryPath, predicate));
    } else if (entry.isFile() && predicate(entryPath, entry.name)) {
      results.push(entryPath);
    }
  }
  return results;
}

function addVersionToAssetUrl(url) {
  if (url.includes('?')) {
    return url;
  }
  return `${url}?v=${assetVersion}`;
}

async function patchCachedAssetUrls() {
  const sourcePath = join(outputDir, 'static', '_astro', 'index.ljMySSss.css');
  const versionedPagefindName = `pagefind-astro-ui-${assetVersion}.css`;
  const versionedPath = join(outputDir, 'static', 'pagefind', versionedPagefindName);
  const astroCssFiles = await findFiles(join(outputDir, 'static', '_astro'), (_path, name) => name.endsWith('.css'));

  if (existsSync(sourcePath)) {
    await copyFile(sourcePath, versionedPath);
  }

  let patchedFiles = 0;
  const htmlFiles = await findFiles(join(outputDir, 'static'), (_path, name) => name.endsWith('.html'));
  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf-8');
    let updated = html;

    if (existsSync(sourcePath)) {
      updated = updated.replaceAll('/_astro/index.ljMySSss.css', `/pagefind/${versionedPagefindName}`);
      updated = updated.replaceAll('/pagefind/pagefind-astro-ui-2026-04-08.css', `/pagefind/${versionedPagefindName}`);
    }

    updated = updated.replace(/(href=")(\/_astro\/[^"?]+\.css)(")/g, (_match, prefix, url, suffix) => {
      return `${prefix}${addVersionToAssetUrl(url)}${suffix}`;
    });
    updated = updated.replace(/(href=")(\/pagefind\/[^"?]+\.css)(")/g, (_match, prefix, url, suffix) => {
      return `${prefix}${addVersionToAssetUrl(url)}${suffix}`;
    });

    if (updated !== html) {
      await writeFile(htmlFile, updated);
      patchedFiles += 1;
    }
  }

  console.log(`Versioned CSS asset URLs in ${patchedFiles} HTML files with v=${assetVersion}.`);
  console.log(`Found ${astroCssFiles.length} Astro CSS files in output.`);
}

async function verifyLocalCssReferences() {
  const htmlFiles = await findFiles(join(outputDir, 'static'), (_path, name) => name.endsWith('.html'));
  const missing = new Set();

  for (const htmlFile of htmlFiles) {
    const html = await readFile(htmlFile, 'utf-8');
    const matches = html.matchAll(/href="(\/(?:_astro|pagefind)\/[^"?]+\.css)(?:\?[^"]*)?"/g);
    for (const match of matches) {
      const assetPath = join(outputDir, 'static', match[1]);
      if (!existsSync(assetPath)) {
        missing.add(match[1]);
      }
    }
  }

  if (missing.size > 0) {
    throw new Error(`Missing CSS assets referenced by HTML: ${[...missing].join(', ')}`);
  }

  console.log(`Verified CSS references across ${htmlFiles.length} HTML files.`);
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

async function verifyProjectLink() {
  const linkPath = '.vercel/project.json';
  const link = JSON.parse(await readFile(linkPath, 'utf-8'));

  if (link.projectName !== project) {
    throw new Error(`${linkPath} is linked to ${link.projectName}, expected ${project}. Run: ${vercelCommand} link --yes --scope ${scope} --project ${project}`);
  }

  console.log(`Verified local Vercel link targets ${scope}/${project}`);
}

try {
  await mkdir(deployLogDir, { recursive: true });
  console.log(`Deploying to Vercel project: ${scope}/${project}`);
  console.log(`Using Vercel CLI command: ${vercelCommand}`);
  console.log(`Using deploy asset version: ${assetVersion}`);
  await verifyScope();
  await verifyProjectLink();
  await run(vercelCommand, ['build', '--yes', '--scope', scope, '--target', 'production']);
  await fixVercelRoutes();
  await patchCachedAssetUrls();
  await verifyLocalCssReferences();
  await verifyOutput();
  await run(vercelCommand, ['deploy', '--prebuilt', '--scope', scope, '--target', 'production', '--archive=tgz']);
  console.log(`\nDeploy complete. Total runtime: ${elapsed()}`);
} catch (error) {
  console.error(`\nDeploy failed after ${elapsed()}`);
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
