import { createHash } from 'node:crypto';
import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import pLimit from 'p-limit';
import { remark } from 'remark';
import html from 'remark-html';

const sourceDir = 'src/content/frozen-issues';
const outputDir = 'public/frozen-issues';
const manifestPath = '.vercel/cache/frozen-issues-manifest.json';
const templateVersion = 'inline-css-v2';
const concurrency = Number(process.env.FROZEN_ISSUE_CONCURRENCY || 8);
const processor = remark().use(html);

function hashSource(contents) {
  return createHash('sha256')
    .update(templateVersion)
    .update('\0')
    .update(contents)
    .digest('hex');
}

async function readManifest() {
  const contents = await readFile(manifestPath, 'utf-8').catch((error) => {
    if (error && error.code === 'ENOENT') return null;
    throw error;
  });
  if (!contents) return { files: {} };
  return JSON.parse(contents);
}

function splitFrontmatter(contents) {
  const match = contents.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/);
  if (!match) {
    return { frontmatter: '', body: contents };
  }
  return { frontmatter: match[1], body: match[2] };
}

function readScalar(frontmatter, key) {
  const lines = frontmatter.split('\n');
  const index = lines.findIndex((line) => line.startsWith(`${key}:`));
  if (index === -1) {
    return '';
  }

  const firstValue = lines[index].slice(key.length + 1).trim();
  if (firstValue === '>-') {
    const values = [];
    for (let i = index + 1; i < lines.length; i += 1) {
      const line = lines[i];
      if (!line.startsWith('  ')) {
        break;
      }
      values.push(line.trim());
    }
    return values.join(' ');
  }

  return firstValue.replace(/^['"]|['"]$/g, '');
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function pageTemplate({ title, description, canonicalPath, body }) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(description)}">
  <link rel="canonical" href="https://news.smol.ai${canonicalPath}">
  <style>
    body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#fff;color:#111;line-height:1.65}
    main{max-width:820px;margin:0 auto;padding:32px 20px 80px}
    header{border-bottom:1px solid #e5e5e5;margin-bottom:32px;padding-bottom:24px}
    a{color:#2563eb;text-decoration:underline;text-underline-offset:2px}
    h1{font-size:2.25rem;line-height:1.1;margin:0 0 12px}
    h2{font-size:1.6rem;margin-top:2.2rem}
    h3{font-size:1.25rem;margin-top:1.8rem}
    blockquote{border-left:4px solid #ddd;margin-left:0;padding-left:1rem;color:#444}
    pre{overflow:auto;background:#f6f8fa;padding:1rem;border-radius:8px}
    code{background:#f6f8fa;padding:.1rem .25rem;border-radius:4px}
    pre code{padding:0;background:transparent}
    img{max-width:100%;height:auto}
    .eyebrow{font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:#666;margin-bottom:12px}
    @media (prefers-color-scheme: dark){body{background:#0a0a0a;color:#f5f5f5}header{border-color:#333}blockquote{border-color:#444;color:#ccc}pre,code{background:#171717}a{color:#93c5fd}}
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Frozen AI News archive</div>
      <h1>${escapeHtml(title)}</h1>
      <p>${escapeHtml(description)}</p>
      <p><a href="${canonicalPath}">Canonical issue URL</a></p>
    </header>
    <article>${body}</article>
  </main>
</body>
</html>`;
}

const started = Date.now();
await mkdir(outputDir, { recursive: true });
await mkdir(join('.vercel', 'cache'), { recursive: true });
const manifest = await readManifest();
const files = (await readdir(sourceDir)).filter((file) => /\.(md|mdx)$/.test(file));
const limit = pLimit(concurrency);

async function generateIssue(file) {
  const id = basename(file).replace(/\.(md|mdx)$/, '');
  const contents = await readFile(join(sourceDir, file), 'utf-8');
  const sourceHash = hashSource(contents);
  const outputPath = join(outputDir, `${id}.html`);

  if (manifest.files[file] === sourceHash) {
    const outputExists = await access(outputPath).then(() => true).catch((error) => {
      if (error && error.code === 'ENOENT') return false;
      throw error;
    });
    if (outputExists) {
      return { changed: false, skipped: true, file, sourceHash };
    }
  }

  const { frontmatter, body } = splitFrontmatter(contents);
  const title = readScalar(frontmatter, 'title') || id;
  const description = readScalar(frontmatter, 'description') || 'AI News archive issue';
  const rendered = String(await processor.process(body));
  const output = pageTemplate({ title, description, canonicalPath: `/issues/${id}`, body: rendered });
  const existing = await readFile(outputPath, 'utf-8').catch((error) => {
    if (error && error.code === 'ENOENT') return null;
    throw error;
  });

  if (existing === output) {
    return { changed: false, skipped: false, file, sourceHash };
  }

  await writeFile(outputPath, output);
  return { changed: true, skipped: false, file, sourceHash };
}

const results = await Promise.all(files.map((file) => limit(() => generateIssue(file))));
const changed = results.filter((result) => result.changed).length;
const skipped = results.filter((result) => result.skipped).length;
const rendered = results.length - skipped;
const nextManifest = {
  templateVersion,
  files: Object.fromEntries(results.map((result) => [result.file, result.sourceHash])),
};

await writeFile(manifestPath, `${JSON.stringify(nextManifest, null, 2)}\n`);

console.log(`Generated ${results.length} frozen issue pages in ${((Date.now() - started) / 1000).toFixed(1)}s (${changed} updated, ${rendered - changed} unchanged, ${skipped} skipped, concurrency ${concurrency})`);
