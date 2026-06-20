import { createHash } from 'node:crypto';
import { access, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import pLimit from 'p-limit';
import { remark } from 'remark';
import html from 'remark-html';

const sourceDir = 'src/content/frozen-issues';
const outputDir = 'public/frozen-issues';
const manifestPath = '.vercel/cache/frozen-issues-manifest.json';
const templateVersion = 'inline-css-v4-modern-archive-no-summary';
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

function readList(frontmatter, key) {
  const lines = frontmatter.split('\n');
  const index = lines.findIndex((line) => line.startsWith(`${key}:`));
  if (index === -1) {
    return [];
  }

  const values = [];
  for (let i = index + 1; i < lines.length; i += 1) {
    const line = lines[i];
    if (!line.startsWith('  - ')) {
      break;
    }
    values.push(line.slice(4).trim().replace(/^['"]|['"]$/g, ''));
  }
  return values.filter(Boolean);
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

function stripMarkdown(value) {
  return value
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function buildOgImagePath({ title, description, date }) {
  const params = new URLSearchParams({
    title,
    description,
  });
  if (date) {
    params.set('date', date);
  }
  return `/api/og?${params.toString()}`;
}

function renderTagSection(label, values) {
  if (!values.length) return '';
  const tags = values
    .map((value) => `<a href="/tags/${encodeURIComponent(value)}">${escapeHtml(value)}</a>`)
    .join('');
  return `<div class="tag-group"><h2>${label}</h2><div>${tags}</div></div>`;
}

function pageTemplate({ title, description, canonicalPath, body, date, companies, models, topics, people }) {
  const ogImagePath = buildOgImagePath({ title, description, date });
  const tagSections = [
    renderTagSection('Companies', companies),
    renderTagSection('Models', models),
    renderTagSection('Topics', topics),
    renderTagSection('People', people),
  ].filter(Boolean).join('');

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(description)}">
  <link rel="canonical" href="https://news.smol.ai${canonicalPath}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${escapeHtml(title)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="https://news.smol.ai${canonicalPath}">
  <meta property="og:image" content="https://news.smol.ai${ogImagePath}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${escapeHtml(title)}">
  <meta name="twitter:description" content="${escapeHtml(description)}">
  <meta name="twitter:image" content="https://news.smol.ai${ogImagePath}">
  <style>
    :root{color-scheme:light;--bg:#f5f5f5;--fg:#111;--muted:#737373;--line:rgb(0 0 0 / .15);--soft:rgb(0 0 0 / .05);--soft-strong:rgb(0 0 0 / .08);--link:#111;--max:1536px;--content:960px}
    *{box-sizing:border-box}
    html{scroll-padding-top:100px}
    body{margin:0;background:var(--bg);color:rgb(0 0 0 / .75);font-family:"Geist Sans",ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.65;-webkit-font-smoothing:antialiased;overflow-wrap:break-word}
    .site-header{position:fixed;top:0;left:0;right:0;z-index:10;padding:24px 0;background:rgb(245 245 245 / .75);backdrop-filter:blur(8px) saturate(1.8)}
    .container{max-width:var(--max);margin:0 auto;padding:0 16px}
    .site-header .container{display:flex;align-items:center;justify-content:space-between;gap:16px}
    .brand{display:inline-block;background:#000;color:#fff;padding:0 8px;font-weight:600;text-decoration:none}
    .site-nav{display:flex;align-items:center;gap:6px;font-size:14px}
    .site-nav a{color:inherit;text-decoration:underline;text-underline-offset:3px;text-decoration-color:rgb(0 0 0 / .3)}
    .site-nav span{color:var(--muted)}
    .page{padding:128px 0 80px}
    .archive-back{display:inline-flex;align-items:center;gap:8px;margin-bottom:32px;color:var(--muted);font-size:14px;text-decoration:none}
    .archive-back:hover{color:var(--fg)}
    .issue-shell{max-width:1280px;margin:0 auto}
    .issue-hero{display:flex;flex-direction:column;gap:16px;margin-bottom:24px;padding-bottom:32px}
    .eyebrow{color:#9ca3af;font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:14px;letter-spacing:.06em;text-transform:uppercase}
    h1{margin:0;color:var(--fg);font-size:clamp(2rem,3vw,3rem);font-weight:600;line-height:1.1;letter-spacing:0}
    .archive-note{display:flex;flex-wrap:wrap;gap:10px 12px;align-items:center;margin-top:8px;color:var(--muted);font-size:14px}
    .archive-note a{color:inherit;text-decoration:underline;text-underline-offset:3px;text-decoration-color:rgb(0 0 0 / .3)}
    .archive-tags{display:flex;flex-direction:column;gap:12px;margin-top:12px}
    .tag-group h2{margin:0 0 4px;color:var(--muted);font-size:12px;font-weight:500;letter-spacing:.05em;text-transform:uppercase}
    .tag-group div{display:flex;flex-wrap:wrap;gap:8px}
    .tag-group a{border:1px solid rgb(0 0 0 / .1);border-radius:8px;background:var(--soft);padding:4px 8px;color:var(--fg);font-size:12px;text-decoration:none}
    .tag-group a:hover{background:var(--soft-strong)}
    .content-wrap{border-top:1px solid var(--line);padding-top:32px}
    article{max-width:var(--content);margin:0 auto;padding:0 16px;color:rgb(0 0 0 / .78);font-size:16px;line-height:1.75}
    article p{margin:1.5rem 0}
    article h1,article h2,article h3,article h4{color:var(--fg);font-weight:600;line-height:1.2;letter-spacing:0}
    article h1{font-size:2rem;margin:3rem 0 1rem}
    article h2{font-size:1.5rem;margin:3rem 0 1rem}
    article h3{font-size:1.25rem;margin:2rem 0 1rem}
    article h4{font-size:1rem;margin:1.75rem 0 .75rem}
    article a{color:var(--link);text-decoration:underline;text-underline-offset:3px;text-decoration-color:rgb(0 0 0 / .3)}
    article a:hover{text-decoration-color:rgb(0 0 0 / .55)}
    article ul,article ol{margin:1.5rem 0;padding-left:1.5rem}
    article li{margin:.5rem 0;padding-left:.35rem}
    article li>ul,article li>ol{margin:.5rem 0}
    article blockquote{margin:1.5rem 0;border-left:4px solid rgb(0 0 0 / .2);padding-left:1rem;color:rgb(0 0 0 / .68);font-style:italic}
    article hr{border:0;border-top:1px solid var(--line);margin:3rem 0}
    article pre{margin:1.5rem 0;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fafafa;padding:1rem}
    article code{border-radius:4px;background:var(--soft);padding:.125rem .375rem;font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:.9em}
    article pre code{background:transparent;padding:0}
    article img{display:block;max-width:100%;height:auto;max-height:300px;object-fit:contain;margin:2rem auto;border-radius:8px}
    article table{display:block;width:100%;overflow-x:auto;border-collapse:collapse}
    article th,article td{border:1px solid var(--line);padding:.5rem;vertical-align:top}
    .site-footer{padding:48px 0;color:var(--muted);font-size:14px}
    .site-footer .container{display:flex;align-items:center;justify-content:space-between;gap:16px;border-top:1px solid var(--line);padding-top:24px}
    .site-footer a{color:inherit;text-decoration:underline;text-underline-offset:3px}
    @media (min-width:640px){.container{padding:0 24px}}
    @media (min-width:768px){.container{padding:0 32px}}
    @media (min-width:1024px){.container{padding:0 48px}article{padding:0 24px}}
    @media (max-width:640px){.site-header{padding:18px 0}.site-header .container{align-items:flex-start}.site-nav{flex-wrap:wrap;justify-content:flex-end}.page{padding-top:112px}.site-footer .container{align-items:flex-start;flex-direction:column}}
    @media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#171717;--fg:#fff;--muted:#a3a3a3;--line:rgb(255 255 255 / .15);--soft:rgb(255 255 255 / .06);--soft-strong:rgb(255 255 255 / .1);--link:#fff}body{color:rgb(255 255 255 / .76)}.site-header{background:rgb(23 23 23 / .75)}.brand{background:#fff;color:#000}.site-nav a,.archive-note a{color:inherit;text-decoration-color:rgb(255 255 255 / .3)}.tag-group a{border-color:rgb(255 255 255 / .1)}article{color:rgb(255 255 255 / .78)}article a{text-decoration-color:rgb(255 255 255 / .35)}article a:hover{text-decoration-color:rgb(255 255 255 / .65)}article blockquote{border-color:rgb(255 255 255 / .22);color:rgb(255 255 255 / .68)}article pre{background:#0f0f0f}.brand:focus-visible,.site-nav a:focus-visible,.archive-back:focus-visible,article a:focus-visible{outline:2px solid #fff;outline-offset:3px}}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="container">
      <a class="brand" href="/">AINews</a>
      <nav class="site-nav" aria-label="Primary navigation">
        <a href="/subscribe">subscribe</a><span>/</span>
        <a href="/issues/">issues</a><span>/</span>
        <a href="/tags">tags</a>
      </nav>
    </div>
  </header>
  <main class="page">
    <div class="container">
      <a class="archive-back" href="/issues">&larr; Back to issues</a>
      <section class="issue-shell">
        <header class="issue-hero">
          <div class="eyebrow">${date ? escapeHtml(date.slice(0, 10)) : 'Frozen AI News archive'}</div>
      <h1>${escapeHtml(title)}</h1>
          <div class="archive-note">
            <span>Static archive snapshot</span>
            <span aria-hidden="true">/</span>
            <a href="${canonicalPath}">Canonical issue URL</a>
          </div>
          ${tagSections ? `<div class="archive-tags" aria-label="Article metadata">${tagSections}</div>` : ''}
        </header>
        <div class="content-wrap">
          <article>${body}</article>
        </div>
      </section>
    </div>
  </main>
  <footer class="site-footer">
    <div class="container">
      <div>&copy; ${new Date().getFullYear()} &bull; AINews</div>
      <div>You can also subscribe by <a href="/rss.xml">rss</a>.</div>
    </div>
  </footer>
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
  const rawDescription = readScalar(frontmatter, 'description') || 'AI News archive issue';
  const description = stripMarkdown(rawDescription);
  const date = readScalar(frontmatter, 'date');
  const companies = readList(frontmatter, 'companies');
  const models = readList(frontmatter, 'models');
  const topics = readList(frontmatter, 'topics');
  const people = readList(frontmatter, 'people');
  const rendered = String(await processor.process(body));
  const output = pageTemplate({
    title,
    description,
    canonicalPath: `/issues/${id}`,
    body: rendered,
    date,
    companies,
    models,
    topics,
    people,
  });
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
