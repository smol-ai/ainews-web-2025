import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { basename, join } from 'node:path';
import { remark } from 'remark';
import html from 'remark-html';

const sourceDir = 'src/content/frozen-issues';
const outputDir = 'public/frozen-issues';

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
  <link rel="stylesheet" href="/_astro/index.BNfmcC--.css">
  <link rel="stylesheet" href="/_astro/index.ljMySSss.css">
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
const files = (await readdir(sourceDir)).filter((file) => /\.(md|mdx)$/.test(file));
let count = 0;

for (const file of files) {
  const id = basename(file).replace(/\.(md|mdx)$/, '');
  const contents = await readFile(join(sourceDir, file), 'utf-8');
  const { frontmatter, body } = splitFrontmatter(contents);
  const title = readScalar(frontmatter, 'title') || id;
  const description = readScalar(frontmatter, 'description') || 'AI News archive issue';
  const rendered = String(await remark().use(html).process(body));
  await writeFile(
    join(outputDir, `${id}.html`),
    pageTemplate({ title, description, canonicalPath: `/issues/${id}`, body: rendered })
  );
  count += 1;
}

console.log(`Generated ${count} frozen issue pages in ${((Date.now() - started) / 1000).toFixed(1)}s`);
