# AINews - Weekday recaps of top News for AI Engineers

AINews is a daily newsletter that summarizes top AI discords, reddits, and X/Twitter posts, delivering a comprehensive roundup to AI engineers every weekday.

## Features

- Daily AI news roundups
- Curated from top AI communities
- Easy to read and digest
- Focused on practical AI engineering content

## Getting Started

1. Clone the repository:
```sh
git clone https://github.com/smol_ai/ainews-web.git
```

2. Install dependencies:
```sh
cd ainews-web
pnpm install
```

3. Start the development server:
```sh
pnpm dev
```

4. Build for production:
```sh
pnpm build
```

## Content archive strategy

To keep Vercel builds under the 8 GB build-memory limit, issues before 2026 are frozen as static HTML instead of being fully rendered through Astro content collections on every build.

- Metadata for pre-2026 posts stays in `src/content/issues/` so listings, tags, RSS metadata, and search facets still work.
- Full pre-2026 Markdown bodies are preserved in `src/content/frozen-issues/`.
- Static HTML snapshots are generated into `public/frozen-issues/` by `scripts/generate-frozen-issues.mjs`.
- `pnpm build` runs the frozen-issue generator before `astro check && astro build`.
- Pre-2026 `/issues/<slug>/` routes redirect to `/frozen-issues/<slug>.html`; 2026+ issues continue to render through Astro normally.
- `src/content/oldissues/` was removed because it duplicated archived content and inflated the Astro content store.

If you edit an archived pre-2026 issue, update the preserved file in `src/content/frozen-issues/`, then run:

```sh
node scripts/generate-frozen-issues.mjs
pnpm build
```

Recent build timing after freezing: frozen HTML generation takes about 45 seconds for 538 archived posts, Astro content sync takes under 1 second, and a full production build completes in about 2–3 minutes locally instead of failing with Vercel out-of-memory errors during server entrypoint bundling.

## Newsletter Subscription

AINews includes a newsletter subscription functionality using Resend. To set up the subscription feature:

1. Create a Resend account at [resend.com](https://resend.com)
2. Set up an audience in Resend to collect subscribers
3. Add the following environment variables to your `.env` file:

```sh
RESEND_API_KEY=re_your_api_key_here
RESEND_AUDIENCE_ID=your_audience_id_here
```

The subscription system features:
- A dedicated `/subscribe` page (with `/signup` alias)
- An embedded form on the homepage
- Form validation and submission animations
- Success and error state handling

## Deployment

The production site is deployed on Vercel under the `smol-ai` scope as `ainews-web-2025` and serves `news.smol.ai`.

### Preferred production deploy

Use the checked deploy script for normal production deploys:

```sh
pnpm deploy:vercel
```

Why this is preferred:

- It forces the global Vercel CLI on this machine (`/Users/shawnwang/Library/pnpm/vercel`) so deploys do not accidentally use the old repo-local Vercel CLI.
- It targets the production project explicitly: `smol-ai/ainews-web-2025`.
- It verifies `.vercel/project.json` is linked to `ainews-web-2025` before building, so a deploy cannot silently go to `swyxs-projects/ainews-web`.
- It runs `vercel build` locally and deploys the generated `.vercel/output` with `vercel deploy --prebuilt`.
- It uses `--archive=tgz`, which avoids Vercel's per-file upload limit (`api-upload-free`) when the output contains thousands of static issue/search files.
- It validates and repairs the Vercel Build Output route order so static files are served before generated 404 routes.
- It rewrites known Cloudflare-poisoned asset URLs to versioned paths so cached 404s do not keep breaking CSS after a good deploy.

Observed timing from the April 2026 fix:

- Frozen issue generation: about 45 seconds for 538 archived pre-2026 issues.
- Full local Vercel build: about 160–165 seconds.
- Prebuilt upload: about 27–30 seconds for a 162–163 MB tgz upload.
- Vercel output deployment/aliasing: about 70–75 seconds.
- End-to-end `pnpm deploy:vercel`: about 236–240 seconds.

The script reports elapsed time for each major step and the total runtime.

### Scope and CLI overrides

Override the Vercel scope if needed, but production should normally stay on `smol-ai`:

```sh
VERCEL_SCOPE=smol-ai pnpm deploy:vercel
```

Override the project name only for intentional one-off deploys:

```sh
VERCEL_PROJECT=ainews-web-2025 pnpm deploy:vercel
```

Override the Vercel binary if you have already verified your shell resolves to a modern CLI:

```sh
VERCEL_CLI=vercel pnpm deploy:vercel
```

If the output mentions `swyxs-projects/ainews-web`, stop. That is the wrong Vercel project and will not update `news.smol.ai`. Relink the repo before deploying:

```sh
/Users/shawnwang/Library/pnpm/vercel link --yes --scope smol-ai --project ainews-web-2025
```

Expected `.vercel/project.json` shape:

```json
{
  "projectName": "ainews-web-2025"
}
```

### Fast prebuilt redeploy

If `vercel build` already succeeded and you only changed `.vercel/output` post-processing, redeploy the prebuilt output directly:

```sh
/Users/shawnwang/Library/pnpm/vercel deploy --prebuilt --prod --scope smol-ai --target production --archive=tgz
```

Expected timing:

- Upload: about 27–30 seconds for the current 162–163 MB tgz.
- Vercel output deployment/aliasing: about 60–75 seconds.
- Total: about 1.5–2 minutes.

Use this only when `.vercel/output` is already current. It skips `pnpm build`, so it will not regenerate frozen issues, Pagefind, Astro output, or route/asset patches from the source scripts unless those patches have already been applied to `.vercel/output`.

### Raw Vercel deploy

Avoid using raw `vercel --prod --scope smol-ai` for this repo unless you are intentionally debugging Vercel itself.

Problems with the raw command:

- It can use the locally linked project from `.vercel/project.json`; if that link points at `swyxs-projects/ainews-web`, the deploy will succeed but `news.smol.ai` will not change.
- Without `--archive=tgz`, it can hit Vercel's upload file-count/rate limit for this large static output.
- It can rebuild remotely on the Vercel 8 GB build machine, which previously failed out-of-memory while bundling server entrypoints.
- It does not run this repo's route-order and cache-busting checks before upload.

If you must run a raw deploy, prefer the prebuilt, archived form after a successful local build:

```sh
pnpm build
/Users/shawnwang/Library/pnpm/vercel deploy --prebuilt --prod --scope smol-ai --target production --archive=tgz
```

### Vercel configuration notes

- `vercel.json` intentionally does not set `outputDirectory`; the `@astrojs/vercel` adapter emits Vercel Build Output under `.vercel/output`.
- `src/pages/api/og-alt.ts` is server-rendered (`prerender = false`) to avoid build-time network fetches from the OG image renderer.
- Vercel project settings should show Node.js 22.x, framework preset Astro, root directory `.`, and no output directory override.
- The build output is large by design because pre-2026 issues are frozen as static HTML and Pagefind indexes thousands of pages. Recent output reports were about 529–530 MB on disk, compressed to about 162–163 MB for tgz upload.
- If production returns a Vercel `NOT_FOUND`, confirm the deploy is targeting `smol-ai/ainews-web-2025`, not a personal `swyxs-projects/ainews-web` project.
- If production HTML loads but CSS 404s, check for Cloudflare-cached 404s on immutable asset paths. The deploy script currently rewrites known affected assets to versioned URLs; if new hashed asset paths are poisoned, bump or version those URLs before redeploying.

### Post-deploy verification

After deploy, check the live page and the assets it references:

```sh
node -e 'const base="https://news.smol.ai"; const html=await (await fetch(base+"/",{headers:{"cache-control":"no-cache"}})).text(); const assets=[...new Set([...html.matchAll(/\/(?:_astro\/[^"'"'"'\s]+\.(?:css|js|woff2?)|pagefind\/[^"'"'"'\s]+\.(?:css|js)(?:\?[^"'"'"'\s]+)?|favicon\.ico(?:\?[^"'"'"'\s]+)?)/g)].map(m=>m[0]))]; for (const path of assets) { const res=await fetch(base+path,{headers:{"cache-control":"no-cache"}}); console.log(path, res.status, res.headers.get("content-type")); }'
```

Expected result: all referenced CSS, JS, font, Pagefind, and favicon assets should return `200`.

The site can also be deployed to Vercel or Netlify with a single click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/smol_ai/ainews-web)
[![Deploy with Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/smol_ai/ainews-web)

## run script

```bash
pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/*.md | head -n 1)" && gadmit "latest post" && gpom

## or

pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/25-08-13*.md | head -n 1)" && gadmit "latest post" && gpom

```
