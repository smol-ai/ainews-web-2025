# AINews - Weekday recaps of top News for AI Engineers

AINews is a daily newsletter that summarizes top AI discords, reddits, and X/Twitter posts, delivering a comprehensive roundup to AI engineers every weekday.

## Features

- Daily AI news roundups
- Curated from top AI communities
- Easy to read and digest
- Focused on practical AI engineering content

## Getting Started

For weekday Reddit and Twitter/Nitter collection, use the [local scrapers](tools/scrapers/README.md). They share run tracking and checkpoint recovery:

```sh
pnpm --dir tools/scrapers install --frozen-lockfile
pnpm scrape run daily
pnpm scrape resume RUN_ID
```

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

5. daily - bash updates
```sh
 pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/*.md | head -n 1)" && gadmit "latest post" && gpom
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

The Cloudflare Worker is `smol-news` in account
`2d017c943ff16e4c52783635ef05e535`, serving the `news.smol.ai` site.
The Astro 5 adapter emits static pages and Pagefind assets to `dist/`, with
server endpoints in `dist/_worker.js/index.js`. The `.assetsignore` file excludes
server code from public assets. OG routes retain the existing React layouts and
bundle Noto Sans locally for rendering with `workers-og`.

```sh
pnpm install --frozen-lockfile
pnpm build
pnpm preview --port 8792
# In another terminal:
pnpm check:cloudflare http://127.0.0.1:8792
```

Deploy the verified build with `pnpm exec wrangler deploy`, or build and deploy
with `pnpm run deploy`. Before deployment, verify the Cloudflare account and Worker
name match `wrangler.jsonc`. No custom domain is attached by the initial config;
verify the Workers preview before moving the production hostname.

After deployment, run `pnpm check:cloudflare https://news.smol.ai` and inspect the
homepage's linked CSS, JavaScript and fonts. Existing origin cache rules may
require a host-specific cache purge during the migration.

Manual deployments use `.github/workflows/cloudflare.yml` (workflow_dispatch only).
Automatic publishing is not enabled until credentials are configured. Configure the repository
secret `CLOUDFLARE_API_TOKEN` with Worker deployment access to this account.
The content freshness check is unchanged; intentional archived snapshot builds
can explicitly set `BYPASS_RECENT_CONTENT_CHECK=true`.

## run script

```bash
pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/*.md | head -n 1)" && gadmit "latest post" && gpom

## or

pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/25-08-13*.md | head -n 1)" && gadmit "latest post" && gpom

```
