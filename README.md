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

```sh
vercel --prod --scope smol-ai
```

For local deploys, prefer the checked script because it deploys Vercel's prebuilt output with `--archive=tgz` and verifies static assets will be served before uploading:

```sh
pnpm deploy:vercel
```

Override the scope if needed:

```sh
VERCEL_SCOPE=smol-ai pnpm deploy:vercel
```

The script intentionally verifies that the `ainews-web-2025` project exists under the selected scope before deploying. If the deploy output mentions `swyxs-projects/ainews-web`, it targeted the wrong project and will not update `news.smol.ai`.

The script uses the global Vercel CLI by default on this machine because the repo-local `vercel` devDependency is older. Override the binary if needed:

```sh
VERCEL_CLI=vercel pnpm deploy:vercel
```

Important deployment notes:

- `vercel.json` intentionally does not set `outputDirectory`; the `@astrojs/vercel` adapter emits Vercel Build Output under `.vercel/output`.
- `src/pages/api/og-alt.ts` is server-rendered (`prerender = false`) to avoid build-time network fetches from the OG image renderer.
- If production returns a Vercel `NOT_FOUND`, confirm the deploy is targeting the `smol-ai/ainews-web-2025` project rather than a personal `swyxs-projects/ainews-web` project.

The site can also be deployed to Vercel or Netlify with a single click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/smol_ai/ainews-web)
[![Deploy with Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/smol_ai/ainews-web)

## run script

```bash
pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/*.md | head -n 1)" && gadmit "latest post" && gpom

## or

pnpm tsx oneoffs/process-emails.ts --file "$(ls -t src/content/issues/25-08-13*.md | head -n 1)" && gadmit "latest post" && gpom

```
