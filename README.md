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
npm install
```

3. Start the development server:
```sh
npm run dev
```

4. Build for production:
```sh
npm run build
```

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

The site can be deployed to Vercel or Netlify with a single click:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/smol_ai/ainews-web)
[![Deploy with Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/smol_ai/ainews-web)

## run script

```bash
pnpm ts-node oneoffs/process-emails.ts --file "$(ls -t src/content/issues/*.md | head -n 1)" && gadmit "latest post" && gpom
```