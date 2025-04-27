# Vercel Build Retry GitHub Action

This GitHub Action automatically deploys your Astro website to Vercel and includes retry logic to handle temporary failures. It's specifically designed to work with the content date validation in your Astro build.

## How It Works

1. The workflow runs on pushes to the main branch, when PRs are merged to main, and can also be triggered manually.
2. It leverages GitHub Actions caching for Node.js modules (via `pnpm-lock.yaml`) to speed up dependency installation.
3. It attempts to deploy to Vercel up to 3 times with exponential backoff (60s, 120s, 240s between retries).
4. On retries, it sets `BYPASS_RECENT_CONTENT_CHECK=true` to potentially bypass the content date validation in your Astro build.

## Prerequisites

- **Vercel CLI:** This workflow assumes `vercel` is installed as a `devDependency` in your `package.json`.

## Required Secrets

You need to set up these secrets in your GitHub repository:

- `VERCEL_TOKEN`: Your Vercel authentication token.
- `VERCEL_ORG_ID`: Your Vercel organization ID.
- `VERCEL_PROJECT_ID`: Your Vercel project ID.

## How to Set Up Secrets

1. Generate a Vercel token:
   - Go to your [Vercel account settings](https://vercel.com/account/tokens).
   - Create a new token with appropriate permissions (e.g., deploy access).

2. Get your Organization ID and Project ID:
   - **Option 1 (Vercel CLI):**
     - Run `pnpm exec vercel whoami` to find your Organization ID.
     - Run `pnpm exec vercel link` (if not linked) and check `.vercel/project.json` for `orgId` and `projectId`.
     - Alternatively, use `pnpm exec vercel project ls` and `pnpm exec vercel project info <project-name>`.
   - **Option 2 (Vercel Dashboard):**
     - Find the `Project ID` in your Vercel project settings.
     - Find the `Organization ID` (or User ID for personal accounts) in your Vercel account/team settings.

3. Add the secrets to your GitHub repository:
   - Go to your repository on GitHub -> Settings -> Secrets and variables -> Actions.
   - Add each secret (`VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`) with its corresponding value.

## Manual Trigger

You can manually trigger this workflow by:
1. Going to the Actions tab in your GitHub repository.
2. Selecting the "Vercel Build with Retry Logic" workflow.
3. Clicking "Run workflow".

## Customization

- Adjust the trigger events (`on: ...`).
- Modify the number of retry attempts by changing `MAX_ATTEMPTS` in the workflow file.
- Adjust the initial delay (`DELAY`) and backoff strategy within the deployment script. 