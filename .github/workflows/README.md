# Vercel Build Retry GitHub Action

This GitHub Action automatically deploys your Astro website to Vercel and includes retry logic to handle temporary failures. It's specifically designed to work with the content date validation in your Astro build.

## How It Works

1. The workflow runs on a schedule (every 8 hours) and can also be triggered manually
2. It attempts to deploy to Vercel up to 3 times with exponential backoff (60s, 120s, 240s between retries)
3. On retries, it sets `BYPASS_RECENT_CONTENT_CHECK=true` to bypass the content date validation

## Required Secrets

You need to set up these secrets in your GitHub repository:

- `VERCEL_TOKEN`: Your Vercel authentication token
- `VERCEL_ORG_ID`: Your Vercel organization ID
- `VERCEL_PROJECT_ID`: Your Vercel project ID

## How to Set Up Secrets

1. Generate a Vercel token:
   - Go to your [Vercel account settings](https://vercel.com/account/tokens)
   - Create a new token with appropriate permissions

2. Get your Organization ID and Project ID:
   - Run `vercel whoami` to find your Organization ID
   - Run `vercel projects ls` to list projects
   - Run `vercel project info <project-name>` to get the Project ID

3. Add the secrets to your GitHub repository:
   - Go to your repository on GitHub
   - Click on "Settings" > "Secrets and variables" > "Actions"
   - Add each secret with the corresponding value

## Manual Trigger

You can manually trigger this workflow by:
1. Going to the Actions tab in your GitHub repository
2. Selecting the "Vercel Build with Retry Logic" workflow
3. Clicking "Run workflow"

## Customization

- Adjust the schedule by modifying the cron expression (`'0 */8 * * *'`)
- Change the number of retry attempts by modifying `MAX_ATTEMPTS`
- Adjust initial delay and backoff strategy in the workflow file 