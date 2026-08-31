# Local AI News scrapers

Reddit and Twitter/Nitter collection now live together here. This is a local Node/TypeScript tool, separate from the Astro site build. It uses system Google Chrome when Nitter asks for a browser challenge. Solve any CAPTCHA yourself in the window; a timeout leaves the run resumable.

## Setup

From the repository root:

```sh
pnpm --dir tools/scrapers install --frozen-lockfile
cp tools/scrapers/.env.example tools/scrapers/.env
pnpm scrape --help
```

Set `REDDIT_SESSION` and your AI credentials in that local `.env`. Existing process environment variables take precedence. Reddit supports the existing AI Gateway / LiteLLM settings; Twitter supports `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and LiteLLM settings. Credentials are never stored in run manifests. Google Chrome must be installed; no hosted browser service is required.

## Daily operation

```sh
pnpm scrape run daily
pnpm scrape run reddit-local
pnpm scrape run reddit-all
pnpm scrape run twitter --target 1585430245762441216
pnpm scrape run twitter --target @eugeneyan --start 3d
pnpm scrape run daily --scrape-only
```

`daily` runs both Reddit groups and Twitter in sequence. A failed group does not prevent the other groups from collecting. Runs summarize by default, which makes billable calls using the configured provider. Use `--scrape-only` to collect without AI calls. No scheduled task or automatic publication is installed.

Reddit preserves the source subreddit sets, score thresholds, low-volume subreddit overrides, comments, image analysis, clustering prompts, and post/comment summaries. Twitter preserves list/profile inputs, relative time windows, instance selection, focus topics, custom prompts, and JSON/Markdown summaries. Twitter defaults to the preceding 24 hours; use `--start 3d` for weekend coverage when desired. Reddit retains its original hot-feed and age-filter behavior, rather than pretending it supports historical cursor pagination.

## Recovery

Every new run prints its ID and recovery command. Run state lives in `tools/scrapers/runs/<ID>/run.json`; each source has raw data, checkpoints, and reports beneath that directory.

```sh
pnpm scrape status
pnpm scrape status RUN_ID
pnpm scrape resume RUN_ID
pnpm scrape resume RUN_ID --maxPages 300
pnpm scrape resume RUN_ID --maxChars 400000
pnpm scrape resume RUN_ID --summarize
```

- Fix the underlying problem (expired Reddit cookie, Nitter outage, browser challenge, provider credentials), then resume the same ID.
- Reddit resumes after the last completed subreddit, including genuinely empty results. Failed subreddit/comment requests remain incomplete and retryable.
- Twitter resumes from the saved next-page cursor and deduplicates tweet IDs. The original time window stays frozen. Page caps, invalid HTML, outages, and unresolved challenges remain incomplete; increasing `--maxPages` continues collection.
- Reddit saves successful image analysis, clustering, and individual post/comment summaries. Resume reuses those steps and retries failures. Twitter also checkpoints its successful summary response before writing reports. A failed Twitter summary retries using saved raw data. Inputs exceeding the default 200,000-character budget stop visibly; increase `--maxChars` to fit the selected model rather than silently dropping tweets.
- Reddit images are downloaded with a 50 MB bound and resized to JPEG for vision; animated images use their first frame. Deleted images (HTTP 404/410) are explicitly marked unavailable without blocking text-based analysis. Other download/provider failures remain retryable.
- A completed run is a no-op on resume. Add `--summarize` to process a scrape-only run. Changing source inputs requires a new run.
- Checkpoints are written by atomic replacement. Corrupt state fails visibly. A process lock prevents simultaneous execution of the same run; a dead process lock is recovered on resume.
- A provider request interrupted before its response is checkpointed may be billed again on retry. Local checkpoints cannot guarantee exactly-once execution at an external provider.

If a Nitter cursor expires, start a new run with the original `--start` and `--end`. Keep the incomplete run as evidence; do not edit its checkpoint to claim completion.

## Reprocess existing files

```sh
pnpm scrape run reddit-local --json /absolute/path/reddit_posts.json
pnpm scrape run reddit-all --json /absolute/path/reddit_posts.json --clusters 4
pnpm scrape run twitter --json /absolute/path/list-data.json --focusTopic 'AI agents'
```

Inputs are copied into the new run. Twitter rejects raw data marked partial. Use a new run to change analysis options. Reddit model configuration comes from `.env`; its AI cache includes the model and base URL, so changing either starts fresh AI steps.

## Old command mapping

| Previous command | New command from repository root |
| --- | --- |
| Reddit `parallel` | `pnpm scrape run daily` (also includes Twitter) |
| Reddit `scrapeL` | `pnpm scrape run reddit-local --scrape-only` |
| Reddit `scrapeAll` | `pnpm scrape run reddit-all --scrape-only` |
| Reddit `scrapeAndClusterL` / `scrapeAndClusterAll` | `pnpm scrape run reddit-local` / `reddit-all` |
| Reddit `clusterL` / `clusterAll` | `pnpm scrape resume RUN_ID --summarize` |
| Twitter `scrape` | `pnpm scrape run twitter --target LIST_OR_HANDLE --scrape-only` |
| Twitter `summarize:list` | `pnpm scrape run twitter --target LIST_OR_HANDLE` |
| Twitter `summarize:list --json FILE` | `pnpm scrape run twitter --json FILE` |

One-off debug scripts, obsolete shell orchestration, hard-coded October 2025 paths, and nonexistent README commands were not carried forward. The original local outputs have been copied to gitignored `archive/reddit/{specials,checkpoints}` and `archive/twitter/data`. These historical formats are retained as evidence; use `--json` to reprocess raw files, rather than treating old checkpoints as new run manifests. The old repositories remain available until live collection and AI outputs have been accepted; they must not be deleted before local archives and credentials are accounted for.

## Validation

```sh
pnpm scrape:check
pnpm scrape:test
```

Tests use local fixtures and simulated failures; they do not make paid model calls or prove current Reddit/Nitter availability.
