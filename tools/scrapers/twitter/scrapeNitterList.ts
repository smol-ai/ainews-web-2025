import fs from 'fs';
import { readJson, writeJson } from '../shared/state.js';
import path from 'path';
import axios from 'axios';
import * as cheerio from 'cheerio';
import { MediaAttachment, MediaType, Quote, ScrapeResult, Stats, Tweet, UserRef, Diagnostics } from './types.js';
import { isChallengePage, hasBrowserSession, fetchWithBrowser, closeBrowser } from './challengeSolver.js';

const BASE_URL_DEFAULT = 'https://nitter.miningtcup.me';

export interface ScrapeOptions {
  checkpointPath?: string;
  start?: Date | string;
  end?: Date | string;
  baseUrl?: string;
  outputDir?: string;
  maxPages?: number;
  maxChars?: number; // stop scraping once estimated formatted output exceeds this
  mockHtmlDir?: string; // if provided, read page1.html, page2.html, ... from this dir instead of fetching
  debug?: boolean;
}

export interface TargetSpec {
  pageType: string; // e.g., 'list', 'profile', future: 'search', 'hashtag'
  identifier: string; // list id, handle, etc.
}

interface PageCheckpoint {
  version: 1;
  config: { target: TargetSpec; start: string; end: string; baseUrl: string; mockHtmlDir?: string };
  cursor?: string;
  pageIndex: number;
  pagesFetched: number;
  bytesFetched: number;
  estimatedChars: number;
  tweets: Tweet[];
  complete: boolean;
}

function toISO(d: Date | string | undefined): string {
  if (!d) return new Date().toISOString();
  if (d instanceof Date) return d.toISOString();
  const dt = new Date(d);
  return isNaN(dt.getTime()) ? new Date().toISOString() : dt.toISOString();
}

function parseDateFromTitle(title?: string): Date | null {
  if (!title) return null;
  const cleaned = title.replace(/\s*[·•]\s*/g, ' ');
  const dt = new Date(cleaned);
  if (!isNaN(dt.getTime())) return dt;
  const dt2 = new Date(cleaned.replace(/,/g, ''));
  return isNaN(dt2.getTime()) ? null : dt2;
}

function absUrl(baseUrl: string, u?: string): string | undefined {
  if (!u) return undefined;
  if (u.startsWith('http://') || u.startsWith('https://') || u.startsWith('data:')) return u;
  return baseUrl.replace(/\/$/, '') + (u.startsWith('/') ? u : `/${u}`);
}

function textOrEmpty($el: cheerio.Cheerio<any>): string {
  return ($el.text() || '').trim();
}

function parseStats($: cheerio.CheerioAPI, $tweet: cheerio.Cheerio<any>): Stats {
  const result: Stats = { replies: 0, retweets: 0, quotes: 0, likes: 0 };
  $tweet.find('.tweet-stats .tweet-stat').each((_: number, stat: any) => {
    const $stat = $(stat);
    const icon = $stat.find('.icon-container > span').attr('class') || '';
    const numStr = ($stat.text() || '').replace(/[^0-9]/g, '');
    const n = numStr ? parseInt(numStr, 10) : 0;
    if (icon.includes('icon-comment')) result.replies = n;
    else if (icon.includes('icon-retweet')) result.retweets = n;
    else if (icon.includes('icon-quote')) result.quotes = n;
    else if (icon.includes('icon-heart')) result.likes = n;
    else if (icon.includes('icon-play')) result.plays = n;
  });
  return result;
}

function parseAttachments($: cheerio.CheerioAPI, baseUrl: string, $scope: cheerio.Cheerio<any>): MediaAttachment[] {
  const attachments: MediaAttachment[] = [];
  $scope.find('.attachments .attachment.image').each((_: number, el: any) => {
    const $el = $(el);
    const imgSrc = $el.find('img').attr('src');
    const fullHref = $el.find('a.still-image').attr('href');
    const alt = $el.find('img').attr('alt');
    attachments.push({
      type: 'image',
      previewUrl: absUrl(baseUrl, imgSrc)!,
      fullUrl: absUrl(baseUrl, fullHref),
      alt,
    });
  });
  $scope.find('.attachments .gallery-video, .attachments.card .gallery-video').each((_: number, el: any) => {
    const $el = $(el);
    const poster = $el.find('img').attr('src');
    attachments.push({
      type: 'video',
      previewUrl: absUrl(baseUrl, poster) || '',
    });
  });
  $scope.find('.card, .attachments.card').each((_: number, card: any) => {
    const href = $(card).find('a.card-container').attr('href');
    if (href) attachments.push({ type: 'card', previewUrl: absUrl(baseUrl, href) || href });
  });
  return attachments;
}

function parseUser($: cheerio.CheerioAPI, baseUrl: string, $tweet: cheerio.Cheerio<any>): UserRef {
  const $username = $tweet.find('.tweet-header a.username').first();
  const $fullname = $tweet.find('.tweet-header a.fullname').first();
  const $avatar = $tweet.find('.tweet-header a.tweet-avatar img.avatar').first();
  const username = ($username.attr('title') || $username.text() || '').trim();
  const displayName = ($fullname.attr('title') || $fullname.text() || '').trim();
  const profileUrl = absUrl(baseUrl, $username.attr('href'));
  const avatarUrl = absUrl(baseUrl, $avatar.attr('src'));
  return { username, displayName, profileUrl, avatarUrl };
}

function parseQuote($: cheerio.CheerioAPI, baseUrl: string, $tweet: cheerio.Cheerio<any>): Quote | undefined {
  const $quote = $tweet.find('.quote').first();
  if ($quote.length === 0) return undefined;
  const link = $quote.find('a.quote-link').attr('href');
  const url = absUrl(baseUrl, link);
  let tweetId: string | undefined;
  if (link) {
    const m = link.match(/status\/(\d+)/);
    if (m) tweetId = m[1];
  }
  const user: UserRef | undefined = (() => {
    const $uname = $quote.find('.tweet-name-row a.username').first();
    const $fname = $quote.find('.tweet-name-row a.fullname').first();
    if ($uname.length === 0 && $fname.length === 0) return undefined;
    const username = ($uname.attr('title') || $uname.text() || '').trim();
    const displayName = ($fname.attr('title') || $fname.text() || '').trim();
    const profileUrl = absUrl(baseUrl, $uname.attr('href'));
    const avatarUrl = absUrl(baseUrl, $quote.find('.tweet-name-row img.avatar').attr('src'));
    return { username, displayName, profileUrl, avatarUrl };
  })();
  const text = textOrEmpty($quote.find('.quote-text'));
  const media = parseAttachments($, baseUrl, $quote);
  const timestamp = (() => {
    const title = $quote.find('.tweet-date a').attr('title');
    const dt = parseDateFromTitle(title);
    return dt ? dt.toISOString() : undefined;
  })();
  return { tweetId, url, user, text, media, timestamp };
}

function parseOneTweet($: cheerio.CheerioAPI, baseUrl: string, $item: cheerio.Cheerio<any>): Tweet | null {
  const $link = $item.find('a.tweet-link').first();
  const href = $link.attr('href');
  if (!href) return null;
  const url = absUrl(baseUrl, href)!;
  const idMatch = href.match(/status\/(\d+)/);
  const id = idMatch ? idMatch[1] : href;
  const user = parseUser($, baseUrl, $item);
  const title = $item.find('.tweet-date a').attr('title');
  const dt = parseDateFromTitle(title);
  if (!dt) return null;
  const text = textOrEmpty($item.find('.tweet-content.media-body'));
  const attachments = parseAttachments($, baseUrl, $item);
  const stats = parseStats($, $item);
  const retweetedBy = textOrEmpty($item.find('.retweet-header')) || undefined;
  const quoted = parseQuote($, baseUrl, $item);
  const isThread = $item.hasClass('thread');
  return {
    id,
    url,
    user,
    timestamp: dt.toISOString(),
    timestampMs: dt.getTime(),
    text,
    attachments,
    stats,
    retweetedBy,
    quoted,
    isThread,
  };
}

function buildPathForTarget(target: TargetSpec): string {
  switch (target.pageType) {
    case 'list':
      return `/i/lists/${encodeURIComponent(target.identifier)}`;
    case 'profile':
      return `/${encodeURIComponent(target.identifier.replace(/^@/, ''))}`;
    default:
      // Default to profile-like path for future types unless overridden
      return `/${encodeURIComponent(target.identifier.replace(/^@/, ''))}`;
  }
}

const FETCH_HEADERS: Record<string, string> = {
  'User-Agent': 'Mozilla/5.0 (compatible; newSmolTalkScraper/1.0) Node.js',
  'Accept-Language': 'en-US,en;q=0.9',
  'Accept': 'text/html,application/xhtml+xml',
  'Accept-Encoding': 'gzip, compress, deflate, br',
};

// Once a challenge is detected, all subsequent fetches use the browser
let useBrowserMode = false;

async function fetchPage(baseUrl: string, target: TargetSpec, cursor: string | undefined, opts: { mockHtmlDir?: string; pageIndex?: number }): Promise<{ html: string; url: string; }> {
  if (opts.mockHtmlDir) {
    const idx = opts.pageIndex ?? 1;
    const byIndex = path.resolve(opts.mockHtmlDir, `page${idx}.html`);
    if (fs.existsSync(byIndex)) {
      const html = fs.readFileSync(byIndex, 'utf8');
      return { html, url: byIndex };
    }
    if (idx === 1) {
      const profileFallback = path.resolve(opts.mockHtmlDir, 'profile.html');
      if (fs.existsSync(profileFallback)) {
        const html = fs.readFileSync(profileFallback, 'utf8');
        return { html, url: profileFallback };
      }
    }
    throw new Error(`Mock HTML page ${idx} not found in ${opts.mockHtmlDir}`);
  }

  const pathPart = buildPathForTarget(target);
  const url = `${baseUrl.replace(/\/$/, '')}${pathPart}${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`;
  const debug = process.env.SCRAPE_DEBUG === '1';

  // If we've already switched to browser mode, use it directly
  if (useBrowserMode) {
    if (debug) console.log(`[scrape] Fetching via browser: ${url}`);
    const html = await fetchWithBrowser(url, debug);
    return { html, url };
  }

  const maxRetries = 10;
  const baseDelayMs = 2000;
  const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      if (debug && attempt === 0) console.log(`[scrape] Attempting to fetch URL: ${url}`);

      const res = await axios.get(url, {
        headers: FETCH_HEADERS,
        timeout: 30000,
        responseType: 'text',
        validateStatus: (s) => (s >= 200 && s < 400) || s === 429 || s === 502 || s === 503,
      });

      const html = res.data as string;

      // Detect anti-bot challenge — switch to browser mode permanently
      if (isChallengePage(html)) {
        if (debug) console.log(`[scrape] Anti-bot challenge detected (status ${res.status}), switching to browser mode...`);
        useBrowserMode = true;
        const browserHtml = await fetchWithBrowser(url, debug);
        return { html: browserHtml, url };
      }

      if (res.status === 200) {
        return { html, url };
      }

      if (res.status === 429 || res.status === 502 || res.status === 503) {
        if (attempt >= maxRetries) {
          console.log(` ${res.status} — giving up after ${attempt + 1} attempts`);
          throw new Error(`HTTP ${res.status} after ${attempt + 1} attempts for ${url}`);
        }

        const ra = (res.headers?.['retry-after'] ?? res.headers?.['Retry-After']) as string | undefined;
        if (ra && /^\d+$/.test(ra)) {
          const headerSec = parseInt(ra, 10);
          const waitMs = Math.min((headerSec + 2) * 1000, 5 * 60 * 1000);
          console.log(` ${res.status} — Retry-After: ${headerSec}s, waiting ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
          await sleep(waitMs);
          continue;
        }
        const reset = (res.headers?.['x-rate-limit-reset'] as string | undefined) || (res.headers?.['X-Rate-Limit-Reset'] as string | undefined);
        if (reset && /^\d+$/.test(reset)) {
          const resetMs = parseInt(reset, 10) * 1000;
          const waitMs = Math.min(Math.max(resetMs - Date.now(), 1000), 5 * 60 * 1000);
          console.log(` ${res.status} — rate reset in ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
          await sleep(waitMs);
          continue;
        }
        const jitter = Math.floor(Math.random() * 250);
        const waitMs = Math.min(baseDelayMs * Math.pow(2, attempt) + jitter, 60 * 1000);
        console.log(` ${res.status} — no Retry-After header, backoff ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
        await sleep(waitMs);
        continue;
      }

      throw new Error(`Unexpected HTTP ${res.status}`);
    } catch (err: any) {
      if (err.message?.startsWith('Rate limited') || err.message?.startsWith('HTTP ') || err.message?.startsWith('Unexpected HTTP') || err.message?.startsWith('Network error') || err.message?.startsWith('Failed to fetch')) {
        throw err;
      }

      const status = err?.response?.status;
      if (status === 429 || status === 503) {
        const errBody = err?.response?.data as string | undefined;
        if (errBody && isChallengePage(errBody)) {
          console.log(` anti-bot challenge detected, switching to browser mode`);
          useBrowserMode = true;
          const browserHtml = await fetchWithBrowser(url, debug);
          return { html: browserHtml, url };
        }

        if (attempt >= maxRetries) {
          console.log(` rate limited (${status}), giving up after ${attempt + 1} attempts`);
          throw new Error(`Rate limited (${status}) after ${attempt + 1} attempts for ${url}: ${err.message}`);
        }

        const ra = (err.response?.headers?.['retry-after'] ?? err.response?.headers?.['Retry-After']) as string | undefined;
        if (ra && /^\d+$/.test(ra)) {
          const headerSec = parseInt(ra, 10);
          const waitMs = Math.min((headerSec + 2) * 1000, 5 * 60 * 1000);
          console.log(` ${status} — Retry-After: ${headerSec}s, waiting ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
          await sleep(waitMs);
          continue;
        }
        const reset = (err.response?.headers?.['x-rate-limit-reset'] as string | undefined) || (err.response?.headers?.['X-Rate-Limit-Reset'] as string | undefined);
        if (reset && /^\d+$/.test(reset)) {
          const resetMs = parseInt(reset, 10) * 1000;
          const waitMs = Math.min(Math.max(resetMs - Date.now(), 1000), 5 * 60 * 1000);
          console.log(` ${status} — rate reset in ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
          await sleep(waitMs);
          continue;
        }
        const jitter = Math.floor(Math.random() * 250);
        const waitMs = Math.min(baseDelayMs * Math.pow(2, attempt) + jitter, 60 * 1000);
        console.log(` ${status} — no Retry-After header, backoff ${Math.ceil(waitMs / 1000)}s (retry ${attempt + 1}/${maxRetries})`);
        await sleep(waitMs);
        continue;
      }

      // Definitive HTTP 4xx (e.g. 404 list not found, 403 blocked) — retrying won't help; fail fast so the waterfall can fall back.
      if (status && status >= 400 && status < 500) {
        const errBody = err?.response?.data as string | undefined;
        if (errBody && isChallengePage(errBody)) {
          console.log(` anti-bot challenge detected (${status}), switching to browser mode`);
          useBrowserMode = true;
          const browserHtml = await fetchWithBrowser(url, debug);
          return { html: browserHtml, url };
        }
        console.log(` HTTP ${status} — not retryable, giving up`);
        throw new Error(`HTTP ${status} for ${url}`);
      }

      if (attempt >= maxRetries) {
        console.log(` network error, giving up after ${attempt + 1} attempts`);
        throw new Error(`Network error after ${attempt + 1} attempts for ${url}: ${err.message}`);
      }
      const jitter = Math.floor(Math.random() * 250);
      const waitMs = Math.min(baseDelayMs * Math.pow(2, attempt) + jitter, 60 * 1000);
      console.log(` network error (${err.code || err.message}) — retry ${attempt + 1}/${maxRetries}, backoff ${Math.ceil(waitMs / 1000)}s`);

      if (debug) {
        console.log(`[scrape] error details: code=${err.code} message=${err.message}`);
        if (err.cause) console.log(`[scrape] error cause: ${err.cause}`);
        if (err.syscall) console.log(`[scrape] syscall=${err.syscall} hostname=${err.hostname}`);
        if (err.errno) console.log(`[scrape] errno=${err.errno}`);
      }

      await sleep(waitMs);
    }
  }

  const errorMsg = `Failed to fetch ${url} after ${maxRetries + 1} attempts`;
  if (debug) console.log(`[scrape] ${errorMsg}`);
  throw new Error(errorMsg);
}

function findNextCursor($: cheerio.CheerioAPI): string | undefined {
  const direct = $('div.show-more a').last().attr('href') || '';
  const fromDirect = direct.match(/[?&]cursor=([^&]+)/);
  if (fromDirect) return decodeURIComponent(fromDirect[1]);
  let candidate: string | undefined;
  $('a[href*="?cursor="]').each((_: number, el: any) => {
    const h = $(el).attr('href') || '';
    const m = h.match(/[?&]cursor=([^&]+)/);
    if (m) candidate = decodeURIComponent(m[1]);
  });
  return candidate;
}

export async function scrapeTarget(target: TargetSpec, options: ScrapeOptions = {}): Promise<ScrapeResult> {
  try {
    return await scrapeTargetOnce(target, options);
  } finally {
    if (hasBrowserSession()) await closeBrowser();
    useBrowserMode = false;
  }
}

async function scrapeTargetOnce(target: TargetSpec, options: ScrapeOptions = {}): Promise<ScrapeResult> {
  useBrowserMode = false;
  const baseUrl = options.baseUrl || BASE_URL_DEFAULT;
  const sourceUrl = new URL(baseUrl);
  if (!['http:', 'https:'].includes(sourceUrl.protocol) || sourceUrl.username || sourceUrl.password || sourceUrl.search || sourceUrl.hash) {
    throw new Error('baseUrl must be HTTP(S) without credentials, query parameters, or fragments');
  }
  const checkpointExists = Boolean(options.checkpointPath && fs.existsSync(options.checkpointPath));
  const previous = checkpointExists
    ? await readJson<PageCheckpoint>(options.checkpointPath!) : undefined;
  if (checkpointExists && (!previous || previous.version !== 1 || !previous.config || !Array.isArray(previous.tweets)
    || !Number.isInteger(previous.pageIndex) || previous.pageIndex < 1
    || !Number.isInteger(previous.pagesFetched) || previous.pagesFetched < 0
    || !Number.isFinite(previous.bytesFetched) || previous.bytesFetched < 0
    || !Number.isFinite(previous.estimatedChars) || previous.estimatedChars < 0
    || typeof previous.complete !== 'boolean')) {
    throw new Error('Invalid Twitter page checkpoint');
  }
  const endDate = new Date(options.end ?? previous?.config.end ?? Date.now());
  const startDate = new Date(options.start ?? previous?.config.start ?? endDate.getTime() - 24 * 60 * 60 * 1000);
  if (isNaN(endDate.getTime())) throw new Error('Invalid end datetime');
  if (isNaN(startDate.getTime())) throw new Error('Invalid start datetime');
  if (startDate >= endDate) throw new Error('Start must be before end');
  const debug = options.debug || process.env.SCRAPE_DEBUG === '1';

  const outputDir = options.outputDir || path.resolve(process.cwd(), 'data');
  const maxPages = options.maxPages ?? 100;
  const charBudget = options.maxChars ?? 0; // 0 = no limit
  const mockHtmlDir = options.mockHtmlDir;
  if (!Number.isInteger(maxPages) || maxPages <= 0) throw new Error('maxPages must be a positive integer');
  if (!Number.isFinite(charBudget) || charBudget < 0) throw new Error('maxChars must be nonnegative');
  const config: PageCheckpoint['config'] = {
    target: { pageType: target.pageType, identifier: target.identifier },
    start: startDate.toISOString(), end: endDate.toISOString(), baseUrl,
    ...(mockHtmlDir ? { mockHtmlDir: path.resolve(mockHtmlDir) } : {}),
  };
  if (previous && JSON.stringify(previous.config) !== JSON.stringify(config)) {
    throw new Error('Twitter checkpoint target/window/source does not match this run; use the original configuration or a new checkpoint');
  }

  const executionStart = new Date();
  let pagesFetched = previous?.pagesFetched ?? 0;
  let bytesFetched = previous?.bytesFetched ?? 0;
  let estimatedChars = previous?.estimatedChars ?? 0;
  let cursor: string | undefined = previous?.cursor;
  let pageIndex = previous?.pageIndex ?? 1;
  const collected: Tweet[] = previous?.tweets ?? [];
  const seen = new Set(collected.map((tweet) => tweet.id));
  let complete = previous?.complete ?? false;
  const saveCheckpoint = async () => {
    if (!options.checkpointPath) return;
    await writeJson(options.checkpointPath, {
      version: 1, config, cursor, pageIndex, pagesFetched, bytesFetched, estimatedChars,
      tweets: collected, complete,
    } satisfies PageCheckpoint);
  };
  await saveCheckpoint();
  let sawOlderThanStart = false; // page-level decision only
  let hitCharBudget = false;
  let wasPartial = false;
  let lastErrorMessage: string | undefined;

  if (debug) {
    console.log(`[scrape] target=${target.pageType}:${target.identifier} start=${startDate.toISOString()} end=${endDate.toISOString()} baseUrl=${baseUrl}`);
  }

  const dim = '\x1b[2m';
  const reset = '\x1b[0m';
  const green = '\x1b[32m';
  const yellow = '\x1b[33m';

  while (!complete && pagesFetched < maxPages && (!charBudget || estimatedChars < charBudget)) {
    let html: string;
    let url: string;

    const pageStart = Date.now();
    const elapsedSec = ((Date.now() - executionStart.getTime()) / 1000).toFixed(1);
    process.stdout.write(`${dim}[scrape] Fetching page ${pageIndex}...${reset}`);

    try {
      const result = await fetchPage(baseUrl, target, cursor, { mockHtmlDir, pageIndex });
      html = result.html;
      url = result.url;
    } catch (err: any) {
      wasPartial = true;
      lastErrorMessage = err?.message || String(err);
      const fetchMs = Date.now() - pageStart;
      console.log(` ${yellow}FAILED${reset} (${fetchMs}ms) — ${lastErrorMessage}`);
      console.log(`${yellow}[scrape] Returning partial results: ${collected.length} tweets from ${pagesFetched} pages${reset}`);
      break;
    }

    const fetchMs = Date.now() - pageStart;
    const $ = cheerio.load(html);
    const $items = $('.timeline .timeline-item').filter((_, el) => $(el).find('.tweet-body').length > 0);

    if (isChallengePage(html) || $('.error-panel').length || !$('.timeline').length) {
      wasPartial = true;
      lastErrorMessage = 'Instance returned a challenge, error, or missing timeline; retry this page after recovery';
      break;
    }
    pagesFetched += 1;
    bytesFetched += Buffer.byteLength(html, 'utf8');

    // Surface instance outage/error notices instead of silently reporting 0 tweets
    if ($items.length === 0 && pageIndex === 1) {
      const notice = ($('.error-panel').text() || $('body').text() || '').trim().replace(/\s+/g, ' ').slice(0, 250);
      if (notice) console.log(`\n${yellow}[scrape] Instance returned no timeline. Page says: "${notice}"${reset}`);
    }

    const pageTweetsKept: Tweet[] = [];
    let pageOlderRetweets = 0;
    let pageNonRTMinMs: number | undefined;
    let pageNonRTMaxMs: number | undefined;
    let pageAllMinMs: number | undefined;
    let pageAllMaxMs: number | undefined;
    let pageNonRTWithinCount = 0;
    let unparsedItems = 0;

    $items.each((_, el) => {
      const $el = $(el);
      const t = parseOneTweet($, baseUrl, $el);
      if (!t) { unparsedItems += 1; return; }
      const tDate = new Date(t.timestamp);
      if (tDate > endDate) {
        return;
      }
      const tMs = tDate.getTime();
      pageAllMinMs = pageAllMinMs === undefined ? tMs : Math.min(pageAllMinMs, tMs);
      pageAllMaxMs = pageAllMaxMs === undefined ? tMs : Math.max(pageAllMaxMs, tMs);
      if (!t.retweetedBy) {
        pageNonRTMinMs = pageNonRTMinMs === undefined ? tMs : Math.min(pageNonRTMinMs, tMs);
        pageNonRTMaxMs = pageNonRTMaxMs === undefined ? tMs : Math.max(pageNonRTMaxMs, tMs);
      }

      if (tMs < startDate.getTime()) {
        if (t.retweetedBy) pageOlderRetweets += 1;
        return;
      }

      if (!t.retweetedBy) pageNonRTWithinCount += 1;
      if (!seen.has(t.id)) {
        seen.add(t.id);
        pageTweetsKept.push(t);
      }
    });

    if (unparsedItems) {
      wasPartial = true;
      lastErrorMessage = `Could not parse ${unparsedItems} timeline items; page checkpoint was not advanced`;
      pagesFetched -= 1;
      break;
    }
    collected.push(...pageTweetsKept);

    // Estimate formatted output size: "(cleanedText, score, url)\n\n" per tweet
    const PER_TWEET_OVERHEAD = 80; // url (~60) + score + parens + separator
    for (const t of pageTweetsKept) {
      estimatedChars += (t.text?.length || 0) + PER_TWEET_OVERHEAD;
    }

    // Always-on compact progress line
    const charInfo = charBudget ? ` | ~${(estimatedChars / 1000).toFixed(0)}k/${(charBudget / 1000).toFixed(0)}k chars` : '';
    console.log(` ${green}+${pageTweetsKept.length}${reset} tweets (${collected.length} total) ${dim}${fetchMs}ms | ${$items.length} items on page${charInfo}${reset}`);

    if (debug) {
      const keptMin = [...pageTweetsKept].sort((a, b) => a.timestampMs - b.timestampMs)[0]?.timestamp;
      const keptMax = [...pageTweetsKept].sort((a, b) => b.timestampMs - a.timestampMs)[0]?.timestamp;
      const nonRtMinIso = pageNonRTMinMs ? new Date(pageNonRTMinMs).toISOString() : undefined;
      const nonRtMaxIso = pageNonRTMaxMs ? new Date(pageNonRTMaxMs).toISOString() : undefined;
      const allMinIso = pageAllMinMs ? new Date(pageAllMinMs).toISOString() : undefined;
      const allMaxIso = pageAllMaxMs ? new Date(pageAllMaxMs).toISOString() : undefined;
      console.log(`[scrape]   ranges: nonRT=[${nonRtMinIso} .. ${nonRtMaxIso}] all=[${allMinIso} .. ${allMaxIso}] kept=[${keptMin} .. ${keptMax}]`);
    }

    const newestRelevantMs = pageNonRTMaxMs ?? pageAllMaxMs;
    if (newestRelevantMs !== undefined && newestRelevantMs < startDate.getTime()) {
      console.log(`${dim}[scrape] Reached tweets older than start date; stopping pagination${reset}`);
      sawOlderThanStart = true;
      complete = true;
      await saveCheckpoint();
      break;
    }

    const next = findNextCursor($);
    if (!next) {
      console.log(`${dim}[scrape] No more pages (no next cursor)${reset}`);
      complete = true;
      await saveCheckpoint();
      break;
    }
    if (next === cursor) {
      wasPartial = true;
      lastErrorMessage = 'Pagination returned the same cursor; coverage is incomplete';
      // Retry this page on resume rather than checkpointing a cursor loop as progress.
      pagesFetched -= 1;
      break;
    }
    if (debug) console.log(`[scrape] next cursor found`);
    cursor = next;
    pageIndex += 1;
    await saveCheckpoint();
  }

  if (!complete) {
    wasPartial = true;
    hitCharBudget = Boolean(charBudget && estimatedChars >= charBudget);
    lastErrorMessage ??= hitCharBudget
      ? 'Character budget reached before full window coverage; raise maxChars and resume'
      : 'Page limit reached before full window coverage; raise maxPages and resume';
  }

  collected.sort((a, b) => a.timestampMs - b.timestampMs);

  const earliest = collected[0]?.timestamp;
  const latest = collected[collected.length - 1]?.timestamp;

  const executionEnd = new Date();
  const scrapeDurationSec = ((executionEnd.getTime() - executionStart.getTime()) / 1000).toFixed(1);
  console.log(`[scrape] Done: ${collected.length} tweets from ${pagesFetched} pages in ${scrapeDurationSec}s`);
  const diagnostics: Diagnostics = {
    listId: `${target.pageType}:${target.identifier}`,
    baseUrl,
    pagesFetched,
    bytesFetched,
    tweetCount: collected.length,
    startParam: startDate.toISOString(),
    endParam: endDate.toISOString(),
    earliestTimestamp: earliest,
    latestTimestamp: latest,
    executionStart: executionStart.toISOString(),
    executionEnd: executionEnd.toISOString(),
    durationMs: executionEnd.getTime() - executionStart.getTime(),
    wasPartial,
    errorMessage: wasPartial ? lastErrorMessage : undefined,
    stoppedByCharBudget: hitCharBudget,
  };

  if (wasPartial && debug) {
    console.log(`[scrape] Completed with partial results due to: ${lastErrorMessage}`);
    console.log(`[scrape] Successfully collected ${collected.length} tweets from ${pagesFetched} pages`);
  }

  return { tweets: collected, diagnostics };
}

export async function scrapeList(listId: string, options: ScrapeOptions = {}): Promise<ScrapeResult> {
  return scrapeTarget({ pageType: 'list', identifier: listId }, options);
}

export async function writeResultToJson(result: ScrapeResult, listId: string, start: Date, end: Date, outputDir?: string): Promise<string> {
  const outDir = outputDir || path.resolve(process.cwd(), 'data');
  fs.mkdirSync(outDir, { recursive: true });
  const sanitize = (s: string) => s.replace(/[:]/g, '').replace(/\./g, '').replace(/Z$/, 'Z');
  const fileName = `list-${listId}-${sanitize(start.toISOString())}-${sanitize(end.toISOString())}.json`;
  const target = path.join(outDir, fileName);
  const payload = {
    tweets: result.tweets,
    diagnostics: result.diagnostics,
  };
  await writeJson(target, payload);
  return target;
}
