/**
 * API functions for interacting with the Reddit API
 * Handles fetching posts and comments with rate limiting and retries
 */
import axios, { AxiosError } from 'axios';
import { RedditComment, RedditPost, SubredditData } from './types';
import { RateLimiter } from './rateLimiter';
import { processCommentData } from './utils';
import chalk from 'chalk';

// Create a global rate limiter instance
const rateLimiter = new RateLimiter(10, 20);

const USER_AGENT = 'typescript:reddit-scraper:v1.0.0 (by /u/your_username)';

// Warn at most once if the session cookie is missing.
let warnedMissingSession = false;

/**
 * Builds request headers for Reddit. As of 2026 Reddit blocks anonymous
 * `.json` access with a 403 "Blocked" page; the only thing needed to get
 * past it is a logged-in `reddit_session` cookie (a JWT). We read it from
 * the REDDIT_SESSION env var so the secret stays out of source control.
 *
 * Minimal recipe (verified): User-Agent + `Cookie: reddit_session=<jwt>`.
 * No other cookies or browser headers are required.
 */
function buildRedditHeaders(): Record<string, string> {
  const headers: Record<string, string> = { 'User-Agent': USER_AGENT };
  const session = process.env.REDDIT_SESSION?.trim();
  if (session) {
    headers['Cookie'] = `reddit_session=${session}`;
  } else if (!warnedMissingSession) {
    warnedMissingSession = true;
    console.warn(
      chalk.yellow(
        '[WARN] REDDIT_SESSION env var is not set. Anonymous Reddit requests will be blocked (403). ' +
          'Copy the `reddit_session` cookie from a logged-in browser into .env.'
      )
    );
  }
  return headers;
}

/** Detects Reddit's anti-bot 403 "Blocked" response (usually = expired/missing session). */
function isRedditBlock(err: unknown): boolean {
  return axios.isAxiosError(err) && err.response?.status === 403;
}

/** Max number of below-cutoff posts to keep when no posts pass `minimumScore`.
 * 1 = keep just the most active sub-cutoff post (per user request: option (c)). */
const MAX_FALLBACK_POSTS = 1;

function reddit429WaitMs(headers: Record<string, unknown> | undefined): number {
  if (!headers) return 0;
  const retryAfterRaw = headers['retry-after'];
  if (typeof retryAfterRaw === 'string') {
    const sec = parseInt(retryAfterRaw, 10);
    if (!Number.isNaN(sec)) return Math.max(sec * 1000, 500);
  }
  const resetRaw = headers['x-ratelimit-reset'];
  const resetNum =
    typeof resetRaw === 'string' ? parseFloat(resetRaw) : typeof resetRaw === 'number' ? resetRaw : NaN;
  if (!Number.isNaN(resetNum)) {
    return Math.max(Math.ceil(resetNum * 1000) + 2000, 500);
  }
  return 0;
}

const sleepMs = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));

function logAxiosCatch(context: string, err: unknown, timestamp: string): void {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError;
    const brief = ax.response
      ? { status: ax.response.status, statusText: ax.response.statusText }
      : { code: ax.code };
    console.error(chalk.red(`[${timestamp}] [ERROR] ${context}: ${ax.message}`), brief);
  } else {
    console.error(chalk.red(`[${timestamp}] [ERROR] ${context}:`), err);
  }
}

/**
 * Fetches top comments for a given post permalink
 * Handles rate limiting, retries, and error handling
 *
 * @param postPermalink - The permalink to the Reddit post
 * @returns Array of top comments sorted by score
 */
export const fetchTopCommentsForPost = async (postPermalink: string, customRateLimiter?: RateLimiter): Promise<RedditComment[]> => {
  let retryCount = 0;
  const maxRetries = 3;
  const timestamp = new Date().toISOString();

  console.log(chalk.blue(`[${timestamp}] [COMMENTS] Starting fetch for ${postPermalink}`));

  const limiter = customRateLimiter || rateLimiter;

  while (retryCount <= maxRetries) {
    try {
      // Wait for rate limiter before making the request
      await limiter.throttle();

      console.log(chalk.cyan(`[${timestamp}] [COMMENTS] Fetching comments for post: ${postPermalink}`));

      // Add .json to get the JSON data
      const response = await axios.get(
        `${postPermalink}.json`,
        {
          headers: buildRedditHeaders(),
          timeout: 30000
        }
      );

      // Record successful request
      limiter.recordSuccess();

      // Reddit returns an array with 2 objects:
      // - response[0] contains the post data
      // - response[1] contains the comments data
      if (response.data && Array.isArray(response.data) && response.data.length >= 2) {
        const commentsListing = response.data[1];

        if (commentsListing.data && Array.isArray(commentsListing.data.children)) {
          console.log(chalk.green(`[${timestamp}] [COMMENTS] Found ${commentsListing.data.children.length} top-level comments`));

          // Process all top-level comments
          const comments: RedditComment[] = [];

          commentsListing.data.children.forEach((commentData: any) => {
            // Skip "more" type comments
            if (commentData.kind === 'more') {
              console.log(chalk.yellow(`[${timestamp}] [COMMENTS] Skipping 'more' comment with ${commentData.data.count} additional comments`));
              return;
            }

            const processedComment = processCommentData(commentData);
            if (processedComment) {
              comments.push(processedComment);
            }
          });

          // Sort by score (highest first) and take top 5
          comments.sort((a, b) => b.score - a.score);
          const topComments = comments.slice(0, 5);

          // console.log(chalk.green(`[${timestamp}] [COMMENTS] Returning ${topComments.length} top comments`));

          // // // Debug: Check if any comments have replies
          // // const commentsWithReplies = topComments.filter(c => c.replies && c.replies.length > 0);
          // // // if (commentsWithReplies.length > 0) {
          // // //   console.log(chalk.green(`[${timestamp}] [COMMENTS] ${commentsWithReplies.length} comments have replies`));
          // // //   commentsWithReplies.forEach(c => {
          // // //     console.log(chalk.green(`[${timestamp}] [COMMENTS] Comment by ${c.author} has ${c.replies?.length} replies`));
          // // //   });
          // // // } else {
          // // //   console.log(chalk.yellow(`[${timestamp}] [COMMENTS] No comments have replies`));
          // // // }

          return topComments;
        }
      }

      throw new Error('Reddit returned an invalid comments listing.');
    } catch (error) {
      logAxiosCatch(`Comments fetch failed for ${postPermalink}`, error, timestamp);
      if (isRedditBlock(error)) throw new Error('Reddit blocked comments; refresh REDDIT_SESSION and resume.');

      if (axios.isAxiosError(error) && error.response?.status === 429) {
        const w429 = reddit429WaitMs(error.response.headers as Record<string, unknown>);
        if (w429 > 0) {
          console.log(
            chalk.yellow(
              `[${timestamp}] [429] Waiting ${Math.round(w429 / 1000)}s per Reddit headers before retry backoff...`
            )
          );
          await sleepMs(w429);
        }
      }

      if (limiter.recordFailure() && retryCount < maxRetries) {
        retryCount++;
        console.log(chalk.yellow(`[${timestamp}] [RETRY] Retrying (${retryCount}/${maxRetries})...`));
        await sleepMs(limiter.backoffTime);
      } else {
        console.error(
          chalk.red(`[${timestamp}] [ERROR] Max retries reached or too many failures. Giving up on fetching comments for ${postPermalink}`)
        );
        throw new Error(`Comments fetch failed for ${postPermalink}; resume to retry this subreddit.`);
      }
    }
  }

  return [];
};

/**
 * Fetches top posts from a specified subreddit
 * Handles rate limiting, retries, and error handling
 *
 * @param subreddit - The subreddit name to fetch posts from
 * @param minimumScore - Minimum score threshold for posts to include
 * @param customRateLimiter - Optional custom rate limiter to use
 * @returns SubredditData containing posts and their comments
 */
export const fetchTopPostsFromSubreddit = async (
  subreddit: string,
  minimumScore: number = 100,
  customRateLimiter?: RateLimiter
): Promise<SubredditData> => {
  // Trim leading '/r/' if present
  const subredditName = subreddit.replace(/^\/r\//, '');
  const timestamp = new Date().toISOString();

  // Validate subreddit name format
  if (!subredditName || subredditName.trim() === '') {
    console.error(chalk.red(`[${timestamp}] [ERROR] Empty subreddit name provided`));
    return {
      subreddit: subredditName,
      posts: [],
      filterStats: { totalPostsViewed: 0, postsAfterFilter: 0, minimumScore }
    };
  }

  // Warn about suspicious subreddit names (numeric-only, very short, etc.)
  if (/^\d+$/.test(subredditName)) {
    console.error(chalk.red(`[${timestamp}] [ERROR] Invalid subreddit name: "${subredditName}" (numeric-only). This is likely a parsing error from command-line arguments!`));
    return {
      subreddit: subredditName,
      posts: [],
      filterStats: { totalPostsViewed: 0, postsAfterFilter: 0, minimumScore }
    };
  }

  if (subredditName.length < 3) {
    console.warn(chalk.yellow(`[${timestamp}] [WARN] Suspicious subreddit name: "${subredditName}" (too short). Proceeding anyway...`));
  }

  // Use provided rate limiter or the default one
  const limiter = customRateLimiter || rateLimiter;

  console.log(chalk.blue(`[${timestamp}] [POSTS] Fetching top posts from r/${subredditName}...`));

  let retryCount = 0;
  const maxRetries = 3;

  while (retryCount <= maxRetries) {
    try {
      let fallbackBelowMinimumCount: number | undefined = undefined;

      // Wait for rate limiter before making the request
      await limiter.throttle();

      // Reddit's API allows up to 100 posts per request
      // Using 100 to maximize the pool before filtering by minimum score
      const REDDIT_API_LIMIT = 100;
      console.log(chalk.cyan(`[${timestamp}] [DEBUG] Requesting hot posts from Reddit API: limit=${REDDIT_API_LIMIT}, min_score=${minimumScore}`));

      // Using 'hot' instead of 'top?t=day' because Reddit's top/day API has been bugged
      // since ~March 2025 and doesn't return actual top posts from the day.
      // NOTE: 'hot' has a different issue - it can return very old posts (1+ week old).
      // We apply a post age filter below to only include posts from the last 2 days.
      const response = await axios.get(
        `https://www.reddit.com/r/${subredditName}/hot.json`,
        {
          params: { limit: REDDIT_API_LIMIT },
          headers: buildRedditHeaders(),
          timeout: 30000
        }
      );

      if (!Array.isArray(response.data?.data?.children)) throw new Error('Reddit returned an invalid posts listing.');
      // Record successful request
      limiter.recordSuccess();

      // Log raw response info for debugging - including HTTP status and response structure
      console.log(chalk.cyan(`[${timestamp}] [DEBUG] Reddit API response status: ${response.status}`));
      console.log(chalk.cyan(`[${timestamp}] [DEBUG] Response has data.data: ${!!response.data?.data}, has children: ${!!response.data?.data?.children}`));

      const rawPostCount = response.data?.data?.children?.length || 0;
      if (rawPostCount === 0) {
        // Enhanced logging for empty responses - might indicate rate limiting
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ⚠️  REDDIT RETURNED 0 POSTS FOR r/${subredditName}`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF] No posts available to analyze - subreddit may be:`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF]   • Inactive or low-traffic today`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF]   • Private or quarantined`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF]   • Non-existent (check spelling)`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF]   • Rate limited by Reddit API`));
        console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));
        console.log(chalk.red(`[${timestamp}] [DEBUG] Full response structure: ${JSON.stringify(Object.keys(response.data || {}))}`));
        console.log(chalk.red(`[${timestamp}] [DEBUG] Response.data.data keys: ${JSON.stringify(Object.keys(response.data?.data || {}))}`));
      } else {
        console.log(chalk.cyan(`[${timestamp}] [INFO] Reddit returned ${rawPostCount} posts for r/${subredditName} before score filtering (requested limit: ${REDDIT_API_LIMIT})`));
      }

      // Extract posts from response
      let posts: RedditPost[] = response.data.data.children.map((child: any) => {
        const post = child.data;
        return {
          title: post.title,
          author: post.author,
          url: post.url,
          permalink: `https://www.reddit.com${post.permalink}`,
          score: post.score,
          num_comments: post.num_comments,
          created_utc: post.created_utc,
          selftext: post.selftext
        };
      });

      // Filter out posts older than 2 days (hot endpoint can return week-old posts)
      const MAX_POST_AGE_DAYS = 2;
      const maxAgeSeconds = MAX_POST_AGE_DAYS * 24 * 60 * 60;
      const nowSeconds = Date.now() / 1000;
      const postsBeforeAgeFilter = posts.length;
      posts = posts.filter(post => {
        const postAgeSeconds = nowSeconds - post.created_utc;
        return postAgeSeconds <= maxAgeSeconds;
      });
      const postsFilteredByAge = postsBeforeAgeFilter - posts.length;
      if (postsFilteredByAge > 0) {
        console.log(chalk.yellow(`[${timestamp}] [AGE-FILTER] Filtered out ${postsFilteredByAge} posts older than ${MAX_POST_AGE_DAYS} days from r/${subredditName}`));
      }

      // Filter posts by minimum score - but keep track of filtered out posts
      const originalPostCount = posts.length;
      const allPostsBeforeFilter = [...posts]; // Save all posts before filtering
      posts = posts.filter(post => post.score >= minimumScore);
      const filteredPostCount = posts.length;

      // Log detailed filtering statistics
      const filteredOutCount = originalPostCount - filteredPostCount;
      const filterPercentage = originalPostCount > 0 ? ((filteredOutCount / originalPostCount) * 100).toFixed(1) : 0;

      if (originalPostCount !== filteredPostCount) {
        console.log(chalk.yellow(`[${timestamp}] [FILTER] Filtered out ${filteredOutCount} posts (${filterPercentage}%) with score below ${minimumScore} from r/${subredditName}`));
        console.log(chalk.yellow(`[${timestamp}] [FILTER] Original count: ${originalPostCount}, After filter: ${filteredPostCount}`));

        // Log score distribution of filtered posts for debugging
        const scores = posts.map(p => p.score).sort((a, b) => b - a);
        if (scores.length > 0) {
          console.log(chalk.cyan(`[${timestamp}] [FILTER] Score range of posts that passed: ${scores[scores.length - 1]} to ${scores[0]}`));
        }

        // === CUTOFF REPORT: Include top posts below cutoff when ALL posts filtered out ===
        if (filteredPostCount === 0 && originalPostCount > 0) {
          // Sort all posts by activity (score + num_comments * 2) and take top 5
          const sortedFilteredPosts = allPostsBeforeFilter.sort((a, b) => {
            const activityA = (a.score || 0) + (a.num_comments || 0) * 2;
            const activityB = (b.score || 0) + (b.num_comments || 0) * 2;
            return activityB - activityA;
          });

          // Keep only MAX_FALLBACK_POSTS most-active sub-cutoff posts (option (c))
          const topFilteredPosts = sortedFilteredPosts.slice(0, MAX_FALLBACK_POSTS);

          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ⚠️  NO POSTS FROM r/${subredditName} MET THE MIN-SCORE CUTOFF (${minimumScore})`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] Total posts fetched: ${originalPostCount}, but all had scores below ${minimumScore}`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF]`));
          console.log(chalk.green(`[${timestamp}] [CUTOFF] 📊 INCLUDING TOP ${topFilteredPosts.length} POSTS BELOW CUTOFF IN OUTPUT (no comments fetched):`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ───────────────────────────────────────────────────────────────────`));

          topFilteredPosts.forEach((post, i) => {
            const activity = (post.score || 0) + (post.num_comments || 0) * 2;
            const titlePreview = post.title.length > 65 ? post.title.substring(0, 65) + '...' : post.title;
            console.log(chalk.magenta(`[${timestamp}] [CUTOFF]   ${i + 1}. "${titlePreview}"`));
            console.log(chalk.magenta(`[${timestamp}] [CUTOFF]      Score: ${post.score} | Comments: ${post.num_comments} | Activity: ${activity}`));
            console.log(chalk.magenta(`[${timestamp}] [CUTOFF]      URL: ${post.permalink}`));
            if (post.selftext) {
              const previewText = post.selftext.substring(0, 120).replace(/\n/g, ' ');
              console.log(chalk.magenta(`[${timestamp}] [CUTOFF]      Preview: ${previewText}${post.selftext.length > 120 ? '...' : ''}`));
            }
            console.log(chalk.magenta(`[${timestamp}] [CUTOFF]`));
          });

          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ───────────────────────────────────────────────────────────────────`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] 💡 TIP: Lower --min-score to include more posts (currently: ${minimumScore})`));
          console.log(chalk.magenta(`[${timestamp}] [CUTOFF] ═══════════════════════════════════════════════════════════════════`));

          // USE THE TOP POSTS BELOW CUTOFF instead of returning empty
          posts = topFilteredPosts;
          fallbackBelowMinimumCount = posts.length;
        }
      } else {
        console.log(chalk.green(`[${timestamp}] [FILTER] All ${originalPostCount} posts passed the minimum score filter of ${minimumScore}`));
      }

      // For each post, fetch its top comments — BUT skip comment fetches for fallback (sub-cutoff) posts
      // to save API calls; they're included for completeness, not for deep summarization.
      const isFallbackBatch = fallbackBelowMinimumCount !== undefined && fallbackBelowMinimumCount > 0;
      if (isFallbackBatch) {
        console.log(chalk.yellow(`[${timestamp}] [POSTS] Skipping comment fetch for ${posts.length} fallback (sub-cutoff) post(s) in r/${subredditName}`));
        for (const post of posts) {
          post.top_comments = [];
        }
      } else {
        console.log(chalk.cyan(`[${timestamp}] [POSTS] Fetching comments for ${posts.length} posts in r/${subredditName}...`));
        for (const post of posts) {
          post.top_comments = await fetchTopCommentsForPost(post.permalink, limiter);
        }
      }

      return {
        subreddit: subredditName,
        posts,
        filterStats: {
          totalPostsViewed: originalPostCount,
          postsAfterFilter: posts.length,
          minimumScore,
          ...(fallbackBelowMinimumCount !== undefined ? { fallbackBelowMinimumCount } : {})
        }
      };
    } catch (error) {
      logAxiosCatch(`Posts fetch failed for r/${subredditName}`, error, timestamp);

      if (isRedditBlock(error)) {
        console.error(
          chalk.red(
            `[${timestamp}] [BLOCKED] Reddit returned 403 for r/${subredditName}. ` +
              'This usually means REDDIT_SESSION is missing or expired — refresh the `reddit_session` cookie in .env.'
          )
        );
        throw new Error('Reddit blocked posts; refresh REDDIT_SESSION and resume.');
      }

      if (axios.isAxiosError(error) && error.response?.status === 429) {
        const w429 = reddit429WaitMs(error.response.headers as Record<string, unknown>);
        if (w429 > 0) {
          console.log(
            chalk.yellow(
              `[${timestamp}] [429] Waiting ${Math.round(w429 / 1000)}s per Reddit headers before retry backoff...`
            )
          );
          await sleepMs(w429);
        }
      }

      if (limiter.recordFailure() && retryCount < maxRetries) {
        retryCount++;
        console.log(chalk.yellow(`[${timestamp}] [RETRY] Retrying (${retryCount}/${maxRetries})...`));
        await sleepMs(limiter.backoffTime);
      } else {
        console.error(chalk.red(`[${timestamp}] [ERROR] Max retries reached or too many failures. Giving up on fetching posts from r/${subredditName}`));
        throw new Error(`Posts fetch failed for r/${subredditName}; check REDDIT_SESSION and resume.`);
      }
    }
  }

  return {
    subreddit: subredditName,
    posts: [],
    filterStats: {
      totalPostsViewed: 0,
      postsAfterFilter: 0,
      minimumScore: minimumScore
    }
  };
};

/**
 * Get the rate limiter instance for use in other modules
 */
export const getRateLimiter = (): RateLimiter => {
  return rateLimiter;
};
