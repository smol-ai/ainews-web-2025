// clusterWithLLM.ts
// Simplified Reddit post clustering using only LLM (Instructor) for topic clusters
// Major assumptions and warnings:
// - Only top-level posts are clustered (comments ignored)
// - Each post gets a unique ID for robust mapping
// - Instructor (LLM) is used to cluster posts and name clusters
// - N retries if unique IDs fail to match
// - Logging is frequent, with stage prefixes, timestamps, and color codes
// - Types/utility functions are imported from types.ts/utils.ts

import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import { v4 as uuidv4 } from 'uuid';
import { generateObject, generateText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { RedditPost, SubredditData } from './types';
import { saveResultsToFile } from './utils';
import { z } from 'zod';
import pLimit from 'p-limit';
import axios from 'axios';
import 'dotenv/config';
import { checkpoint, fingerprint } from '../shared/state';

// Try to load environment variables from .env file if dotenv is available
try {
  // Dynamic import to avoid dependency issues if dotenv isn't installed
  // Environment is loaded by the shared CLI before this child starts.
  console.log(chalk.green('[INFO] Loaded environment variables from .env file'));
} catch (err) {
  console.log(chalk.yellow('[WARN] Could not load dotenv, using existing environment variables'));
}

// ========== LOGGING UTILS ==========
const log = (level: 'INFO' | 'ERROR' | 'SUCCESS' | 'WARN' | 'CUTOFF', msg: string) => {
  const t = new Date().toISOString();
  let colorFn = chalk.blue;
  if (level === 'ERROR') colorFn = chalk.red;
  else if (level === 'SUCCESS') colorFn = chalk.green;
  else if (level === 'WARN') colorFn = chalk.yellow;
  else if (level === 'CUTOFF') colorFn = chalk.magenta;
  console.log(colorFn(`[${level}][${t}] ${msg}`));
};

// ========== MODEL CONFIG ==========
/** Single LLM model used for vision, summarization, and clustering.
 * Override with CLUSTER_LLM_MODEL env var. */
const CLUSTER_LLM_MODEL = process.env.CLUSTER_LLM_MODEL || 'gpt-5.5';

// ========== MAIN TYPES ==========
interface PostWithId extends RedditPost {
  uniqueId: string;
  subreddit: string;
  imageInfo?: {
    isImage: boolean;
    imageType?: string;
    url?: string;
    imageDescription?: string;
  };
}

interface TopicCluster {
  name: string;
  postIds: string[];
}

interface ClusterResult {
  clusters: TopicCluster[];
  unmatchedIds: string[];
}

// ========== LOAD AND FLATTEN ==========
function loadRedditPostsJson(filePath: string): SubredditData[] {
  log('INFO', `Loading Reddit posts from ${filePath}`);
  const jsonData = fs.readFileSync(filePath, 'utf8');
  return JSON.parse(jsonData) as SubredditData[];
}

function flattenTopLevelPosts(subredditData: SubredditData[]): PostWithId[] {
  const posts: PostWithId[] = [];
  let postIndex = 1;
  for (const sub of subredditData) {
    for (const post of sub.posts) {
      // Use simple sequential IDs like "p1", "p2", etc. to prevent LLM hallucination
      // The LLM tends to hallucinate random hex strings but is much better at preserving simple sequential IDs
      posts.push({ ...post, subreddit: sub.subreddit, uniqueId: `p${postIndex}` });
      postIndex++;
    }
  }
  log('INFO', `Flattened to ${posts.length} top-level posts with sequential IDs (p1 to p${posts.length}).`);
  return posts;
}

// ========== IMAGE ANALYSIS UTILS ==========
/**
 * Analyze if a Reddit post is an image post and, if so, call the OpenAI Vision API to describe it
 * Returns an object: { isImage: boolean, imageType?: string, url?: string, imageDescription?: string }
 * Assumes image if URL ends with common image extensions or is a known Reddit image host
 *
 * [LINT FIX] Use the OpenAI client directly for vision calls, not instructor.openai (which does not exist).
 */
async function analyzeImagePost(post: PostWithId): Promise<{ isImage: boolean; imageType?: string; url?: string; imageDescription?: string }> {
  if (!post.url) return { isImage: false };
  const match = post.url.match(/\.(jpg|jpeg|png|gif|webp)$/i);
  const isRedditImage = post.url.includes('i.redd.it') || post.url.includes('preview.redd.it');
  const isImage = !!match || isRedditImage;
  let imageType = match ? match[1].toLowerCase() : undefined;
  let imageDescription = '';
  if (isImage) {
    log('INFO', `[VISION][${new Date().toISOString()}] Analyzing image from URL: ${post.url}`);
    try {
      // Use AI SDK for vision API calls
      const visionResponse = await generateText({
        model: openai(CLUSTER_LLM_MODEL),
        messages: [
          {
            role: 'user',
            content: [
              { type: 'text', text: `Describe what you see in this image in 1-4 sentences. Focus on the main subject and any relevant details given the title: ${post.title}` },
              { type: 'image', image: post.url }
            ]
          }
        ],
      });
      imageDescription = visionResponse.text.trim();
      log('SUCCESS', `[VISION][${new Date().toISOString()}] Successfully analyzed image: ${imageDescription}`);
    } catch (visionError: any) {
      log('ERROR', `[VISION][${new Date().toISOString()}] Error analyzing image: ${visionError.message}`);
      throw visionError;
    }
  }
  return { isImage, imageType, url: post.url, imageDescription };
}

// ========== LLM SUMMARIZATION ==========
/**
 * Summarize a post's content and top comments (title + selftext + top 3 comments)
 * Uses pre-analyzed image info that was processed before clustering
 * If the post is an external link (not image, not self), fetches markdown from r.jina.ai and summarizes it as well.
 * @param post - The post to summarize
 * @param imageInfo - Output of analyzeImagePost (must be provided, no longer analyzed inline)
 */
async function summarizePostContent(post: PostWithId): Promise<string> {
  const timestamp = new Date().toISOString();
  log('INFO', `[SUMMARIZE][${timestamp}] Summarizing post content for: ${post.title.substring(0, 60)}... (${post.uniqueId})`);

  // Use the imageInfo attached to the post (analyzed earlier in the workflow)
  const img = post.imageInfo || { isImage: false };

  // Combine title, selftext, and top comments for LLM context
  let combinedText = `Title: ${post.title}\n\n`;
  if (post.selftext) {
    combinedText += `Content: ${post.selftext}\n\n`;
  }
  if (img.isImage && img.imageDescription) {
    combinedText += `[Image Description: ${img.imageDescription}]\n\n`;
  }
  if (post.top_comments && post.top_comments.length > 0) {
    combinedText += "Top Comments (technical, up to 3):\n";
    for (const comment of post.top_comments.slice(0, 3)) {
      combinedText += `- ${comment.body}\n`;
    }
  }
  if (img.isImage) {
    combinedText += `\nImage URL: ${img.url} (type: ${img.imageType})\n`;
  }

  // ========== EXTERNAL LINK SCRAPE & SUMMARY ==========
  // If not an image, not a selfpost, but is an external link, fetch markdown and summarize
  let externalSummary = '';
  const isSelfPost = !!post.selftext && (!post.url || post.url.includes('reddit.com'));
  const isExternalLink = post.url && !img.isImage && !isSelfPost;
  if (isExternalLink) {
    try {
      log('INFO', `[SCRAPE][${timestamp}] Fetching markdown from r.jina.ai for url: ${post.url}`);
      const fetchUrl = `https://r.jina.ai/${encodeURIComponent(post.url)}`;
      // Use axios for HTTP GET with timeout and custom UA
      const res = await axios.get(fetchUrl, {
        headers: { 'User-Agent': 'Mozilla/5.0' },
        timeout: 20000
      });
      if (!res.data) {
        log('ERROR', `[SCRAPE][${timestamp}] No data returned from axios for url: ${fetchUrl}`);
      } else {
        const scrapedMd = res.data;
        log('SUCCESS', `[SCRAPE][${timestamp}] Successfully fetched markdown for url: ${post.url}`);
        // Summarize the fetched markdown with LLM
        log('INFO', `[LLM][${timestamp}] Summarizing external markdown for url: ${post.url}`);
        const extSummarySchema = z.object({
          summary: z.string().describe('A concise technical summary (2-3 sentences) of the main points of the fetched blog/article. Use markdown, link to the original if possible.')
        });
        const extSummaryResp = await generateObject({
          model: openai(CLUSTER_LLM_MODEL),
          schema: extSummarySchema,
          system: 'You are a technical summarizer. Summarize the following markdown content from an external blog/article for an expert audience. Focus on technical, factual content.',
          prompt: scrapedMd.slice(0, 12000), // truncate to avoid token limits
        });
        if (extSummaryResp && extSummaryResp.object.summary) {
          externalSummary = `\n\n[External Link Summary]\n${extSummaryResp.object.summary}`;
        } else {
          log('WARN', `[LLM][${timestamp}] No summary generated for external url: ${post.url}`);
        }
      }
    } catch (err: any) {
      log('ERROR', `[SCRAPE][${timestamp}] Error scraping/summarizing external url: ${err.message}`);
      log('ERROR', `[ERROR STATE] postId=${post.uniqueId}, url=${post.url}`);
    }
  }

  // Zod schema for structured summary (factual + opinions)
  // NOTE: Use `.nullable()` not `.optional()` — strict structured-output mode (gpt-5.5+)
  // requires every property to appear in `required`; only the *value* may be null.
  const SummarySchema = z.object({
    factualInfo: z.string().describe('A concise, highly technical summary (2-3 sentences) of the main announcement, finding, or question, focusing on benchmarks, model details, implementation, bugs, or deep technical debate. Link inline any relevant external links and use markdown formatting for emphasizing as needed, e.g. **bold key people/companies and facts**, `backtick key numbers and  statistics` or *italicize direct quotes* (sprinkle in direct quotes to illustrate, with links where possible). Only include if it would be valuable to a technical reader. Exclude jokes, memes, generic praise, or off-topic discussion.'),
    opinions: z.string().nullable().describe('Any notable opinions or debates from comments, in 1-2 sentences. Pass null if there are none worth including (do NOT omit the key).')
  });

  // Choose system prompt based on image or not
  let systemPrompt = img.isImage
    ? 'You are a technical Reddit summarizer. The following Reddit post contains an image. Summarize the technical or contextual significance of the image, referencing the title, selftext, and comments for context. If the image is a meme or non-technical, state that explicitly. Use markdown and include a link to the image.'
    : 'You are a technical Reddit summarizer. Summarize the following Reddit post and its top comments for an expert audience. Focus on technical, factual content, and use markdown formatting and inline links.';

  try {
    const summaryResponse = await generateObject({
      model: openai(CLUSTER_LLM_MODEL),
      schema: SummarySchema,
      system: systemPrompt,
      prompt: combinedText + externalSummary,
    });
    if (summaryResponse && summaryResponse.object.factualInfo) {
      let summary = `**${summaryResponse.object.factualInfo}**`;
      if (summaryResponse.object.opinions) {
        summary += ` ${summaryResponse.object.opinions}`;
      }
      return summary;
    }
    log('WARN', `[SUMMARIZE][${timestamp}] No summary generated for postId=${post.uniqueId}`);
    return 'No summary generated.';
  } catch (err: any) {
    log('ERROR', `[SUMMARIZE][${timestamp}] Error summarizing post: ${err.message}`);
    log('ERROR', `[ERROR STATE] postId=${post.uniqueId}, title=${post.title}`);
    throw err;
  }
}

// Summarize the general thrust of comments
async function summarizeCommentsStructured(post: PostWithId): Promise<string> {
  const timestamp = new Date().toISOString();
  log('INFO', `[SUMMARIZE][${timestamp}] Summarizing comments for post: ${post.title.substring(0, 50)}... (${post.uniqueId})`);
  try {
    // Prepare comments
    const commentTexts: string[] = [];
    if (post.top_comments && post.top_comments.length > 0) {
      const topComments = post.top_comments.slice(0, 20);
      for (const comment of topComments) {
        commentTexts.push(`${comment.author}: ${comment.body}`);
      }
    }
    // Zod schema for structured summary
    const CommentSummarySchema = z.object({
      commentSummaries: z.array(
        z.string().describe('A 2-4 sentence summary of a highly technical or insightful discussion point (e.g., benchmarks, model details, performance, implementation, bugs, or deep technical debate). Link inline any relevant external links and use markdown formatting for emphasizing as needed, e.g. **bold key people/companies and facts**, `backtick key numbers and statistics` or *italicize direct quotes* (sprinkle in direct quotes to illustrate, with links where possible). Only include if it would be valuable to a technical reader. Exclude jokes, memes, generic praise, or off-topic discussion.')
      ).max(3)
    });
    // Use AI SDK for structured output
    const summaryResponse = await generateObject({
      model: openai(CLUSTER_LLM_MODEL),
      schema: CommentSummarySchema,
      system: `You are an AI assistant that summarizes Reddit comments for technical readers.\n\nIMPORTANT: Only include technical, detailed, or highly insightful comment themes (e.g., benchmarks, performance, implementation details, bugs, or deep technical debate). Exclude jokes, memes, generic praise, or off-topic discussion.\n\nFor each summary, focus on technical content, relevant statistics, direct references to model names, benchmarks, or performance. Limit to 1-3 of the most technically substantive points.\n\nYou must respond with a valid JSON object that matches the schema. Do not include markdown formatting, headers, or any text outside the JSON structure.`,
      prompt: `Post Title: ${post.title}\nComments:\n${commentTexts.join('\n\n')}`,
    });
    if (summaryResponse && summaryResponse.object.commentSummaries && Array.isArray(summaryResponse.object.commentSummaries)) {
      // Format each summary as a bullet point with 4 leading spaces
      const formattedSummaries = summaryResponse.object.commentSummaries
        .map((s: string) => {
          const stripped = s.replace(/\n/g, ' ').trim();
          return stripped ? `    - ${stripped}` : null;
        })
        .filter((s): s is string => s !== null);

      return formattedSummaries.length > 0 ? formattedSummaries.join('\n') : '';
    }
    return 'No summary generated.';
  } catch (err: any) {
    log('ERROR', `[SUMMARIZE][${timestamp}] Failed to summarize comments: ${err.message}`);
    log('ERROR', `[ERROR STATE] postId=${post.uniqueId}, title=${post.title}`);
    throw err;
  }
}

// ========== LLM CLUSTERING ==========
// Check for required environment variables
const apiKey = process.env.AI_GATEWAY_API_KEY || process.env.OPENAI_API_KEY || process.env.LITELLM_PROXY_API_KEY;
const baseURL = process.env.AI_GATEWAY_BASE_URL || process.env.LITELLM_BASE_URL;

if (!apiKey) {
  console.error(chalk.red(`[ERROR] Missing API key environment variables. Please set one of:
  - AI_GATEWAY_API_KEY (for Vercel AI Gateway)
  - OPENAI_API_KEY (for direct OpenAI access)
  - LITELLM_PROXY_API_KEY (for LiteLLM proxy)

You can set these in a .env file or directly in your environment.`));
  // Don't exit here to allow for better error handling
}

if (baseURL) {
  log('INFO', `Using custom base URL: ${baseURL}`);
}

// Initialize OpenAI provider with AI SDK
const openai = createOpenAI({
  apiKey: apiKey || 'sk-missing-key-please-set-env-var',
  baseURL: baseURL,
});

/** Sort IDs like p1, p2, … p10 (not lexicographic p1, p10, p2…) */
function sortPostIdsNaturally(ids: string[]): string[] {
  return [...ids].sort((a, b) => {
    const ma = /^p(\d+)$/i.exec(a);
    const mb = /^p(\d+)$/i.exec(b);
    if (ma && mb) return parseInt(ma[1], 10) - parseInt(mb[1], 10);
    return a.localeCompare(b);
  });
}

/** Activity score used for newsletter-worthiness ranking. */
function postActivity(p: PostWithId): number {
  return (p.score || 0) + (p.num_comments || 0) * 2;
}

/** Pick the top K posts by activity for clustering input. */
function selectTopByActivity(posts: PostWithId[], k: number): PostWithId[] {
  return [...posts].sort((a, b) => postActivity(b) - postActivity(a)).slice(0, k);
}

// Zod schema for Instructor response_model enforcing min/max clusters and post count per cluster
function getClusterSchema(minClusters: number, maxClusters: number, minPostIdsPerCluster: number, maxPostIdsPerCluster: number) {
  // Defensive: ensure max >= min for both dimensions.
  const safeMaxClusters = Math.max(minClusters, maxClusters);
  const safeMaxPerCluster = Math.max(minPostIdsPerCluster, maxPostIdsPerCluster);
  return z.object({
    clusters: z.array(
      z.object({
        name: z.string().min(6)
          .describe('Short, highly specific cluster name (e.g. "OpenAI o4-mini Release Benchmarks")'),
        postIds: z
          .array(z.string())
          .min(minPostIdsPerCluster)
          .max(safeMaxPerCluster)
          .describe(`Between ${minPostIdsPerCluster} and ${safeMaxPerCluster} unique post IDs; copy IDs exactly from the input list`)
      })
    )
      .min(minClusters)
      .max(safeMaxClusters)
      .describe(`Between ${minClusters} and ${safeMaxClusters} thematic clusters of the most newsletter-worthy posts (you may omit posts that don't fit)`)
  });
}

/**
 * Cluster posts using Instructor, enforcing cluster count and size via Zod schema
 * Implements smarter retries: on failure, shows the LLM what was good (matched clusters) and what needs fixing (unmatched posts)
 *
 * IMPORTANT: Uses simple sequential IDs (p1, p2, p3...) to prevent LLM hallucination of random hex strings.
 * Validates that ALL returned postIds exist in the input, and filters out any hallucinated IDs.
 *
 * @param posts - Posts to cluster
 * @param nClusters - Desired number of clusters (min 2)
 * @param maxRetries - Retry attempts
 */
async function clusterPostsWithLLM(
  posts: PostWithId[],
  nClusters: number,
  maxRetries = 3,
  options: { minPostsPerCluster?: number; maxPostsPerCluster?: number } = {}
): Promise<ClusterResult> {
  const minClusters = 1;
  const maxClusters = Math.max(minClusters, nClusters);
  const minPostsPerCluster = Math.max(1, options.minPostsPerCluster ?? 2);
  const maxPostsPerCluster = Math.max(minPostsPerCluster, options.maxPostsPerCluster ?? 5);
  const ClusterSchema = getClusterSchema(minClusters, maxClusters, minPostsPerCluster, maxPostsPerCluster);

  // Build a set of valid IDs for quick validation
  const validIdSet = new Set(posts.map(p => p.uniqueId));
  const validIdList = sortPostIdsNaturally(Array.from(validIdSet));

  // Include image descriptions in the LLM input to improve clustering
  const llmInput = posts.map(p => {
    const baseInput = {
      uniqueId: p.uniqueId,
      title: p.title,
      selftext: p.selftext || ''
    };

    // Add image description if available
    if (p.imageInfo?.isImage && p.imageInfo?.imageDescription) {
      return {
        ...baseInput,
        hasImage: true,
        imageDescription: p.imageInfo.imageDescription
      };
    }

    return baseInput;
  });

  let retries = 0;
  let lastResult: ClusterResult = { clusters: [], unmatchedIds: [] };
  let prevValidClusters: TopicCluster[] = [];
  let prevUnmatchedIds: string[] = [];
  let prevHallucinatedIds: string[] = [];

  while (retries < maxRetries) {
    log('INFO', `LLM clustering attempt ${retries + 1}/${maxRetries}`);

    // Build the system prompt with explicit ID list to prevent hallucination
    let systemPrompt = `You are a senior AI-newsletter editor curating ${nClusters} thematic clusters from a pool of Reddit posts.

OUTPUT TARGET: ${minClusters}-${maxClusters} clusters, each containing ${minPostsPerCluster}-${maxPostsPerCluster} posts. Quality over quantity — this is a tight, hand-edited newsletter, not an exhaustive index.

SELECTION RULES (in priority order):
1. Pick the most newsletter-worthy posts: new model releases, benchmarks, novel research, significant launches, technically meaty discussions. De-prioritize memes, shallow opinion threads, and self-promotion.
2. Within each cluster, pick the ${minPostsPerCluster}-${maxPostsPerCluster} STRONGEST posts — DROP the rest. Do NOT create catch-all buckets. Do NOT include weak posts just to fill a cluster.
3. Each post ID may appear in AT MOST ONE cluster. No duplicates across clusters.
4. It is EXPECTED and CORRECT that many input posts will not appear in any cluster. Omit aggressively.

ID RULES:
- Use ONLY IDs from this exact list (case-sensitive, copy verbatim): [${validIdList.join(', ')}]
- Do NOT invent, modify, or hallucinate IDs.

CONTENT NOTES:
- When hasImage=true, treat imageDescription as post content.
- Cluster names must be concise and specific (e.g. "Qwen 3.6 27B Coding Benchmarks"), not vague (e.g. "AI tools").

Output JSON matching the schema.`;

    // On retry, give LLM feedback about what was good and what needs fixing
    if (retries > 0) {
      if (prevHallucinatedIds.length > 0) {
        systemPrompt += `\n\n[CRITICAL ERROR - HALLUCINATED IDS]\nYour previous response contained INVALID post IDs that don't exist: ${prevHallucinatedIds.join(', ')}\nYou MUST ONLY use IDs from this list: [${validIdList.join(', ')}]\nCopy the IDs EXACTLY.`;
      }
    }

    // Log the input for debugging (truncated)
    log('INFO', `[LLM INPUT] Sending ${posts.length} posts with IDs: ${validIdList.slice(0, 10).join(', ')}${validIdList.length > 10 ? '...' : ''}`);

    try {
      // Track when the LLM request starts
      const llmStartTime = Date.now();
      log('INFO', `[LLM][${new Date().toISOString()}] Sending clustering request to LLM...`);

      const llmResponse = await generateObject({
        model: openai(CLUSTER_LLM_MODEL),
        schema: ClusterSchema,
        system: systemPrompt,
        prompt: JSON.stringify(llmInput),
      });

      log('SUCCESS', `[LLM][${new Date().toISOString()}] Received clustering response in ${Date.now() - llmStartTime}ms`);
      const parsed = llmResponse.object as { clusters: TopicCluster[] };
      if (!parsed.clusters || !Array.isArray(parsed.clusters)) {
        log('ERROR', `[LLM RESPONSE] Missing or invalid 'clusters' property. Full response: ${JSON.stringify(llmResponse, null, 2)}`);
        throw new Error('clusters property missing or not an array');
      }

      // ========== STRONG VALIDATION ==========
      // Step 1: Find any hallucinated IDs (IDs returned by LLM that don't exist in input)
      const allReturnedIds = parsed.clusters.flatMap((c: any) => Array.isArray(c.postIds) ? c.postIds : []);
      const hallucinatedIds = allReturnedIds.filter(id => !validIdSet.has(id));

      if (hallucinatedIds.length > 0) {
        log('ERROR', `[VALIDATION][${new Date().toISOString()}] LLM HALLUCINATED ${hallucinatedIds.length} invalid IDs: ${hallucinatedIds.join(', ')}`);
        prevHallucinatedIds = hallucinatedIds;
      } else {
        log('SUCCESS', `[VALIDATION][${new Date().toISOString()}] All returned IDs are valid (no hallucination detected)`);
        prevHallucinatedIds = [];
      }

      // Step 2: Filter out hallucinated IDs AND dedupe across clusters (keep first occurrence)
      const seenIds = new Set<string>();
      const validatedClusters: TopicCluster[] = parsed.clusters
        .map(cluster => {
          const dedupedIds: string[] = [];
          for (const id of cluster.postIds) {
            if (!validIdSet.has(id)) continue; // hallucinated
            if (seenIds.has(id)) continue; // already in another cluster
            seenIds.add(id);
            dedupedIds.push(id);
          }
          return { name: cluster.name, postIds: dedupedIds };
        })
        .filter(cluster => cluster.postIds.length >= 1);

      const totalReturned = parsed.clusters.reduce(
        (sum, c) => sum + (Array.isArray(c.postIds) ? c.postIds.length : 0),
        0
      );
      const totalKept = validatedClusters.reduce((sum, c) => sum + c.postIds.length, 0);
      const duplicatesDropped = totalReturned - hallucinatedIds.length - totalKept;
      if (duplicatesDropped > 0) {
        log('WARN', `[VALIDATION][${new Date().toISOString()}] Dropped ${duplicatesDropped} duplicate ID occurrence(s) across clusters.`);
      }

      // Step 3: Posts not used by the LLM are EXPECTED in tight-selection mode (we deliberately drop weak posts).
      const usedValidIds = seenIds;
      const unmatchedIds = Array.from(validIdSet).filter(id => !usedValidIds.has(id));
      const totalSelected = usedValidIds.size;

      log('INFO', `[VALIDATION][${new Date().toISOString()}] Clusters: ${validatedClusters.length}, Posts selected: ${totalSelected}/${posts.length} (the rest were intentionally dropped), Hallucinated: ${hallucinatedIds.length}`);

      // Success criteria for tight selection: at least 1 cluster, ≥ minPostsPerCluster posts in each, no hallucinations.
      const allClustersWellSized = validatedClusters.every(c => c.postIds.length >= minPostsPerCluster);
      const goodEnough = validatedClusters.length >= 1 && totalSelected >= minPostsPerCluster;

      if (goodEnough && allClustersWellSized && hallucinatedIds.length === 0) {
        log('SUCCESS', `[CLUSTER][${new Date().toISOString()}] Tight selection: ${validatedClusters.length} clusters, ${totalSelected} posts (omitted ${unmatchedIds.length}).`);
        saveClusterCheckpoint(validatedClusters, unmatchedIds, posts.length, retries + 1, false);
        return { clusters: validatedClusters, unmatchedIds };
      }

      if (goodEnough) {
        // Has clusters but maybe a too-small one or hallucinated IDs — accept best-effort, do not retry forever.
        log('WARN', `[CLUSTER][${new Date().toISOString()}] Acceptable result: ${validatedClusters.length} clusters, ${totalSelected} posts. ${hallucinatedIds.length ? `${hallucinatedIds.length} hallucinated IDs (filtered).` : ''}${allClustersWellSized ? '' : ' Some clusters below min size.'}`);
        lastResult = { clusters: validatedClusters, unmatchedIds };
        prevValidClusters = validatedClusters;
        prevUnmatchedIds = unmatchedIds;

        if (hallucinatedIds.length > 0 && retries < maxRetries - 1) {
          log('INFO', `[CLUSTER] Retrying to fix hallucinated IDs...`);
          saveClusterCheckpoint(validatedClusters, unmatchedIds, posts.length, retries + 1, true);
          retries++;
          continue;
        }

        saveClusterCheckpoint(validatedClusters, unmatchedIds, posts.length, retries + 1, true);
        return { clusters: validatedClusters, unmatchedIds };
      }

      // Poor result - retry
      log('WARN', `[CLUSTER][${new Date().toISOString()}] Poor result: ${validatedClusters.length} clusters / ${totalSelected} posts, retrying...`);
      lastResult = { clusters: validatedClusters, unmatchedIds };
      prevValidClusters = validatedClusters;
      prevUnmatchedIds = unmatchedIds;
      saveClusterCheckpoint(validatedClusters, unmatchedIds, posts.length, retries + 1, true);

    } catch (err: any) {
      log('ERROR', `LLM clustering failed: ${err.message}`);
      log('ERROR', `Error stack: ${err.stack}`);
    }
    retries++;
  }

  // After all retries, return best effort
  if (lastResult.clusters.length > 0) {
    const validPostCount = lastResult.clusters.reduce((sum, c) => sum + c.postIds.length, 0);
    log('WARN', `[CLUSTER] Returning best effort after ${maxRetries} retries: ${lastResult.clusters.length} clusters with ${validPostCount} posts assigned, ${lastResult.unmatchedIds.length} unmatched`);
  } else {
    log('ERROR', `[CLUSTER] Failed to create any valid clusters after ${maxRetries} retries.`);
  }

  return lastResult;
}

/**
 * Helper function to save cluster checkpoint
 */
function saveClusterCheckpoint(
  clusters: TopicCluster[],
  unmatchedIds: string[],
  totalPosts: number,
  attempt: number,
  isPartial: boolean
) {
  const checkpointData = {
    timestamp: new Date().toISOString(),
    runId: uuidv4().slice(0, 8),
    clusters,
    unmatchedIds,
    stats: {
      totalPosts,
      clusterCount: clusters.length,
      assignedPosts: clusters.reduce((sum, c) => sum + c.postIds.length, 0),
      unmatchedCount: unmatchedIds.length,
      attempt
    }
  };
  const checkpointDir = path.join(process.cwd(), 'checkpoints');
  if (!fs.existsSync(checkpointDir)) fs.mkdirSync(checkpointDir);
  const suffix = isPartial ? '_partial_cluster_checkpoint.json' : '_cluster_checkpoint.json';
  const checkpointPath = path.join(checkpointDir, `${new Date().toISOString().replace(/[:.]/g, '-')}${suffix}`);
  fs.writeFileSync(checkpointPath, JSON.stringify(checkpointData, null, 2));
  log('INFO', `[CHECKPOINT][${new Date().toISOString()}] Saved ${isPartial ? 'partial' : 'complete'} clustering to ${checkpointPath}`);
}

// ========== MARKDOWN OUTPUT FORMATTING ==========
/**
 * WARNING: This function formats markdown output for Reddit clusters.
 * Major assumption: Each bullet (post) should have two leading spaces for proper markdown indentation, including the first bullet.
 * If you change bullet formatting, check for rendering in all markdown viewers.
 *
 * Fix: Add two leading spaces to every bullet, including the first, for consistent indentation.
 *
 * Logs: Prints informative logs with stage prefixes, timestamps, and color codes.
 */
function clustersToMarkdown(clusters: TopicCluster[], postsById: Map<string, PostWithId>, postSummaries: Map<string, string>, commentSummaries: Map<string, string>): string {
  const timestamp = new Date().toISOString();
  log('INFO', `[MARKDOWN][${timestamp}] Formatting clusters to markdown with proper bullet indentation.`);
  let md = ``;
  clusters.forEach((cluster, i) => {
    md += `### ${i + 1}. ${cluster.name}\n\n`;
    cluster.postIds.forEach(id => {
      const post = postsById.get(id);
      if (post) {
        // Add two leading spaces to every bullet for consistent indentation
        md += `  - **[${post.title}](${post.permalink})** (Activity: ${post.score + post.num_comments * 2})`;
        // Post summary (if any)
        const postSummary = postSummaries.get(id);
        const hasPostSummary = !!(postSummary && postSummary.trim());
        if (hasPostSummary) {
          // Ensure post summary ends with two newlines for clear separation from comments
          md += `: ${postSummary.replace(/\n/g, ' ').trim()}\n\n`;
        }
        // Comments summary (if any)
        const commentSummary = commentSummaries.get(id);
        if (commentSummary && commentSummary.trim()) {
          // If there was no post summary, ensure the first comment bullet starts on a new line
          if (!hasPostSummary) {
            md += `\n`;
          }
          // Append comment summary directly, preserving leading spaces for indentation
          md += `${commentSummary.trimEnd()}\n`;
        }
        md += `\n`;
      } else {
        md += `  - [MISSING POST: ${id}]\n`;
      }
    });
    md += `\n`;
  });
  log('SUCCESS', `[MARKDOWN][${timestamp}] Markdown formatting complete.`);
  return md;
}

// ========== MAIN WORKFLOW ==========
const SUMMARY_CONCURRENCY = 4; // Tune as needed for rate limits
const summaryLimit = pLimit(SUMMARY_CONCURRENCY);

export async function main() {
  // Parse args
  const args = process.argv.slice(2);
  const fileArg = args.indexOf('--file');
  const filePath = fileArg >= 0 ? args[fileArg + 1] : undefined;
  const nClustersArg = args.indexOf('--clusters');
  const nClusters = nClustersArg >= 0 ? parseInt(args[nClustersArg + 1]) : 3;

  // Tight-newsletter knobs
  const topKArg = args.indexOf('--top-k');
  const topK = topKArg >= 0 ? parseInt(args[topKArg + 1]) : 25;
  const maxPerClusterArg = args.indexOf('--max-per-cluster');
  const maxPerCluster = maxPerClusterArg >= 0 ? parseInt(args[maxPerClusterArg + 1]) : 5;
  const minPerClusterArg = args.indexOf('--min-per-cluster');
  const minPerCluster = minPerClusterArg >= 0 ? parseInt(args[minPerClusterArg + 1]) : 2;

  if (!filePath) {
    log('ERROR', 'No --file argument provided.');
    process.exit(1);
  }

  // Check for API keys before starting
  const hasApiKey = process.env.AI_GATEWAY_API_KEY || process.env.OPENAI_API_KEY || process.env.LITELLM_PROXY_API_KEY;
  if (!hasApiKey) {
    log('ERROR', 'Missing required API keys. Please set AI_GATEWAY_API_KEY, OPENAI_API_KEY, or LITELLM_PROXY_API_KEY environment variables.');
    log('INFO', 'You can create a .env file with these variables or set them in your environment.');
    process.exit(1);
  }

  log('INFO', `Starting clustering workflow (model=${CLUSTER_LLM_MODEL}, clusters=${nClusters}, top-k=${topK}, per-cluster=${minPerCluster}-${maxPerCluster})`);
  const start = Date.now();
  const subredditData = loadRedditPostsJson(filePath);
  const allPosts = flattenTopLevelPosts(subredditData);

  // --- TOP-K PRE-CLUSTER SELECTION ---
  // Rank by activity (score + 2*comments) and keep only the top K candidates.
  // This caps the LLM input so the editor can do a tight selection rather than indexing the whole pool.
  const posts = selectTopByActivity(allPosts, topK);
  if (allPosts.length > posts.length) {
    const minKept = Math.min(...posts.map(postActivity));
    const droppedCount = allPosts.length - posts.length;
    log('INFO', `[TOP-K] Pre-cluster selection: kept top ${posts.length}/${allPosts.length} posts by activity (dropped ${droppedCount}; activity floor = ${minKept})`);
  }
  // Re-key the surviving posts with fresh sequential IDs so the LLM sees a clean list.
  posts.forEach((p, i) => { p.uniqueId = `p${i + 1}`; });
  const postsById = new Map(posts.map(p => [p.uniqueId, p]));
  const cacheRoot = path.join(process.cwd(), 'checkpoints', fingerprint({ posts, nClusters, minPerCluster, maxPerCluster, model: CLUSTER_LLM_MODEL, baseURL, version: 1 }));
  const cached = <T>(name: string, operation: () => Promise<T>) => checkpoint(path.join(cacheRoot, `${name}.json`), operation);
  const failures: string[] = [];

  // --- IMAGE ANALYSIS (before clustering) ---
  log('INFO', `[IMAGE][${new Date().toISOString()}] Analyzing image posts before clustering...`);

  // Process image posts with concurrency and enhance posts directly
  const imageStartTime = Date.now();
  log('INFO', `[IMAGE][${new Date().toISOString()}] Starting image analysis for ${posts.length} posts...`);

  await Promise.all(posts.map(post => summaryLimit(async () => {
    const postStartTime = Date.now();
    try {
      // Analyze image and add info directly to the post object
      post.imageInfo = await cached(`image-${post.uniqueId}`, () => analyzeImagePost(post));
      log('INFO', `[IMAGE][${new Date().toISOString()}] Analyzed post ${post.uniqueId} in ${Date.now() - postStartTime}ms`);
    } catch (err: any) {
      log('ERROR', `[IMAGE][${new Date().toISOString()}] Failed to analyze image for postId=${post.uniqueId}: ${err.message}`);
      log('ERROR', `[IMAGE][${new Date().toISOString()}] Error state: title=${post.title}, url=${post.url || 'none'}`);
      failures.push(`image-${post.uniqueId}`);
    }
  })));

  log('SUCCESS', `[IMAGE][${new Date().toISOString()}] Completed image analysis for ${posts.length} posts in ${Date.now() - imageStartTime}ms`);
  if (failures.length) throw new Error(`Image analysis incomplete: ${failures.join(', ')}. Resume to retry failed items.`);

  // --- CLUSTERING ---
  const clusterResult = await cached('clusters', async () => {
    const result = await clusterPostsWithLLM(posts, nClusters, 3, {
    minPostsPerCluster: minPerCluster,
    maxPostsPerCluster: maxPerCluster,
    });
    if (posts.length && !result.clusters.length) throw new Error('Clustering failed; resume to retry.');
    return result;
  });
  let clusters = clusterResult.clusters;

  // --- CHECK FOR EMPTY/INVALID RESULTS ---
  // With the new validation in clusterPostsWithLLM, clusters should only contain valid postIds
  // But we still check for placeholder/example data or completely invalid results

  // Count how many valid posts are actually in clusters
  const validPostsInClusters = clusters.reduce((sum, c) => {
    const validIds = c.postIds.filter(id => postsById.has(id));
    return sum + validIds.length;
  }, 0);

  // Detect placeholder results (example names or zero valid posts)
  const isPlaceholderResult = clusters.length > 0 && (
    clusters.every(c => c.name.toLowerCase().includes('example')) ||
    validPostsInClusters === 0
  );

  const hasValidPosts = posts.length > 0;
  const hasValidClusters = clusters.length > 0 && validPostsInClusters > 0 && !isPlaceholderResult;

  log('INFO', `[VALIDATION] Posts: ${posts.length}, Clusters: ${clusters.length}, Valid posts in clusters: ${validPostsInClusters}, isPlaceholder: ${isPlaceholderResult}, hasValidClusters: ${hasValidClusters}`);

  // Track failure info for markdown output
  let cutoffTriggered = false;
  // Distinguish: was the failure (a) no input posts, or (b) posts existed but clustering returned nothing?
  // Case (b) almost always means the LLM call failed (model 404, schema rejection, hallucination after retries)
  // — it is NOT a score-cutoff problem and the user shouldn't be told to lower --min-score.
  let failureMode: 'no_input_posts' | 'clustering_failed' | null = null;
  let topPostsBelowCutoff: PostWithId[] = [];

  if (!hasValidPosts || !hasValidClusters) {
    cutoffTriggered = true;
    failureMode = posts.length === 0 ? 'no_input_posts' : 'clustering_failed';

    log('CUTOFF', '═══════════════════════════════════════════════════════════════════');
    if (failureMode === 'no_input_posts') {
      log('CUTOFF', '⚠️  NO INPUT POSTS — nothing to cluster');
      log('CUTOFF', '═══════════════════════════════════════════════════════════════════');
      log('CUTOFF', 'No posts were available in the input data.');
      log('CUTOFF', 'Tip: lower --min-score or expand subreddit list.');
    } else {
      log('CUTOFF', '⚠️  CLUSTERING FAILED — posts were available but the LLM produced no valid clusters');
      log('CUTOFF', '═══════════════════════════════════════════════════════════════════');
      log('CUTOFF', `Total posts that reached the LLM: ${posts.length}.`);
      log('CUTOFF', 'This is usually a model/API problem (model name 404, schema rejection, or repeated hallucinations).');
      log('CUTOFF', 'Check the [LLM] error lines above for the underlying cause.');
      log('CUTOFF', 'Tip: try a different CLUSTER_LLM_MODEL or relax --min-per-cluster.');

      // Show the top inputs we DID send to the LLM, for context
      const sortedPosts = [...posts].sort((a, b) => {
        const activityA = (a.score || 0) + (a.num_comments || 0) * 2;
        const activityB = (b.score || 0) + (b.num_comments || 0) * 2;
        return activityB - activityA;
      });
      topPostsBelowCutoff = sortedPosts.slice(0, 2);
      log('CUTOFF', '');
      log('CUTOFF', '📊 TOP 2 INPUT POSTS (these reached the LLM but no clusters came back):');
      log('CUTOFF', '───────────────────────────────────────────────────────────────────');
      topPostsBelowCutoff.forEach((post, i) => {
        const activity = (post.score || 0) + (post.num_comments || 0) * 2;
        log('CUTOFF', `  ${i + 1}. "${post.title.substring(0, 70)}${post.title.length > 70 ? '...' : ''}"`);
        log('CUTOFF', `     Subreddit: r/${post.subreddit}`);
        log('CUTOFF', `     Score: ${post.score || 0} | Comments: ${post.num_comments || 0} | Activity: ${activity}`);
        log('CUTOFF', `     URL: ${post.permalink || post.url || 'N/A'}`);
        log('CUTOFF', '');
      });
      log('CUTOFF', '───────────────────────────────────────────────────────────────────');
    }
    log('CUTOFF', '═══════════════════════════════════════════════════════════════════');

    // Clear invalid placeholder clusters to avoid confusing output
    clusters = [];
  }

  // --- SUMMARIZATION (concurrent, only clustered posts) ---
  log('INFO', 'Summarizing posts and comments with concurrency (only clustered posts)...');
  const postSummaries = new Map<string, string>();
  const commentSummaries = new Map<string, string>();

  // Collect all unique postIds from clusters
  const clusteredPostIds = Array.from(new Set(clusters.flatMap(cluster => cluster.postIds)));
  log('INFO', `[SUMMARY][${new Date().toISOString()}] Will summarize ${clusteredPostIds.length} clustered posts (across all subreddits if present).`);

  await Promise.all(clusteredPostIds.map(postId => summaryLimit(async () => {
    const post = postsById.get(postId);
    if (!post) {
      log('WARN', `[SUMMARY][${new Date().toISOString()}] PostId ${postId} not found in postsById, skipping.`);
      return;
    }
    try {
      // Use the pre-analyzed image info already attached to the post
      postSummaries.set(postId, await cached(`post-${postId}`, () => summarizePostContent(post)));
    } catch (err: any) {
      log('ERROR', `[SUMMARIZE][POST] Failed for postId=${postId}: ${err.message}`);
      postSummaries.set(postId, 'Error summarizing post.');
      failures.push(`post-${postId}`);
    }
  })));
  await Promise.all(clusteredPostIds.map(postId => summaryLimit(async () => {
    const post = postsById.get(postId);
    if (!post) {
      log('WARN', `[SUMMARY][${new Date().toISOString()}] PostId ${postId} not found in postsById, skipping.`);
      return;
    }
    try {
      commentSummaries.set(postId, await cached(`comments-${postId}`, () => summarizeCommentsStructured(post)));
    } catch (err: any) {
      log('ERROR', `[SUMMARIZE][COMMENTS] Failed for postId=${postId}: ${err.message}`);
      commentSummaries.set(postId, 'Error summarizing comments.');
      failures.push(`comments-${postId}`);
    }
  })));

  if (failures.length) throw new Error(`Summaries incomplete: ${failures.join(', ')}. Resume to retry failed items.`);
  // Save and print JSON
  const runId = uuidv4().slice(0, 8);
  const timestamp = new Date().toISOString();
  // Create the JSON output (image info is already part of the posts)
  const jsonOutput = {
    clusters,
    runId,
    timestamp,
    postSummaries: Object.fromEntries(postSummaries),
    commentSummaries: Object.fromEntries(commentSummaries)
  };
  const jsonPath = saveResultsToFile(jsonOutput, 'llm_clusters', runId);
  log('SUCCESS', `Cluster JSON written to ${jsonPath}`);
  console.log(JSON.stringify(jsonOutput, null, 2));

  // Format and save markdown
  let md = clustersToMarkdown(clusters, postsById, postSummaries, commentSummaries);

  // Add failure note to markdown if triggered
  if (cutoffTriggered) {
    let cutoffMd = '';
    if (failureMode === 'clustering_failed') {
      cutoffMd += `## ⚠️ Clustering Failed\n\n`;
      cutoffMd += `> **Note:** ${posts.length} post(s) reached the LLM, but no valid clusters were returned. This is usually a **model/API issue** (e.g. \`CLUSTER_LLM_MODEL\` not available, schema rejection, or repeated ID hallucination), **not** a score-cutoff issue. Check the run logs for \`[LLM] ... error\` lines.\n\n`;
      cutoffMd += `Current model: \`${CLUSTER_LLM_MODEL}\`\n\n`;

      if (topPostsBelowCutoff.length > 0) {
        cutoffMd += `### 📊 Top ${topPostsBelowCutoff.length} Input Posts (sent to LLM)\n\n`;
        cutoffMd += `These posts were available but the LLM call failed:\n\n`;
        topPostsBelowCutoff.forEach((post, i) => {
          const activity = (post.score || 0) + (post.num_comments || 0) * 2;
          cutoffMd += `${i + 1}. **[${post.title}](${post.permalink || post.url || '#'})**\n`;
          cutoffMd += `   - Subreddit: r/${post.subreddit}\n`;
          cutoffMd += `   - Score: ${post.score || 0} | Comments: ${post.num_comments || 0} | Activity: ${activity}\n`;
          if (post.selftext) {
            const previewText = post.selftext.substring(0, 200).replace(/\n/g, ' ').trim();
            cutoffMd += `   - Preview: *${previewText}${post.selftext.length > 200 ? '...' : ''}*\n`;
          }
          cutoffMd += `\n`;
        });
      }

      cutoffMd += `---\n\n`;
      cutoffMd += `💡 **Tips:** verify \`CLUSTER_LLM_MODEL\` is a model your account can access; relax \`--min-per-cluster\`; re-run.\n`;
    } else {
      cutoffMd += `## ⚠️ No Input Posts\n\n`;
      cutoffMd += `> **Note:** No posts were available in the input data — nothing was sent to the LLM.\n\n`;
      cutoffMd += `💡 **Tip:** Lower \`--min-score\`, expand the subreddit list, or check that the subreddit names are correct.\n`;
    }

    md = cutoffMd;
  }

  const specialsDir = path.join(process.cwd(), 'reports');
  if (!fs.existsSync(specialsDir)) fs.mkdirSync(specialsDir);
  const mdPath = path.join(specialsDir, 'llm_clusters.md');
  fs.writeFileSync(mdPath, md);
  log('SUCCESS', `Cluster markdown written to ${mdPath}`);
  log('INFO', `Workflow complete in ${((Date.now() - start) / 1000).toFixed(2)}s`);
}

{
  main().catch(e => {
    log('ERROR', 'Fatal error in main');
    log('ERROR', e.message);
    log('ERROR', e.stack);
    process.exit(1);
  });
}

// End of file
