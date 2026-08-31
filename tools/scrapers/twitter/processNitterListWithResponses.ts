import path from 'path';
import fs from 'fs';
import { checkpoint, fingerprint, writeJson } from '../shared/state.js';
import minimist from 'minimist';
import * as dotenv from 'dotenv';
import OpenAI from 'openai';
import { scrapeTarget, writeResultToJson } from './scrapeNitterList.js';
import { ScrapeResult, Tweet } from './types.js';
import { resolveWindow } from './time.js';

dotenv.config();

interface PipelineArgs {
  listId?: string;
  handle?: string;
  pageType: string; // 'list' | 'profile' | future types
  start?: string | Date;
  end?: string | Date;
  baseUrl?: string;
  out?: string;
  mockHtmlDir?: string;
  maxPages?: number;
  model?: string;
  system?: string;
  jsonInput?: string; // Path to existing scrape JSON to resume from
  maxChars?: number; // Max character limit for input (default: 200000)
  focusTopic?: string; // Optional topic to deep dive on
}

// ---------- Formatting helpers (vendored) ----------
function cleanTweetText(text: string): string {
  if (!text) return '';
  let cleaned = text;
  cleaned = cleaned.replace(/\n/g, ' ');
  cleaned = cleaned.replace(/\s+/g, ' ');
  cleaned = cleaned.replace(/\(/g, '［').replace(/\)/g, '］');
  return cleaned.trim();
}

function computeEngagement(t: Tweet): number {
  const s = t.stats || ({} as any);
  const likes = s.likes || 0;
  const rts = s.retweets || 0;
  const replies = s.replies || 0;
  const quotes = s.quotes || 0;
  return likes + 2 * rts + 0.5 * replies + 1.5 * quotes;
}

function toTupleLine(t: Tweet): string {
  const text = cleanTweetText(t.text || '');
  const score = computeEngagement(t);
  const url = t.url;
  return `(${text}, ${score.toFixed(2)}, ${url})`;
}

function formatTweetsForModel(tweets: Tweet[], maxChars: number = 300000): { lines: string[]; block: string; truncated: boolean; originalCount: number } {
  const lines = tweets.map(toTupleLine);
  let block = lines.join('\n\n');
  let truncated = false;
  const originalCount = lines.length;

  // If block exceeds maxChars, truncate from the start (keeping most recent tweets)
  if (block.length > maxChars) {
    truncated = true;
    const separator = '\n\n';
    let currentLength = 0;
    let startIndex = lines.length;

    // Work backwards from the end to keep most recent tweets
    for (let i = lines.length - 1; i >= 0; i--) {
      const lineLength = lines[i].length + separator.length;
      if (currentLength + lineLength > maxChars) {
        startIndex = i + 1;
        break;
      }
      currentLength += lineLength;
    }

    const keptLines = lines.slice(startIndex);
    block = keptLines.join(separator);

    return { lines: keptLines, block, truncated, originalCount };
  }

  return { lines, block, truncated, originalCount };
}

async function readSystemPrompt(filePath?: string, focusTopic?: string): Promise<string> {
  const FOCUS_TOPIC_PROMPT = `ROLE
You are an investigative AI editor writing for expert AI engineers. Your job is to provide comprehensive deep-dive analysis on a specific topic alongside standard coverage of other discussions.

AUDIENCE
- Technical, detail-oriented AI engineers and researchers
- Comfortable with model specs, benchmarks, systems, and infra

INPUT FORMAT
- You receive a list of tuples, one per line: (tweet_text, engagement_score, tweet_url)
- tweet_url may use nitter.net or xcancel.com; rewrite all such links to x.com in the output
- A FOCUS TOPIC will be specified that requires comprehensive coverage

OUTPUT GOAL
- Produce a polished digest in Markdown with two main parts:
  1. A comprehensive deep-dive section on the focus topic
  2. Standard coverage of remaining topics
- Concise but information-dense; avoid filler and generic statements
- Grounded: always link to original tweets inline

STRUCTURE AND STYLE

**Part 1: Focus Topic Deep Dive**
- Start with a bold heading: **Top Story: [Topic Name]**
- Followed by a comprehensive narrative covering:
  - **What happened**: A short bold lead-in sentence, then a set of bullet points covering all the top information (one key development per bullet). Do not write this section as a single paragraph.
  - **Facts vs opinions**: Clearly distinguish between factual claims and opinion pieces
  - **Technical details**: Extract all technical specifications, numbers, statistics, data points
  - Emphasize official perspectives and independent research evals, de-emphasize and consolidate vendor self promotion.
  - **Different opinions**: Present all viewpoints (supporting, opposing, neutral)
  - **Context**: Background information that helps understand why this matters
- Include as many relevant tweets about this topic, as long as it contributes a point to the discussion.
- Use subheadings (## with double blank lines after) within the focus section to organize complex narratives
- Make it comprehensive enough that readers don't need to click through
- Do NOT end the focus section with a "Bottom line", "Takeaway", "TL;DR", or any concluding wrap-up sentence/paragraph; end on the last substantive point

**Part 2: Other Topics**
- Organize remaining tweets into up to 5 categories max
- Each category starts with a bold title line by itself, then double blank lines
- Follow with bulleted content using the same style as the standard prompt

CONTENT CURATION RULES
- For the focus topic: Be exhaustive - include every relevant tweet
- For other topics: Merge closely related tweets/threads when appropriate
- Prefer technical substance: new models, benchmarks, training/inference techniques. Omit vague hype and vendor self promotion. Consolidate Ecosystem rollout / availability type posts that all sound the same into one line.
- Be precise: include licensing, sizes, architectures, notable metrics
- Keep tone professional and neutral; avoid hype unless reporting it

LINKS
- Transform all links from nitter.net/... or xcancel.com/... to x.com/... in the output
- Use inline Markdown link format: [@handle](https://x.com/handle/status/...)

QUALITY CHECKLIST
- Focus topic section is comprehensive and exhaustive
- "What happened" uses a bold lead-in plus bullet points (not a single paragraph)
- No "Bottom line"/concluding wrap-up at the end of the focus section
- Remaining topics follow category/heading/bullets structure
- All links rewritten to x.com
- Uses bold sparingly for scannability
- No generic preambles or closings

Before finalizing, self-check that: Focus topic section exists and is comprehensive; "What happened" is bullet-pointed; there is no bottom-line/closing wrap-up; remaining topics use 3-5 categories; each category has bullets with inline links; links are twitter.com.
`;

  const DEFAULT = `ROLE
You are a senior AI editor writing for expert AI engineers. Your job is to convert a set of tweet tuples into a high-quality, readable, well-curated technical digest.

AUDIENCE
- Technical, detail-oriented AI engineers and researchers
- Comfortable with model specs, benchmarks, systems, and infra

INPUT FORMAT
- You receive a list of tuples, one per line: (tweet_text, engagement_score, tweet_url)
- tweet_url may use nitter.net or xcancel.com; rewrite all such links to x.com in the output

OUTPUT GOAL
- Produce a polished daily digest in Markdown resembling the given house style (see FEW-SHOT below)
- Concise but information-dense; avoid filler and generic statements
- Curate: elevate what matters, collapse duplicates/threads, remove noise
- Grounded: always link to original tweets inline

STRUCTURE AND STYLE
- Organize into up to 6 categories max
- Each category starts with a bold title (should be as specific as possible - not just "Model Releases and Performance" but specifically "Zhipu AI's GLM-4.5 Model release and Qwen3 and Kimi K2 Models") line by itself, then a blank line
- Immediately follow with bulleted content:
  - Start some bullets with a bold short label (e.g., **Zhipu AI's GLM-4.5 Models**:) followed by a precise summary
  - Use inline links to tweets in context
  - Bold important nouns or statistics sparingly for scannability
- Prefer one or two rich bullets over many shallow ones
- Keep the whole digest to ~600–1200 words
- No intro/outro; start directly with the first category title
- Never include the raw input tuples in the final output
- Rewrite nitter.net and xcancel.com links to x.com

CONTENT CURATION RULES
- Merge closely related tweets/threads into one summarized bullet when appropriate
- Prefer technical substance: new models, benchmarks, training/inference techniques, system/infra changes, agent frameworks, toolchains
- Be precise: include licensing, sizes, architectures, notable metrics where present; do not invent facts
- Keep tone professional and neutral; avoid hype unless it's part of the reporting

LINKS
- Transform all links from nitter.net/... or xcancel.com/... to x.com/... in the output
- Use inline Markdown link format: [@handle](https://x.com/handle/status/...) or [short summary of the content](https://openai.com/blog/...)

FEW-SHOT EXAMPLE
Input (tuples):
~~~
(We released GLM-4.5 under MIT, 355B MoE (32B active) with strong long-form performance, 122.0, https://nitter.net/Zai_org/status/1950439632363020738#m)
(MetaCLIP 2 scales CLIP multilingual training; paper+code; #ACL2025, 187.0, https://nitter.net/jaseweston/status/1950366185742016935#m)
(Runway Aleph: in-context video model; "make it night" demo vs long manual workflow, 45.0, https://nitter.net/c_valenzuelab/status/1950138170806312974#m)
~~~

Desired Output style (markdown):
~~~
**Model Releases and Performance**

- **China's Open-Source Offensive**: In July, Chinese labs released a wave of powerful, permissively licensed models, a trend highlighted by [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1950034092457939072). Key releases include **GLM-4.5** & **GLM-4.5-Air** from **Zhipu AI**, **Wan-2.2** (video), the **Qwen3 Coder** and **Qwen3-235B** family from **Alibaba**, and **Kimi K2** from **Moonshot AI**. This contrasts with a perceived slowdown in Western open-source releases, prompting [@corbtt](https://twitter.com/corbtt/status/1950334347971874943) to note that orgs avoiding these models are at a "significant competitive disadvantage."
- **Zhipu AI's GLM-4.5 Models**: **Zhipu AI** released **GLM-4.5**, a 355B parameter MoE (32B active) model, and **GLM-4.5-Air**, both with **MIT licenses**. The company announced [they are working to scale resources](https://twitter.com/Zai_org/status/1950164491125043515) due to high demand. The models are noted as being competitive with **Claude 4 Opus** and beating **Gemini 2.5 Pro** [in some benchmarks](https://twitter.com/Zai_org/status/1949970927006949430). The community quickly made them available on platforms like **MLX** and **DeepInfra**.
~~~

QUALITY CHECKLIST
- Follows category/heading/bullets structure above
- Rewrites all links to twitter.com
- Curates and condenses; no raw tuple dump
- Contains 3–6 categories total
- Includes a "Top tweets (by engagement)" block where appropriate - filter out non tech tweets.
- Uses bold sparingly for scannability
- No generic preambles or closings

Before finalizing, self-check that: 3–6 categories exist; each category has bullets with inline links; no analysis headings appear; links are twitter.com; Top tweets block exists.
`;

  // If custom system prompt file is provided, use it
  if (filePath) {
    try {
      return fs.readFileSync(filePath, 'utf8').trim();
    } catch {
      // Fall through to check focusTopic or return default
    }
  }

  // If focusTopic is specified, use the focus topic prompt
  if (focusTopic) {
    return FOCUS_TOPIC_PROMPT;
  }

  // Otherwise use the standard prompt
  return DEFAULT;
}

function buildUserContent(formattedTweets: string, focusTopic?: string): string {
  if (focusTopic) {
    return (
      `Please analyze the following tweets with a specific focus topic for deep analysis.\n\n` +
      `**FOCUS TOPIC: ${focusTopic}**\n\n` +
      `For this topic, gather ALL relevant facts, opinions, perspectives, and context. Present a comprehensive narrative that includes:\n` +
      `- Different perspectives (supporting, opposing, neutral)\n` +
      `- Technical details, statistics, and data points\n` +
      `- Context and implications\n\n` +
      `After the focus topic deep dive, provide a standard summary of the remaining tweets.\n\n` +
      `Tweets to analyze (${formattedTweets.split('\n').length} lines total):\n${formattedTweets}`
    );
  }

  return (
    `Please analyze the following tweets and provide a comprehensive summary. Focus on:\n` +
    `Tweets to analyze (${formattedTweets.split('\n').length} lines total):\n${formattedTweets}`
  );
}

function extractOutputText(resp: OpenAI.Responses.Response): string {
  const asText = (resp as any).output_text as string | undefined;
  if (asText && typeof asText === 'string' && asText.trim()) return asText.trim();
  const parts: string[] = [];
  const output = ((resp as any).output || []) as any[];
  for (const item of output) {
    if (!item || typeof item !== 'object') continue;
    const type = (item as any).type as string | undefined;
    if (type === 'message' && Array.isArray((item as any).content)) {
      for (const c of (item as any).content as any[]) {
        if (c && typeof c === 'object') {
          if (c.type === 'output_text' && typeof c.text === 'string') parts.push(c.text);
          if (c.type === 'text' && typeof c.text === 'string') parts.push(c.text);
        }
      }
    }
    if (type === 'output_text' && typeof (item as any).text === 'string') parts.push((item as any).text);
  }
  return parts.join('').trim();
}

function extractUsage(resp: OpenAI.Responses.Response) {
  const usage = (resp as any).usage || {};
  return {
    input: usage.input_tokens,
    output: usage.output_tokens,
    total: usage.total_tokens,
  } as { input?: number; output?: number; total?: number };
}

async function summarizeTweets(linesBlock: string, model: string, systemPrompt: string, focusTopic?: string): Promise<{ text: string; usage?: { input?: number; output?: number; total?: number } }> {
  const baseURL = process.env.OPENAI_BASE_URL || process.env.LITELLM_BASE_URL || undefined;
  const apiKey = baseURL
    ? (process.env.LITELLM_PROXY_API_KEY || process.env.OPENAI_API_KEY || '')
    : (process.env.OPENAI_API_KEY || process.env.LITELLM_PROXY_API_KEY || '');
  if (!apiKey) throw new Error('Missing OPENAI_API_KEY (or LITELLM_PROXY_API_KEY)');
  const client = new OpenAI({ apiKey, baseURL });

  const preview = (s: string, n = 200) => (s || '').slice(0, n).replace(/\s+/g, ' ');

  const start = Date.now();
  console.log(`[summarize] model=${model}`);
  if (focusTopic) {
    console.log(`[summarize] Focus Topic: "${focusTopic}"`);
  }
  console.log(`[summarize] system.head: ${preview(systemPrompt)}${systemPrompt.length > 200 ? '…' : ''}`);
  const userInput = buildUserContent(linesBlock, focusTopic);
  console.log(`[summarize] user.head: ${preview(userInput)}${userInput.length > 200 ? '…' : ''}`);

  const dim = '\x1b[2m';
  const reset = '\x1b[0m';
  const heartbeat = setInterval(() => {
    const sec = ((Date.now() - start) / 1000).toFixed(0);
    process.stdout.write(`${dim}[summarize] Waiting for model response... ${sec}s${reset}\r`);
  }, 5000);

  let response: OpenAI.Responses.Response;
  try {
    response = await client.responses.create({
      model,
      instructions: systemPrompt,
      input: userInput,
    });
  } finally {
    clearInterval(heartbeat);
    process.stdout.write('\r\x1b[K');
  }
  const elapsed = Date.now() - start;
  console.log(`[summarize] Responses API completed in ${(elapsed / 1000).toFixed(1)}s`);

  const text = extractOutputText(response);
  if (!text) throw new Error('Received empty response text');
  return { text, usage: extractUsage(response) };
}

function isoNoPunct(date: Date): string {
  return date.toISOString().replace(/[:.]/g, '').replace('Z', 'Z');
}

function formatMinSec(ms: number): string {
  if (!isFinite(ms) || ms < 0) return '0m 0s';
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}m ${seconds}s`;
}

async function pipeline(args: PipelineArgs) {
  const pipelineStart = Date.now();
  const outDir = args.out || path.resolve(process.cwd(), 'data');
  const model = args.model || (process.env.DEFAULT_MODEL || 'gpt-5.4');
  const systemPrompt = await readSystemPrompt(args.system, args.focusTopic);

  let scrape: ScrapeResult;
  let identifier: string;
  let scrapePath: string;

  // If jsonInput is provided, load from file instead of scraping
  if (args.jsonInput) {
    console.log(`[pipeline] Loading existing scrape from: ${args.jsonInput}`);

    if (!fs.existsSync(args.jsonInput)) {
      throw new Error(`JSON file not found: ${args.jsonInput}`);
    }

    const jsonContent = fs.readFileSync(args.jsonInput, 'utf8');
    const parsed = JSON.parse(jsonContent);

    // Validate that this is a scrape JSON file, not a summary JSON file
    if (!parsed.tweets || !Array.isArray(parsed.tweets)) {
      const red = '\x1b[31m';
      const yellow = '\x1b[33m';
      const reset = '\x1b[0m';

      console.error(`${red}[pipeline] ERROR: Invalid JSON file format${reset}`);
      console.error(`${yellow}[pipeline] Expected a SCRAPE JSON file (e.g., "list-*.json") but got a file without tweets.${reset}`);
      console.error(`${yellow}[pipeline] If this is a SUMMARY file (e.g., "summary-list-*.json"), you cannot use it with --json.${reset}`);
      console.error(`${yellow}[pipeline] Please use a scrape JSON file like: data/list-1585430245762441216-2026-02-18T044142021Z.json${reset}`);

      throw new Error('Invalid JSON format: missing "tweets" array. Please use a scrape JSON file, not a summary file.');
    }

    scrape = parsed as ScrapeResult;

    console.log(`[pipeline] Loaded ${scrape.tweets.length} tweets from JSON`);
    console.log(`[pipeline] Original scrape: ${scrape.diagnostics.pagesFetched} pages, ${scrape.diagnostics.tweetCount} tweets`);
    console.log(`[pipeline] Time range: ${scrape.diagnostics.startParam} to ${scrape.diagnostics.endParam}`);

    const fullListId = scrape.diagnostics.listId || path.basename(args.jsonInput).split('-')[1] || 'unknown';
    identifier = fullListId.includes(':') ? fullListId.split(':')[1] : fullListId;
    scrapePath = args.jsonInput;
  } else {
    const end = args.end ? new Date(args.end as any) : new Date();
    const start = args.start ? new Date(args.start as any) : new Date(end.getTime() - 24 * 60 * 60 * 1000);
    if (isNaN(start.getTime()) || isNaN(end.getTime())) throw new Error('Invalid start or end time');

    const baseUrl = args.baseUrl; // undefined → scrapeTarget uses nitter.miningtcup.me

    identifier = args.pageType === 'profile' ? (args.handle || '').replace(/^@/, '') : (args.listId || '');
    if (!identifier) throw new Error('Missing identifier for target');

    console.log(`[pipeline] ── Phase 1/3: Scraping ──`);
    console.log(`[pipeline] Target: ${args.pageType}:${identifier} via ${baseUrl || 'default (nitter.miningtcup.me)'}`);
    console.log(`[pipeline] Window: ${start.toISOString()} → ${end.toISOString()}`);

    try {
      scrape = await scrapeTarget({ pageType: args.pageType, identifier }, {
        start,
        end,
        baseUrl,
        outputDir: outDir,
        mockHtmlDir: args.mockHtmlDir,
        maxPages: args.maxPages ?? 100,
        maxChars: args.maxChars,
      });
    } catch (err: any) {
      console.error(`[pipeline] Scraping failed: ${err.message}`);
      throw err;
    }

    const label = args.pageType === 'list' ? `${identifier}` : `${args.pageType}-${identifier}`;
    scrapePath = await writeResultToJson(scrape, label, start, end, outDir);
    console.log(`[pipeline] Saved scrape JSON to: ${scrapePath}`);
  }

  if (scrape.diagnostics.wasPartial) {
    throw new Error(`Refusing to summarize incomplete coverage: ${scrape.diagnostics.errorMessage}. Resume the scrape first.`);
  }

  console.log(`[pipeline] ── Phase 2/3: Formatting ──`);
  const maxChars = args.maxChars ?? 200000;
  const { lines, block, truncated, originalCount } = formatTweetsForModel(scrape.tweets, maxChars);
  console.log(`[pipeline] Prepared ${lines.length} tweets for summarization (${(block.length / 1000).toFixed(1)}k chars)`);

  if (truncated) {
    throw new Error(`Input exceeds maxChars=${maxChars}; raise the limit to cover all ${originalCount} tweets. No summary was requested.`);
  }

  if (lines.length > 0) console.log(`[pipeline] sample tuple[0]: ${lines[0].slice(0, 200)}${lines[0].length > 200 ? '…' : ''}`);

  if (lines.length === 0) {
    const yellow = '\x1b[33m';
    const reset = '\x1b[0m';
    console.log(`${yellow}[pipeline][warn] No tweets in window; skipping summarization step.${reset}`);

    const ts = isoNoPunct(new Date());
    const baseName = `summary-${args.pageType}-${identifier}-${ts}`;
    const jsonPath = path.join(outDir, `${baseName}.json`);
    const mdPath = path.join(outDir, `${baseName}.md`);

    const emptySummary = {
      text: '(No tweets collected in the specified time window)',
      model: model,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };

    writeJson(jsonPath, emptySummary);
    console.log(`[pipeline] Wrote summary JSON (no-gen): ${jsonPath}`);

    const md = [
      `# Summary for ${args.pageType}:${identifier}`,
      ``,
      `**Time Range:** ${scrape.diagnostics.startParam} to ${scrape.diagnostics.endParam}`,
      `**Tweets Collected:** ${scrape.tweets.length}`,
      `**Scrape Status:** ${scrape.diagnostics.wasPartial ? 'Partial (rate limited)' : 'Complete'}`,
      ``,
      `No tweets were collected in the specified time window.`,
      ``,
    ].join('\n');
    fs.writeFileSync(mdPath, md, 'utf8');
    console.log(`\x1b[35m[pipeline] Wrote summary Markdown (no-gen): ${mdPath}\x1b[0m`);

    return;
  }

  console.log(`[pipeline] ── Phase 3/3: Summarizing with ${model} ──`);
  const genStart = Date.now();
  const summaryKey = fingerprint({ block, model, systemPrompt, focusTopic: args.focusTopic, baseURL: process.env.OPENAI_BASE_URL || process.env.LITELLM_BASE_URL, version: 1 });
  const summary = await checkpoint(path.join(outDir, '..', 'checkpoints', `summary-${summaryKey}.json`), () => summarizeTweets(block, model, systemPrompt, args.focusTopic));
  const genElapsedMs = Date.now() - genStart;
  console.log(`[pipeline] Summarize done in ${formatMinSec(genElapsedMs)}`);

  const driftMarkers = [
    'Main themes and topics discussed',
    'Notable patterns or trends',
    'Overall sentiment and tone',
    'Interesting insights and observations',
  ];
  const found = driftMarkers.filter((m) => summary.text.includes(m));
  if (found.length) {
    console.warn(`[pipeline][warn] Structure drift detected: headings found -> ${found.join(', ')}`);
  }
  const categoryLines = summary.text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => /^\*\*[^\*]+\*\*$/.test(l) && !l.startsWith('- '));
  console.log(`[pipeline] categoryLines=${categoryLines.length} (${categoryLines.slice(0, 3).join(' | ')})`);

  const ts = isoNoPunct(new Date());
  const sanitizedIdentifier = identifier.replace(/:/g, '-');
  const baseName = `summary-${args.pageType}-${sanitizedIdentifier}-${ts}`;
  const jsonOut = path.join(outDir, `${baseName}.json`);
  const mdOut = path.join(outDir, `${baseName}.md`);

  const payload = {
    inputs: lines,
    model,
    systemPromptPath: args.system || null,
    systemPromptSample: systemPrompt.slice(0, 160),
    diagnostics: scrape.diagnostics,
    output: summary.text,
    usage: summary.usage,
    generationDurationMs: genElapsedMs,
    timestamp: new Date().toISOString(),
    focusTopic: args.focusTopic || null,
  };
  writeJson(jsonOut, payload);

  const partialWarning = scrape.diagnostics.wasPartial
    ? `\n⚠️  **Partial Scrape**: Rate limited after ${scrape.diagnostics.pagesFetched} pages. ${scrape.diagnostics.errorMessage}\n`
    : '';

  const focusTopicSection = args.focusTopic ? `- **Focus Topic**: ${args.focusTopic}\n` : '';

  const md = `## Metadata\n- **Model**: ${model}\n- **Processed At**: ${payload.timestamp}\n- **Tweets**: ${lines.length}\n- **Pages Fetched**: ${scrape.diagnostics.pagesFetched}${scrape.diagnostics.wasPartial ? ' (partial)' : ''}\n- **Duration (scrape)**: ${formatMinSec(scrape.diagnostics.durationMs)}\n- **Duration (generate)**: ${formatMinSec(genElapsedMs)}\n- **Tokens (in/out/total)**: ${(summary.usage?.input ?? 'n/a')}/${(summary.usage?.output ?? 'n/a')}/${(summary.usage?.total ?? 'n/a')}\n${focusTopicSection}${partialWarning}\n# Summary\n\n${summary.text}\n`;
  fs.writeFileSync(mdOut, md, 'utf8');

  console.log(`[pipeline] Wrote summary JSON: ${jsonOut}`);
  console.log(`\x1b[32m[pipeline] ✓ Done! ${mdOut} (${formatMinSec(Date.now() - pipelineStart)} total)\x1b[0m`);
}

function looksLikeScrapeJsonPath(p: string): boolean {
  const base = path.basename(p.trim());
  return base.startsWith('list-') && base.endsWith('.json');
}

/** Strip POSIX-style `--` sentinels that leak through as their own argv token (e.g. `npm run x -- -- <id>`). If the first token is `--`, minimist treats the rest as unparsed positionals, so `--start` / `--baseUrl` never apply. */
function stripLeadingArgvSeparators(argv: string[]): string[] {
  const out = [...argv];
  while (out.length > 0 && out[0] === '--') {
    out.shift();
  }
  return out;
}

async function main() {
  const raw = stripLeadingArgvSeparators(process.argv.slice(2));
  const args = minimist(raw, {
    string: ['start', 'end', 'baseUrl', 'out', 'mockHtmlDir', 'model', 'system', 'type', 'json', 'maxChars', 'focusTopic'],
    default: {
      out: path.resolve(process.cwd(), 'data'),
      maxPages: '100',
      maxChars: '200000',
      type: '',
    },
  });

  if (args.json) {
    console.log('[main] JSON input mode detected');
    await pipeline({
      pageType: 'list',
      jsonInput: args.json,
      out: args.out,
      model: args.model,
      system: args.system,
      maxChars: parseInt(String(args.maxChars || '200000'), 10),
      focusTopic: args.focusTopic,
    });
    return;
  }

  // First positional `data/list-*.json` is often mistaken for replacing listId; without --json that becomes a bogus @profile and triggers scraping.
  const firstPositional = args._[0];
  if (typeof firstPositional === 'string' && looksLikeScrapeJsonPath(firstPositional)) {
    const jsonPath = path.isAbsolute(firstPositional)
      ? firstPositional
      : path.resolve(process.cwd(), firstPositional);
    const yellow = '\x1b[33m';
    const reset = '\x1b[0m';
    console.log(
      `${yellow}[main] Scrape JSON path detected as first argument — loading file (skipping scrape). Prefer explicit: --json ${firstPositional}${reset}`,
    );
    await pipeline({
      pageType: 'list',
      jsonInput: jsonPath,
      out: args.out,
      model: args.model,
      system: args.system,
      maxChars: parseInt(String(args.maxChars || '200000'), 10),
      focusTopic: args.focusTopic,
    });
    return;
  }

  const input = raw[0];
  if (!input) {
    console.error('Usage: ts-node src/processNitterListWithResponses.ts <listId|@handle> [OPTIONS]');
    console.error('       ts-node src/processNitterListWithResponses.ts --json <path-to-json> [OPTIONS]');
    console.error('');
    console.error('Options:');
    console.error('  --json FILE          Resume from existing scrape JSON file (skips scraping)');
    console.error('  --type TYPE          Page type: list or profile (auto-detected if omitted)');
    console.error('  --start ISO|REL      Start time (ISO date or relative like "3d")');
    console.error('  --end ISO|REL        End time (ISO date or relative)');
    console.error('  --baseUrl URL        Pin a single Nitter instance (default: nitter.miningtcup.me)');
    console.error('  --out DIR            Output directory (default: ./data)');
    console.error('  --mockHtmlDir DIR    Use mock HTML files instead of fetching');
    console.error('  --maxPages N         Max pages to fetch (default: 100)');
    console.error('  --maxChars N         Max input characters, truncates oldest tweets (default: 300000)');
    console.error('  --model MODEL        LLM model to use (default: gpt-5.2)');
    console.error('  --system FILE        Custom system prompt file');
    console.error('  --focusTopic TOPIC   Deep dive into a specific topic, gathering all relevant facts and opinions');
    process.exit(1);
  }

  let pageType = (args.type as string) || '';
  let listId: string | undefined;
  let handle: string | undefined;
  if (!pageType) {
    if (/^@/.test(input)) pageType = 'profile';
    else if (/^\d+$/.test(input)) pageType = 'list';
    else pageType = 'profile';
  }
  if (pageType === 'profile') handle = input.replace(/^@/, '');
  if (pageType === 'list') listId = input;

  const maxPages = parseInt(String(args.maxPages || '100'), 10);
  const maxChars = parseInt(String(args.maxChars || '300000'), 10);
  const { start, end } = resolveWindow(args.start as string | undefined, args.end as string | undefined);
  await pipeline({
    pageType,
    listId,
    handle,
    start,
    end,
    baseUrl: args.baseUrl,
    out: args.out,
    mockHtmlDir: args.mockHtmlDir,
    maxPages,
    model: args.model,
    system: args.system,
    maxChars,
    focusTopic: args.focusTopic,
  });
}

main().catch((err) => {
  console.error('[pipeline] Error:', err);
  process.exit(1);
});
