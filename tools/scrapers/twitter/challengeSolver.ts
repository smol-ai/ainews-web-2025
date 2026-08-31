import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'node:url';
import { chromium, type BrowserContext, type Page } from 'playwright';

const CHALLENGE_MARKERS = [
  'Verifying your request',
  'Verifying your browser',
  '/antibot/captcha',
  'Prove You&#39;re Human',
  "Prove You're Human",
  '/dogcaptcha/',
];
const CACHE_DIR = fileURLToPath(new URL('../.cache/', import.meta.url));
const USER_DATA_DIR = path.join(CACHE_DIR, 'chrome-profile');
const COOKIE_FILE = path.join(CACHE_DIR, 'xcancel-cookies.json');

let _context: BrowserContext | null = null;
let _page: Page | null = null;

export function isChallengePage(html: string): boolean {
  return CHALLENGE_MARKERS.some((m) => html.includes(m));
}

export function hasBrowserSession(): boolean {
  return _context !== null;
}

/**
 * Ensure a persistent browser session is running. Reuses the same browser
 * across all page fetches once activated, and persists the profile (cookies,
 * solved anti-bot sessions) to .cache/chrome-profile across runs.
 */
async function ensureBrowser(debug: boolean): Promise<Page> {
  if (_page && _context) return _page;

  if (debug) console.log(`[challenge] Launching system Chrome for anti-bot bypass...`);

  fs.mkdirSync(USER_DATA_DIR, { recursive: true });
  _context = await chromium.launchPersistentContext(USER_DATA_DIR, {
    channel: 'chrome',
    headless: false,
    viewport: { width: 1280, height: 800 },
    screen: { width: 1440, height: 900 },
    locale: 'en-US',
    timezoneId: Intl.DateTimeFormat().resolvedOptions().timeZone,
    // Drop Playwright's --enable-automation flag: it sets navigator.webdriver and the
    // "controlled by automated test software" infobar, which anti-bot checks detect.
    ignoreDefaultArgs: ['--enable-automation'],
    args: [
      '--disable-blink-features=AutomationControlled',
      '--window-size=1280,800',
      '--window-position=50,50',
    ],
  });

  await _context.addInitScript(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  });

  // Seed previously solved anti-bot cookies (e.g. from scripts/solveCaptcha.ts)
  if (fs.existsSync(COOKIE_FILE)) {
    try {
      const cookies = JSON.parse(fs.readFileSync(COOKIE_FILE, 'utf8'));
      await _context.addCookies(cookies.map((c: any) => ({ ...c, url: undefined, domain: c.domain, path: c.path })));
      if (debug) console.log(`[challenge] Seeded ${cookies.length} cookies from ${COOKIE_FILE}`);
    } catch {}
  }

  _page = _context.pages()[0] || (await _context.newPage());
  return _page;
}

/**
 * Fetch a page using the persistent browser session.
 * Handles the anti-bot challenge transparently by waiting for it to resolve.
 */
export async function fetchWithBrowser(url: string, debug = false): Promise<string> {
  const page = await ensureBrowser(debug);

  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });

  const initialHtml = await page.content();
  if (!isChallengePage(initialHtml)) {
    return initialHtml;
  }

  if (debug) console.log(`[challenge] Anti-bot challenge detected, waiting for auto-resolve...`);

  try {
    await page.waitForSelector('.timeline-item', { timeout: 30000 });
    if (debug) console.log(`[challenge] Challenge solved — timeline loaded`);
    return await page.content();
  } catch {}

  const yellow = '\x1b[33m';
  const reset = '\x1b[0m';

  // "Automated verification failed" — retry the JS check once, then fall through to the image captcha.
  if ((await page.content()).includes('Automated verification failed')) {
    const reload = page.getByText('Reload now', { exact: false });
    if (await reload.count()) {
      console.log(`${yellow}[challenge] Automated verification failed — retrying once...${reset}`);
      await reload.first().click().catch(() => {});
      const solved = await page.waitForSelector('.timeline-item', { timeout: 20000 }).then(() => true).catch(() => false);
      if (solved) {
        await saveCookies(page.context());
        return await page.content();
      }
    }
    const toCaptcha = page.getByText('solve an image challenge', { exact: false });
    if (await toCaptcha.count()) {
      await toCaptcha.first().click().catch(() => {});
      await page.waitForLoadState('domcontentloaded').catch(() => {});
    }
  }

  // Auto-resolve failed — likely an image captcha. Ask the human to solve it in the visible Chrome window.
  console.log(`${yellow}[challenge] CAPTCHA required — please solve it in the Chrome window (waiting up to 3 min)...${reset}`);
  try {
    await page.waitForFunction(
      () => {
        const t = document.body.textContent || '';
        return !location.pathname.startsWith('/antibot')
          && !location.pathname.startsWith('/dogcaptcha')
          && !t.includes('Verifying your request')
          && !t.includes('Verifying your browser')
          && !t.includes("Prove You're Human");
      },
      undefined,
      { timeout: 180000, polling: 2000 },
    );
    console.log(`${yellow}[challenge] Challenge cleared — continuing scrape${reset}`);
    await saveCookies(page.context());
  } catch {
    console.log(`${yellow}[challenge] Timed out waiting for challenge resolution${reset}`);
  }

  return await page.content();
}

async function saveCookies(context: BrowserContext): Promise<void> {
  try {
    const cookies = (await context.cookies()).filter((c) => c.name.includes('antibot'));
    if (cookies.length === 0) return;
    fs.mkdirSync(path.dirname(COOKIE_FILE), { recursive: true });
    fs.writeFileSync(COOKIE_FILE, JSON.stringify(cookies, null, 2));
  } catch {}
}

/**
 * Close the persistent browser session. Call at the end of a scrape run.
 */
export async function closeBrowser(): Promise<void> {
  if (_context) {
    await saveCookies(_context);
    try { await _context.close(); } catch {}
  }
  _page = null;
  _context = null;
}
