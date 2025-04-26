import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";
import pagefind from "astro-pagefind";
import tailwindcss from "@tailwindcss/vite";
import remarkYouTubeEmbed from "./src/remark-youtube-embed.mjs";

import vercel from "@astrojs/vercel";

console.log(`[Astro Config] Environment: ${process.env.NODE_ENV || 'development'}`);
console.log(`[Astro Config] Verbose build: ${process.env.VERBOSE_BUILD || 'false'}`);

// https://astro.build/config
export default defineConfig({
  site: "https://news.smol.ai",

  integrations: [
    sitemap(), 
    mdx(),
    pagefind(),
    // Custom logger integration to track content loading
    {
      name: 'content-logger',
      hooks: {
        'astro:config:setup': ({ config }) => {
          console.log(`[Astro Config] Site URL: ${config.site}`);
        },
        'astro:build:start': () => {
          console.log(`[Astro Build] Starting build process...`);
        },
        'astro:build:done': ({ pages }) => {
          console.log(`[Astro Build] Built ${pages.length} pages`);
          
          // Count and log pages by year
          const issuesPages = pages.filter(page => page.pathname.includes('/issues/'));
          console.log(`[Astro Build] Built ${issuesPages.length} issue pages`);
          
          // Count issues with 2025 in the URL
          const issues2025 = issuesPages.filter(page => page.pathname.includes('/25-'));
          console.log(`[Astro Build] Built ${issues2025.length} pages for 2025 issues`);
          
          if (issues2025.length > 0) {
            console.log(`[Astro Build] Sample 2025 pages:`, issues2025.slice(0, 5).map(p => p.pathname));
          } else {
            console.log(`[Astro Build] WARNING: No 2025 issue pages were built!`);
          }
          
          // Check for recent content (within last 3 weeks)
          const threeWeeksAgo = new Date();
          threeWeeksAgo.setDate(threeWeeksAgo.getDate() - 21);
          
          // Extract date patterns from pathnames
          const recentPages = pages.filter(page => {
            // Match date patterns like YY-MM-DD (e.g., 24-05-14)
            const dateMatch = page.pathname.match(/\/(\d{2})-(\d{2})-(\d{2})-/);
            if (!dateMatch) return false;
            
            // Extract year, month, day from URL pattern
            const [_, year, month, day] = dateMatch;
            const fullYear = parseInt(year) < 50 ? 2000 + parseInt(year) : 1900 + parseInt(year);
            const pageDate = new Date(fullYear, parseInt(month) - 1, parseInt(day));
            
            return pageDate >= threeWeeksAgo;
          });
          
          console.log(`[Astro Build] Found ${recentPages.length} pages with content from the last 3 weeks`);
          if (recentPages.length > 0) {
            console.log(`[Astro Build] Recent content examples:`, recentPages.slice(0, 3).map(p => p.pathname));
          }
          
          // Skip the check in development mode
          if (process.env.NODE_ENV !== 'production') {
            console.log(`[Astro Build] Skipping recent content check in development mode`);
            return;
          }
          
          // Fail build if no recent content and not explicitly bypassed
          if (recentPages.length === 0 && process.env.BYPASS_RECENT_CONTENT_CHECK !== 'true') {
            console.error('[Astro Build] ERROR: No content from the last 3 weeks detected!');
            console.error('[Astro Build] Set BYPASS_RECENT_CONTENT_CHECK=true to bypass this check');
            process.exit(1); // This will fail the build
          }
        }
      }
    }
  ],

  vite: {
    plugins: [tailwindcss()],
    build: {
      rollupOptions: {
        onLog(level, log, handler) {
          // Log content-related warnings
          if (log.message && log.message.includes('content')) {
            console.log(`[Vite Build] ${log.message}`);
          }
          // Let Rollup handle the log
          handler(level, log);
        },
      },
    },
    ssr: {
      // Ensure @astro-community packages are processed correctly by Vite
      noExternal: [/@astro-community\//]
    }
  },

  markdown: {
    shikiConfig: {
      theme: "css-variables",
    },
    remarkPlugins: [remarkYouTubeEmbed],
  },

  pagefind: {
    uiOptions: {
      showImages: false,
      excerptLength: 15,
      resetStyles: false,
      showMeta: ["date"],
      showFilters: ["tag"]
    },
  },

  adapter: vercel(),
});