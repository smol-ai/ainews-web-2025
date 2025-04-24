import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";
import pagefind from "astro-pagefind";
import tailwindcss from "@tailwindcss/vite";

console.log(`[Astro Config] Environment: ${process.env.NODE_ENV || 'development'}`);
console.log(`[Astro Config] Verbose build: ${process.env.VERBOSE_BUILD || 'false'}`);

// https://astro.build/config
export default defineConfig({
  site: "https://astro-micro.vercel.app",
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
  },
  markdown: {
    shikiConfig: {
      theme: "css-variables",
    },
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
});
