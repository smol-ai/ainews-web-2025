import type { z } from 'astro:content';

// Declare Astro loader types to match what's expected
export interface AstroLoaderContext {
  config: any;
  logger: any;
  watcher: any;
  parseData: any;
  store: any;
  generateDigest: any;
  entryTypes: Map<string, any>;
}

export interface AstroLoader {
  name: string;
  load: (context: AstroLoaderContext) => Promise<void>;
}

export interface AstroGlobOptions {
  pattern: string | Array<string>;
  base?: string | URL;
} 