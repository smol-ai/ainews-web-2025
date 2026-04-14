import { defineCollection, z } from "astro:content";
import { glob } from 'astro/loaders';

const verboseBuild = process.env.VERBOSE_BUILD === "true";
const logBuild = (...args: unknown[]) => {
  if (verboseBuild) {
    console.log(...args);
  }
};

logBuild(`[Content Config] Environment: ${import.meta.env.MODE}`);
logBuild(`[Content Config] Initializing collections...`);

const issues = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: "./src/content/issues" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    companies: z.array(z.string()).optional(),
    models: z.array(z.string()).optional(),
    topics: z.array(z.string()).optional(),
    people: z.array(z.string()).optional(),
  }),
});

// Simpler logging for content config
logBuild(`[Content Config] Issues collection defined`);

const projects = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    demoURL: z.string().optional(),
    repoURL: z.string().optional(),
  }),
});

logBuild(`[Content Config] Projects collection defined`);

export const collections = { issues, projects };
