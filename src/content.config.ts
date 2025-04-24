import { defineCollection, z } from "astro:content";
import { glob } from 'astro/loaders';

console.log(`[Content Config] Environment: ${import.meta.env.MODE}`);
console.log(`[Content Config] Initializing collections...`);

const issues = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: "./src/content/issues" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    draft: z.boolean().optional(),
    tags: z.array(z.string()).optional(),
  }),
});

// Simpler logging for content config
console.log(`[Content Config] Issues collection defined`);

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

console.log(`[Content Config] Projects collection defined`);

export const collections = { issues, projects };
