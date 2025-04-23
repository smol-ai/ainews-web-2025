import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import yaml from 'js-yaml';
import { glob } from 'glob';
import PQueue from 'p-queue';
import { load } from 'js-yaml';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const INPUT_DIR = path.join(process.cwd(), 'buttondown-emails');
const OUTPUT_DIR = path.join(process.cwd(), 'processed-emails');
const MAX_CHARS = 5000;
const MAX_CONCURRENCY = 10;
const TEST_MODE = true;
const TEST_COUNT = 10;

interface EmailMetadata {
  id?: string;
  title?: string;
  date?: string;
  status?: string;
  type?: string;
  source?: string;
  metadata?: Record<string, any>;
  description?: string;
  tags?: string[];
  [key: string]: any;
}

async function processFile(filePath: string): Promise<void> {
  try {
    const content = await fs.promises.readFile(filePath, 'utf8');
    const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n/);
    
    if (!frontmatterMatch) {
      console.error(`No frontmatter found in ${filePath}`);
      return;
    }

    const frontmatter = load(frontmatterMatch[1]) as EmailMetadata;
    const body = content.slice(frontmatterMatch[0].length);
    const truncatedBody = body.slice(0, MAX_CHARS);

    // Clean up frontmatter
    delete frontmatter.source;
    delete frontmatter.metadata;
    if (frontmatter.status === 'sent') delete frontmatter.status;
    if (frontmatter.type === 'public') delete frontmatter.type;
    
    // Clean up title
    if (frontmatter.title) {
      frontmatter.title = frontmatter.title.replace(/^\[AINews\]\s*/, '');
    }

    // Generate description and tags
    const prompt = `Given this email content about AI news:
1. Write a 1-3 sentence summary focusing ONLY on the top 1-2 most important stories. Do not start with "Description:" or any other prefix.
2. Generate 3-7 tags, one per line starting with "-". Follow these rules for tags:
   - Split compound terms (e.g. "openai-gpt4" should be two tags: "openai" and "gpt4")
   - Remove redundant "ai" suffixes (e.g. use "coding" not "coding-ai")
   - Separate model names and company names (e.g. use "xai" and "grok-3" not "xai-grok-3")
   - Keep version numbers with model names (e.g. "gpt-4-1" not "gpt" "4" "1")
   - Use lowercase and hyphens for multi-word tags

Content:\n\n${truncatedBody}`;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that generates concise summaries and relevant tags for AI news. Focus on the most important stories and format tags consistently."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      temperature: 0.3,
    });

    const response = completion.choices[0].message.content;
    if (!response) {
      console.error(`No response from OpenAI for ${filePath}`);
      return;
    }

    const lines = response.split('\n');
    const description = lines[0].trim();
    const tags = lines
      .slice(1)
      .filter(line => line.trim().startsWith('-'))
      .map(line => line.trim().replace(/^-\s*/, '').trim())
      .filter(tag => tag.length > 0);

    if (tags.length === 0) {
      console.warn(`No tags generated for ${filePath}`);
    }

    // Update frontmatter
    frontmatter.description = description;
    frontmatter.tags = tags;

    // Create new content
    const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;

    // Write to new file
    const outputPath = path.join(OUTPUT_DIR, path.basename(filePath));
    await fs.promises.writeFile(outputPath, newContent, 'utf8');
    console.log(`Processed: ${path.basename(filePath)}`);
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
  }
}

async function main() {
  try {
    // Create output directory
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    // Get all markdown files
    const files = await glob('*.md', { cwd: INPUT_DIR });
    const filesToProcess = TEST_MODE ? files.slice(0, TEST_COUNT) : files;

    console.log(`Processing ${filesToProcess.length} files...`);

    // Create queue with concurrency limit
    const queue = new PQueue({ concurrency: MAX_CONCURRENCY });

    // Add all files to queue
    for (const file of filesToProcess) {
      queue.add(() => processFile(path.join(INPUT_DIR, file)));
    }

    // Wait for all files to be processed
    await queue.onIdle();
    console.log('Done!');
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main(); 