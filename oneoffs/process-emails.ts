import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import yaml from 'js-yaml';
import { glob } from 'glob';
import PQueue from 'p-queue';
import { load } from 'js-yaml';
import Instructor from '@instructor-ai/instructor';
import { z } from 'zod';

// Regular OpenAI client for non-instructor calls
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Instructor-enhanced client for structured extraction
const instructorClient = Instructor({
  client: openai,
  mode: "FUNCTIONS", // Use function calling
});

const INPUT_DIR = path.join(process.cwd(), 'buttondown-emails');
const OUTPUT_DIR = path.join(process.cwd(), 'processed-emails');
const MAX_CHARS = 5000;
const MAX_CONCURRENCY = 10;
const TEST_MODE = false;
const TEST_COUNT = 10;

// Define schemas for our tag extraction
const CompanySchema = z.string().describe('A company or organization name mentioned in the content');
const ModelSchema = z.string().describe('A specific AI model name, including version numbers if applicable');
const TopicSchema = z.string().describe('A general topic, research area, concept, technology or domain discussed');

const TagsResponseSchema = z.object({
  description: z.string().describe('A concise 1-3 sentence summary focusing on the most important stories'),
  companies: z.array(CompanySchema).describe('List of companies or organizations mentioned. Normalize company names (e.g., "OpenAI" not "open-ai"). NOT twitter handles.'),
  models: z.array(ModelSchema).describe('List of AI model names mentioned, with proper versioning (e.g., "gpt-4", "claude-3-opus", "gemini-1.5-pro")'),
  topics: z.array(TopicSchema).describe('List of topics, research areas, or concepts, using lowercase with hyphens for multi-word terms (e.g., "reinforcement-learning", "multimodal", "text-to-image")'),
});

type TagsResponse = z.infer<typeof TagsResponseSchema>;

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
  companies?: string[];
  models?: string[];
  topics?: string[];
  [key: string]: any;
}

interface ProcessFileResult {
  description: string;
  companies: string[];
  models: string[];
  topics: string[];
  allTags: string[];
  content: string;
}

async function processFile(filePath: string, cliMode = false): Promise<ProcessFileResult | void> {
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

    console.log(`Extracting tags for ${path.basename(filePath)}...`);
    
    try {
      // Use Instructor for structured extraction
      const extractionResult = await instructorClient.chat.completions.create({
        model: "gpt-4.1-nano", // Using the same model as before
        response_model: { 
          schema: TagsResponseSchema,
          name: "NewsTagsExtraction"
        },
        messages: [
          {
            role: "system",
            content: `You are an expert AI news analyst and tagger. Extract a concise description and categorized tags from AI news content. 
            
            Follow these rules for tagging:
            1. Keep tags concise, clear, and normalized (e.g. "openai" not "OpenAI" or "open-ai")
            2. For models, use the exact model name with hyphens (e.g. "gpt-4", "claude-3-opus", "gemini-pro-1.5")
            3. Use lowercase with hyphens for multi-word terms
            4. Company names should be the official company name in lowercase (e.g., "google", "openai", "anthropic", "xai" - not "x"). IGNORE twitter handles.
            5. Avoid redundant tags or overtagging
            6. IMPORTANT: Be comprehensive - don't miss important companies and models mentioned
            7. Output between 5-15 tags in each category for normal content`
          },
          {
            role: "user",
            content: `Extract tags from this AI news content:\n\n${truncatedBody}`
          }
        ],
        temperature: 0.1, // Lower temperature for more consistent results
      });

      // Use the structured extraction results
      const { description, companies, models, topics } = extractionResult;
      
      // Combine all tags and make unique
      const allTags = [...new Set([...companies, ...models, ...topics])];

      if (allTags.length === 0) {
        console.warn(`No tags generated for ${filePath} - unusual with structured extraction`);
        // If we still get no tags (highly unlikely), add a default
        return {
          description,
          companies: ["untagged"],
          models: [],
          topics: ["untagged"],
          allTags: ["untagged"],
          content: ""
        };
      }

      console.log(`\nExtracted ${companies.length} companies, ${models.length} models, ${topics.length} topics for ${path.basename(filePath)}`);
      console.log(`Companies: ${companies.join(', ')}`);
      console.log(`Models: ${models.join(', ')}`);
      console.log(`Topics: ${topics.join(', ')}`);

      // Update frontmatter
      frontmatter.description = description;
      // frontmatter.tags = allTags;
      frontmatter.companies = companies;
      frontmatter.models = models;
      frontmatter.topics = topics;

      // Create new content
      const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;

      if (cliMode) {
        // In CLI mode, return the result without writing to file
        return {
          description,
          companies,
          models,
          topics,
          allTags,
          content: newContent
        };
      } else {
        // Write to new file
        const outputPath = path.join(OUTPUT_DIR, path.basename(filePath));
        await fs.promises.writeFile(outputPath, newContent, 'utf8');
        console.log(`Processed: ${path.basename(filePath)}`);
        return;
      }
    } catch (extractionError) {
      console.error(`Error during tag extraction for ${filePath}:`, extractionError);
      
      // Fallback to old method if instructor extraction fails
      console.log(`Falling back to regular OpenAI completion for ${path.basename(filePath)}`);
      
      // Generate description and categorized tags using standard completion
      const prompt = `Given this email content about AI news:
1. Write a 1-3 sentence summary focusing ONLY on the top 1-2 most important stories. Do not start with "Description:" or any other prefix.

2. Generate specific tag categories. For each category, provide 5-15 relevant tags, one per line with the category heading:

COMPANIES:
- [company names mentioned, normalized to lowercase]

MODELS:
- [specific AI model names mentioned, including version numbers]

TOPICS:
- [general topics, research areas, or domains discussed]

Follow these rules for all tags:
- Use lowercase throughout
- Split compound terms (e.g. "openai-gpt4" should be "openai" under COMPANIES and "gpt4" under MODELS)
- Remove redundant "ai" suffixes (e.g. use "coding" not "coding-ai")
- Keep version numbers with model names (e.g. "gpt-4-1" not "gpt" "4" "1")
- Use lowercase and hyphens for multi-word tags
- Be specific and precise when identifying companies and models
- Include research areas, applications, and general concepts in TOPICS

Content:\n\n${truncatedBody}`;
      const QuestionAnswer = z.object({
        description: z.string().describe("A 1-3 sentence summary focusing ONLY on the top 1-2 most important stories."),
        companies: z.array(z.string()).describe("Company names mentioned, normalized to lowercase"),
        models: z.array(z.string()).describe("Specific AI model names mentioned, including version numbers"),
        topics: z.array(z.string()).describe("General topics, research areas, or domains discussed")
      });

      try {
        const result = await instructorClient.chat.completions.create({
          messages: [
            {
              role: "system",
              content: "You are a helpful assistant that generates concise summaries and categorized tags for AI news. Focus on the most important stories and format tags consistently into specific categories for companies, models, and topics."
            },
            {
              role: "user",
              content: `Given this email content about AI news:
${truncatedBody}

1. Write a 1-3 sentence summary focusing ONLY on the top 1-2 most important stories.

2. Generate specific tag categories. For each category, provide 5-15 relevant tags.

Follow these rules for all tags:
- Use lowercase throughout
- Split compound terms (e.g. "openai-gpt4" should be "openai" under COMPANIES and "gpt4" under MODELS)
- Remove redundant "ai" suffixes (e.g. use "coding" not "coding-ai")
- Keep version numbers with model names (e.g. "gpt-4-1" not "gpt" "4" "1")
- Use lowercase and hyphens for multi-word tags
- Be specific and precise when identifying companies and models
- Include research areas, applications, and general concepts in TOPICS`
            }
          ],
          model: "gpt-4-1106-preview",
          response_model: { schema: QuestionAnswer, name: "AI News Summary and Tags" },
          max_tokens: 1000,
          temperature: 0.3,
        });

        const { description, companies, models, topics } = result;
        const allTags = [...new Set([...companies, ...models, ...topics])];

        console.log(`Generated tags for ${path.basename(filePath)}:`);
        console.log(`Companies (${companies.length}): ${companies.join(', ')}`);
        console.log(`Models (${models.length}): ${models.join(', ')}`);
        console.log(`Topics (${topics.length}): ${topics.join(', ')}`);
        console.log(`Total tags: ${allTags.length}`);

        return { description, companies, models, topics, allTags };

      } catch (error) {
        console.error(`Error during tag extraction for ${filePath}:`, error);
        
        // Fallback to adding a default "untagged" topic
        console.warn(`No tags generated for ${filePath}`);
        const description = "Unable to generate summary due to an error.";
        const companies: string[] = [];
        const models: string[] = [];
        const topics = ["untagged"];
        const allTags = ["untagged"];
        
        return { description, companies, models, topics, allTags };
      }

      // Update frontmatter
      frontmatter.description = description;
      // frontmatter.tags = allTags;
      frontmatter.companies = companies;
      frontmatter.models = models;
      frontmatter.topics = topics;

      // Create new content
      const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;

      if (cliMode) {
        // In CLI mode, return the result without writing to file
        return {
          description,
          companies,
          models,
          topics,
          allTags,
          content: newContent
        };
      } else {
        // Write to new file
        const outputPath = path.join(OUTPUT_DIR, path.basename(filePath));
        await fs.promises.writeFile(outputPath, newContent, 'utf8');
        console.log(`Processed: ${path.basename(filePath)}`);
        return;
      }
    }
  } catch (error) {
    console.error(`Error processing ${filePath}:`, error);
  }
}

async function main() {
  try {
    // Check if a single file is specified via command line
    const args = process.argv.slice(2);
    if (args.length > 0 && args[0] === '--file') {
      if (!args[1]) {
        console.error('Error: No file path provided. Usage: pnpm ts-node oneoffs/process-emails.ts --file <filepath>');
        process.exit(1);
      }
      
      const filePath = args[1];
      if (!fs.existsSync(filePath)) {
        console.error(`Error: File not found: ${filePath}`);
        process.exit(1);
      }
      
      console.log(`Processing single file: ${filePath}`);
      const result = await processFile(filePath, true);
      
      if (result) {
        console.log('\n=== PROCESSING RESULTS ===');
        console.log('\nDescription:');
        console.log(result.description);
        
        console.log('\nCompanies:');
        result.companies.forEach(tag => console.log(`- ${tag}`));
        
        console.log('\nModels:');
        result.models.forEach(tag => console.log(`- ${tag}`));
        
        console.log('\nTopics:');
        result.topics.forEach(tag => console.log(`- ${tag}`));
        
        console.log('\nTotal tags:', result.allTags.length);
      }
      
      return;
    }

    // Batch processing mode
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