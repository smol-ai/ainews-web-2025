import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import yaml from 'js-yaml';
import { glob } from 'glob';
import PQueue from 'p-queue';
import { load } from 'js-yaml';
import Instructor from '@instructor-ai/instructor';
import { z } from 'zod';
import { prefCompanies, prefModels, prefTopics, nonTopics, prefPeople } from './preferredTags.ts';

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
// const TEST_MODE = true;
// const TEST_COUNT = 10;
const TEST_COUNT = MAX_CONCURRENCY;

const systemPrompt = `You are an expert AI news analyst and tagger. Extract a concise description and categorized tags from AI news content. 

For Description: Use **bold** for important companies, models, topics, and numbers/facts, and *italics* for direct quotes from the content. Ignore the "> AI News for ..." header.

Follow these rules for tagging:
1. COMPANIES:
    - Use lowercase normalized company names (e.g., "openai", "google-deepmind", "meta-ai-fair")
    - Use hyphens for multi-word companies (e.g., "hugging-face", "mistral-ai")
    - Favor specific subsidiaries over parent companies when appropriate (e.g., "google-deepmind" not just "google")
    - GOOD EXAMPLES: "openai", "anthropic", "google-deepmind", "hugging-face", "mistral-ai"
    - BAD EXAMPLES: "OpenAI", "Google", "Anthropic", "llama-AI"
    
2. MODELS:
    - Use exact model names with proper versioning using hyphens (e.g., "gpt-4", "claude-3-opus", "gemini-1.5-pro")
    - Include size parameters when mentioned (e.g., "llama-3-70b" not just "llama-3")
    - If the model is a family of models, tag the family name eg "claude-3-opus" also has "claude-3" tag, "llama-3-70b" also has "llama-3" tag.
    - GOOD EXAMPLES: "gpt-4o", "claude-3-sonnet", "llama-3-70b", "mistral-7b", "gemini-1.5-pro"
    - BAD EXAMPLES: "GPT4", "Claude 3", "LLaMA 3", "Mistral"
    
3. TOPICS:
    - Focus on specific technical concepts rather than generic "ai-something" terms
    - Use lowercase with hyphens for multi-word terms
    - Avoid redundant topics like "ai-research", "llm", "ai-models" - prefer more specific versions
    - GOOD EXAMPLES: "fine-tuning", "multimodality", "reinforcement-learning", "quantization", "benchmarking"
    - BAD EXAMPLES: "ai-research", "large-language-models", "ai-models", "llm", "ai-ethics"
    - Here are TOPICS TO AVOID (use more specific versions instead): ${nonTopics.join(', ')}
    
4. PEOPLE:
    - Tag prominent AI researchers, company leaders, and influential figures in the field
    - Try to tag them using their Twitter handle - if not known, use lowercase standardized names with underscores if needed
    - Focus on people who made significant contributions or statements in the content
    - GOOD EXAMPLES: "sama", "demishassabis", "ylecun", "karpathy"
    - BAD EXAMPLES: "Sam Altman", "sam-altman", "demis-hassabis", "yann-lecun", "andrej-karpathy"
    
5. General Guidelines:
    - Be comprehensive - don't miss important companies, models, topics, and people mentioned
    - Output between 5-15 tags in each category for normal content
    - Avoid redundant tags or overtagging
`

// Define schemas for our tag extraction
const CompanySchema = z.string().describe('A company or organization name mentioned in the content');
const ModelSchema = z.string().describe('A specific AI model name, including version numbers if applicable');
const TopicSchema = z.string().describe('A general topic, research area, concept, technology or domain discussed');
const PeopleSchema = z.string().describe('A person mentioned in the content, typically AI researchers, leaders, or prominent figure - their twitter/x handle preferred if known');

// Unified schema for news tag extraction
const NewsTagsSchema = z.object({
  description: z.string().describe('A concise 1-3 sentence summary focusing on the most important stories. Use **bold** for important companies, models, topics, numbers, and numbers/facts, and *italics* for direct quotes from the content. Ignore the "> AI News for ..." header.'),
  companies: z.array(CompanySchema).describe(`List of companies or organizations mentioned. Use official lowercase names with hyphens for multi-word names (e.g., "openai", "google-deepmind", "x-ai"). Meta AI has a few names - use "meta-ai-fair" not "meta-ai" or "facebook-ai-research". Use "x-ai" instead of "xai", "deepseek" instead of "deepseek-ai"..
    
To improve tag density, try to prefer this list of company tags rather than making up new ones as long as they are appropriate: ${prefCompanies.join(', ')}
    `),
  models: z.array(ModelSchema).describe(`List of AI model names mentioned, with proper versioning using hyphens (e.g., "gpt-4", "claude-3-opus", "gemini-1.5-pro", "llama-3-70b"). If there is a specific sub model, also tag the model family name eg "claude-3-opus" also has "claude-3" tag, "llama-3-70b" also has "llama-3" tag.
    
To improve tag density, try to prefer this list of model tags rather than making up new ones as long as they are appropriate: ${prefModels.join(', ')}
    `),
  topics: z.array(TopicSchema).describe(`List of specific technical topics and concepts, using lowercase with hyphens for multi-word terms (e.g., "reinforcement-learning", "multimodality", "fine-tuning"). If a topic is "vision" related just tag "vision", not "vision-language-model" or "vision-model" - only use the "multimodality" tag if it is more than just vision or audio with language models.
    
To improve tag density, try to prefer this list of topic tags rather than making up new ones as long as they are appropriate: ${prefTopics.join(', ')}
    `),
  people: z.array(PeopleSchema).describe(`List of people mentioned in the content, typically AI researchers, company leaders, or prominent figures in the AI space. ONLY include the people that made the news with signifcant updates about them/from them, not the people REPORTING news. Use standardized lowercase names with underscores if needed (e.g. "brett_adcock", "lmarena_ai", "reach_vb").
    
To improve tag density, try to prefer this list of people tags rather than making up new ones as long as they are appropriate: ${prefPeople.join(', ')}
    `),
});

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
  people?: string[];
  [key: string]: any;
}

interface ProcessFileResult {
  description: string;
  companies: string[];
  models: string[];
  topics: string[];
  people: string[];
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
        model: "gpt-4.1-mini", // Using the same model as before
        response_model: { 
          schema: NewsTagsSchema,
          name: "NewsTagsExtraction"
        },
        messages: [
          {
            role: "system",
            content: systemPrompt
          },
          {
            role: "user",
            content: `Extract tags from this AI news content:\n\n${truncatedBody}`
          }
        ],
        temperature: 0.1, // Lower temperature for more consistent results
      });

      // Use the structured extraction results
      const { description, companies, models, topics, people } = extractionResult;


      console.log(`\nExtracted ${companies.length} companies, ${models.length} models, ${topics.length} topics, ${people.length} people for ${path.basename(filePath)}`);
      console.log(`Companies: ${companies.join(', ')}`);
      console.log(`Models: ${models.join(', ')}`);
      console.log(`Topics: ${topics.join(', ')}`);
      console.log(`People: ${people.join(', ')}`);

      // Update frontmatter
      frontmatter.description = description;
      frontmatter.companies = companies;
      frontmatter.models = models;
      frontmatter.topics = topics;
      frontmatter.people = people;

      // Create new content
      const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;

      if (cliMode) {
        // In CLI mode, return the result without writing to file
        return {
          description,
          companies,
          models,
          topics,
          people,
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
      
      try {
        const result = await instructorClient.chat.completions.create({
          messages: [
            {
              role: "system",
              content: systemPrompt
            },
            {
              role: "user",
              content: `Given this email content about AI news:
${truncatedBody}

1. Write a 1-3 sentence summary focusing ONLY on the top 1-2 most important stories.

2. Generate specific tag categories. For each category, provide 5-15 relevant tags.

Follow these rules for all tags:
- Use lowercase throughout
- Split compound terms (e.g. "openai-gpt4" should be "openai" under COMPANIES and "gpt-4" under MODELS)
- Remove redundant "ai" suffixes (e.g. use "coding" not "coding-ai")
- Keep version numbers with model names (e.g. "gpt-4-1" not "gpt" "4" "1")
- Use lowercase and hyphens for multi-word tags
- Be specific and precise when identifying companies and models
- Include research areas, applications, and general concepts in TOPICS
`
            }
          ],
          model: "gpt-4.1-mini", // Using the same model as before
          response_model: { schema: NewsTagsSchema, name: "AI News Summary and Tags" },
          max_tokens: 1000,
          temperature: 0.3,
        });
        
        const { description, companies, models, topics, people = [] } = result;

        console.log(`Generated tags for ${path.basename(filePath)}:`);
        console.log(`Companies (${companies.length}): ${companies.join(', ')}`);
        console.log(`Models (${models.length}): ${models.join(', ')}`);
        console.log(`Topics (${topics.length}): ${topics.join(', ')}`);
        console.log(`People (${people.length}): ${people.join(', ')}`);

        // Update frontmatter
        frontmatter.description = description;
        frontmatter.companies = companies;
        frontmatter.models = models;
        frontmatter.topics = topics;
        frontmatter.people = people || [];

        // Create new content
        const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;

        if (cliMode) {
          // In CLI mode, return the result without writing to file
          return {
            description,
            companies,
            models,
            topics,
            people,
            content: newContent
          };
        } else {
          // Write to new file
          const outputPath = path.join(OUTPUT_DIR, path.basename(filePath));
          await fs.promises.writeFile(outputPath, newContent, 'utf8');
          console.log(`Processed: ${path.basename(filePath)}`);
          return;
        }
      } catch (error) {
        console.error(`Error during tag extraction for ${filePath}:`, error);
        
        // Fallback to adding a default "untagged" topic
        console.warn(`No tags generated for ${filePath}`);
        const description = "Unable to generate summary due to an error.";
        const companies: string[] = [];
        const models: string[] = [];
        const topics = ["untagged"];
        const people: string[] = [];
        
        // Update frontmatter
        frontmatter.description = description;
        frontmatter.companies = companies;
        frontmatter.models = models;
        frontmatter.topics = topics;
        frontmatter.people = people;

        // Create new content
        const newContent = `---\n${yaml.dump(frontmatter)}---\n\n${body}`;
        
        if (cliMode) {
          // In CLI mode, return the result without writing to file
          return {
            description,
            companies,
            models,
            topics,
            people,
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
        console.log('\nCompanies:', result.companies.join(', '));
        console.log('\nModels:', result.models.join(', '));
        console.log('\nTopics:', result.topics.join(', '));
        console.log('\nPeople:', result.people.join(', '));
        
        // Write the updated content back to the original file
        await fs.promises.writeFile(filePath, result.content, 'utf8');
        console.log(`\nUpdated file in place: ${filePath}`);
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