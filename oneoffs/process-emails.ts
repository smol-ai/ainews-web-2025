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
const MAX_CHARS = 10000;
const MAX_CONCURRENCY = 10;
const TEST_MODE = false;
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

    // Generate description and categorized tags
    const prompt = `Given this email content about AI news:
1. Write a 1-3 sentence summary focusing ONLY on the top 1-2 most important stories. Do not start with "Description:" or any other prefix. **Bold** the most important names and companies, and *italicize* key numbers, dates, and other facts..

2. Generate specific tag categories classifying the content. For each category, provide 3-10 relevant tags, one per line with the category heading:

COMPANIES:
- [company names mentioned in the content]

MODELS:
- [specific AI model names mentioned, including version numbers]

TOPICS:
- [general topics, research areas, or domains discussed]

Follow these rules for all tags:
- Split compound terms (e.g. "openai-gpt4" should be "openai" under COMPANIES and "gpt4" under MODELS)
- Remove redundant "ai" suffixes (e.g. use "coding" not "coding-ai")
- Keep version numbers with model names (e.g. "gpt-4.1" not "gpt" "4" "1")
- Use lowercase and hyphens for multi-word tags
- Be specific and precise when identifying companies and models
- Include research areas, applications, and general concepts in TOPICS
- If it was a "quiet day" you can also tag "quiet-day" under TOPICS

Content:\n\n${truncatedBody}`;
    
    const completion = await openai.chat.completions.create({
      model: "gpt-4.1-mini",
      messages: [
        {
          role: "system",
          content: "You are a helpful assistant that generates concise summaries and categorized tags for AI news. Focus on the most important stories and format tags consistently into specific categories for companies, models, and topics. DO NOT end without filling out at least one tag for each category."
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

    // console.log(`\n=== DEBUG: Raw API Response for ${path.basename(filePath)} ===`);
    // console.log(response);
    
    // Parse categorized tags
    const lines = response.split('\n');
    const description = lines[0].trim();
    
    let companies: string[] = [];
    let models: string[] = [];
    let topics: string[] = [];
    let currentCategory: 'companies' | 'models' | 'topics' | null = null;
    let foundCategories = false;

    for (const line of lines.slice(1)) {
      const trimmedLine = line.trim();
      
      if (trimmedLine.toUpperCase() === 'COMPANIES:') {
        currentCategory = 'companies';
        foundCategories = true;
      } else if (trimmedLine.toUpperCase() === 'MODELS:') {
        currentCategory = 'models';
        foundCategories = true;
      } else if (trimmedLine.toUpperCase() === 'TOPICS:') {
        currentCategory = 'topics';
        foundCategories = true;
      } else if (trimmedLine.startsWith('-') && currentCategory) {
        const tag = trimmedLine.replace(/^-\s*/, '').trim();
        if (tag.length > 0) {
          switch (currentCategory) {
            case 'companies': companies.push(tag); break;
            case 'models': models.push(tag); break;
            case 'topics': topics.push(tag); break;
          }
        }
      }
    }

    // Fall back to basic tag parsing if no categories were found
    if (!foundCategories) {
      console.log(`\n=== DEBUG: No categories found in response for ${path.basename(filePath)}, attempting fallback parsing ===`);
      
      // Try to extract any tags that start with a dash
      const fallbackTags = lines
        .slice(1)
        .filter(line => line.trim().startsWith('-'))
        .map(line => line.trim().replace(/^-\s*/, '').trim())
        .filter(tag => tag.length > 0);
      
      if (fallbackTags.length > 0) {
        topics = fallbackTags; // Add all uncategorized tags to topics
        console.log(`Found ${fallbackTags.length} tags using fallback parsing`);
      } else {
        // If we still can't find tags, make a second attempt with more flexible parsing
        const potentialTags = lines
          .slice(1)
          .map(line => line.trim())
          .filter(line => line.length > 0 && !line.includes(':') && line.length < 50)
          .slice(0, 10); // Limit to first 10 potential tags
        
        if (potentialTags.length > 0) {
          topics = potentialTags;
          console.log(`Found ${potentialTags.length} potential tags with flexible parsing`);
        }
      }
    }

    // Combine all tags for compatibility
    const allTags = [...new Set([...companies, ...models, ...topics])];

    if (allTags.length === 0) {
      console.warn(`No tags generated for ${filePath}`);
      
      // For files with no tags, add a default "untagged" topic
      topics = ["untagged"];
      allTags.push("untagged");
      
      // console.log(`\n=== DEBUG: Response structure analysis for ${path.basename(filePath)} ===`);
      // console.log(`Total lines: ${lines.length}`);
      // console.log(`First 5 lines:`);
      // for (let i = 0; i < Math.min(5, lines.length); i++) {
      //   console.log(`Line ${i+1}: "${lines[i]}"`);
      // }
    } else {
      // console.log(`Generated tags for ${path.basename(filePath)}:`);
      // console.log(`Companies (${companies.length}): ${companies.join(', ')}`);
      // console.log(`Models (${models.length}): ${models.join(', ')}`);
      // console.log(`Topics (${topics.length}): ${topics.join(', ')}`);
      // console.log(`Total tags: ${allTags.length}`);
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