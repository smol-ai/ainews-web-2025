import fs from 'fs';
import path from 'path';
import axios from 'axios';
import yaml from 'js-yaml';
import { format } from 'date-fns';
import slugify from 'slugify';

const BUTTONDOWN_API_KEY = process.env.BUTTONDOWN_API_KEY;
const OUTPUT_DIR = path.join(process.cwd(), 'buttondown-emails');

if (!BUTTONDOWN_API_KEY) {
  throw new Error('BUTTONDOWN_API_KEY environment variable is required');
}

interface ButtondownEmail {
  id: string;
  creation_date: string;
  modification_date: string;
  publish_date: string;
  subject: string;
  body: string;
  slug: string;
  status: string;
  email_type: string;
  source: string;
  metadata: Record<string, any>;
  [key: string]: any;
}

async function getAllEmails(): Promise<ButtondownEmail[]> {
  const emails: ButtondownEmail[] = [];
  let page = 1;
  let hasMore = true;

  while (hasMore) {
    console.log(`Fetching page ${page}...`);
    const response = await axios.get('https://api.buttondown.com/v1/emails', {
      headers: {
        Authorization: `Token ${BUTTONDOWN_API_KEY}`,
      },
      params: {
        page,
      },
    });

    const { results, count } = response.data;
    emails.push(...results);

    console.log(`Fetched ${results.length} emails (total: ${emails.length}/${count})`);
    
    if (results.length === 0 || emails.length >= count) {
      hasMore = false;
    } else {
      page++;
    }
  }

  return emails;
}

function formatDate(dateString: string): string {
  return format(new Date(dateString), 'yy-MM-dd');
}

function createFrontmatter(email: ButtondownEmail): string {
  const frontmatter = {
    id: email.id,
    title: email.subject,
    date: email.publish_date,
    status: email.status,
    type: email.email_type,
    source: email.source,
    metadata: email.metadata,
    original_slug: email.slug,
  };

  return `---\n${yaml.dump(frontmatter)}---\n\n`;
}

async function saveEmailAsMarkdown(email: ButtondownEmail) {
  const date = formatDate(email.publish_date);
  const slugifiedTitle = slugify(email.subject, { lower: true, strict: true });
  const filename = `${date}-${slugifiedTitle}.md`;
  const filepath = path.join(OUTPUT_DIR, filename);

  const content = createFrontmatter(email) + email.body;

  await fs.promises.writeFile(filepath, content, 'utf8');
  console.log(`Saved: ${filename}`);
}

async function main() {
  try {
    // Create output directory if it doesn't exist
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    console.log('Fetching all emails from Buttondown...');
    const emails = await getAllEmails();
    console.log(`Total emails fetched: ${emails.length}`);

    console.log('Saving emails as markdown files...');
    for (const email of emails) {
      await saveEmailAsMarkdown(email);
    }

    console.log('Done!');
  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  }
}

main();
