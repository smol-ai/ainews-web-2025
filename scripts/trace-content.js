#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

// Get current directory in ESM
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

console.log('============= CONTENT TRACING SCRIPT =============');
console.log('Environment:', process.env.NODE_ENV || 'development');

// Check content directory
const contentPath = path.join(projectRoot, 'src/content/issues');
console.log(`\nScanning content directory: ${contentPath}`);

// List files in content directory
try {
  const files = fs.readdirSync(contentPath);
  console.log(`Found ${files.length} files in content directory`);
  
  // Count files with 2025 in the name
  const files2025 = files.filter(file => file.includes('25-'));
  console.log(`Found ${files2025.length} files with '25-' in the name`);
  console.log('Sample 2025 files:', files2025.slice(0, 5));
  
  // Sample content from a 2025 file
  if (files2025.length > 0) {
    const sampleFile = files2025[0];
    const sampleFilePath = path.join(contentPath, sampleFile);
    console.log(`\nSample content from ${sampleFile}:`);
    
    const content = fs.readFileSync(sampleFilePath, 'utf8');
    const frontmatter = content.split('---')[1]; // Extract frontmatter
    console.log(frontmatter);
  }
} catch (err) {
  console.error('Error scanning content directory:', err);
}

// Run build with verbose logging
console.log('\n============= RUNNING BUILD WITH TRACING =============');
try {
  // Use pnpm for the build with added environment variables for tracing
  execSync('ASTRO_LOG_LEVEL=debug pnpm build', { 
    stdio: 'inherit',
    env: {
      ...process.env,
      VERBOSE_BUILD: 'true',
      NODE_ENV: 'production',
      ASTRO_LOG_LEVEL: 'debug'
    }
  });
} catch (err) {
  console.error('Build failed:', err);
  process.exit(1);
}

// Check build output
console.log('\n============= CHECKING BUILD OUTPUT =============');
const distPath = path.join(projectRoot, 'dist');
try {
  const distFiles = fs.readdirSync(distPath);
  console.log(`Found ${distFiles.length} items in build output directory`);
  
  // Check for 2025 directories in the built issues folder
  const issuesDistPath = path.join(distPath, 'issues');
  if (fs.existsSync(issuesDistPath)) {
    const issuesDistItems = fs.readdirSync(issuesDistPath);
    console.log(`Found ${issuesDistItems.length} items in issues build output`);
    
    // Count directories that might contain 2025 content
    const issues2025 = issuesDistItems.filter(item => item.startsWith('25-'));
    console.log(`Found ${issues2025.length} items that start with '25-' in built issues`);
    if (issues2025.length > 0) {
      console.log('Sample built 2025 issues:', issues2025.slice(0, 5));
    } else {
      console.log('WARNING: No 2025 issues found in build output!');
    }
  } else {
    console.log('WARNING: Issues directory not found in build output!');
  }
} catch (err) {
  console.error('Error checking build output:', err);
}

console.log('\n============= CONTENT TRACING COMPLETE ============='); 