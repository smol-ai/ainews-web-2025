import path from 'node:path';
import { fetchTopPostsFromSubreddit } from './api';
import { checkpoint, writeJson } from '../shared/state';

export const redditPresets = {
  local: ['localLlama', 'localLLM'],
  all: ['Singularity', 'Oobabooga', 'MachineLearning', 'OpenAI', 'ClaudeAI', 'ClaudeCode', 'DeepSeek', 'Bard', 'GeminiAI', 'Qwen_AI', 'PromptEngineering', 'aivideo', 'cline', 'StableDiffusion', 'ChatGPT', 'ChatGPTCoding', 'veo3', 'SillyTavernAI'],
};
const small: Record<string, number> = { bard: 80, qwen_ai: 80, cline: 80, veo3: 80, promptengineering: 80, sillytavernai: 80, oobabooga: 80, geminiai: 100, aivideo: 100, deepseek: 150 };

export async function scrapeReddit(directory: string, subreddits: string[], minimum: number): Promise<string> {
  if (!process.env.REDDIT_SESSION) throw new Error('Set REDDIT_SESSION in tools/scrapers/.env, then resume this run.');
  const results = [];
  for (const subreddit of subreddits) {
    const key = subreddit.toLowerCase();
    console.log(`[reddit] ${subreddit}`);
    results.push(await checkpoint(path.join(directory, 'checkpoints', `${key}.json`), () =>
      fetchTopPostsFromSubreddit(`/r/${subreddit}`, Math.min(minimum, small[key] ?? minimum))));
  }
  const file = path.join(directory, 'raw.json');
  writeJson(file, results);
  return file;
}
