export type MediaType = 'image' | 'video' | 'card' | 'unknown';

export interface MediaAttachment {
  type: MediaType;
  previewUrl: string;
  fullUrl?: string;
  alt?: string;
}

export interface UserRef {
  username: string;
  displayName: string;
  profileUrl?: string;
  avatarUrl?: string;
}

export interface Stats {
  replies: number;
  retweets: number;
  quotes: number;
  likes: number;
  plays?: number;
}

export interface Quote {
  tweetId?: string;
  url?: string;
  user?: UserRef;
  text?: string;
  media?: MediaAttachment[];
  timestamp?: string; // ISO
}

export interface Tweet {
  id: string;
  url: string;
  user: UserRef;
  timestamp: string; // ISO
  timestampMs: number;
  text: string;
  attachments: MediaAttachment[];
  stats: Stats;
  retweetedBy?: string;
  quoted?: Quote;
  isThread?: boolean;
}

export interface Diagnostics {
  listId: string;
  baseUrl: string;
  pagesFetched: number;
  bytesFetched: number;
  tweetCount: number;
  startParam: string; // ISO
  endParam: string;   // ISO
  earliestTimestamp?: string; // ISO
  latestTimestamp?: string;   // ISO
  executionStart: string; // ISO
  executionEnd: string;   // ISO
  durationMs: number;
  wasPartial?: boolean;        // true if scraping stopped due to error
  errorMessage?: string;       // error message if wasPartial is true
  stoppedByCharBudget?: boolean; // true if scraping stopped because char budget was reached
}

export interface ScrapeResult {
  tweets: Tweet[];
  diagnostics: Diagnostics;
}
