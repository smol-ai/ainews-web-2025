/**
 * Types for the Reddit scraper
 */

export interface RedditComment {
  author: string;
  body: string;
  score: number;
  created_utc: number;
  permalink: string;
  replies?: RedditComment[];
}

export interface RedditPost {
  title: string;
  author: string;
  url: string;
  permalink: string;
  score: number;
  num_comments: number;
  created_utc: number;
  selftext?: string;
  top_comments?: RedditComment[];
}

export interface SubredditData {
  subreddit: string;
  posts: RedditPost[];
  filterStats?: {
    totalPostsViewed: number;
    postsAfterFilter: number;
    minimumScore: number;
    /** Included because no post met minimumScore; these were still added as a small active-sub fallback */
    fallbackBelowMinimumCount?: number;
  };
}
