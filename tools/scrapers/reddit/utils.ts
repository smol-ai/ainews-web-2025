import { RedditComment } from './types';
import * as fs from 'fs';
import * as path from 'path';
import chalk from 'chalk';
import { writeJson } from '../shared/state';

/**
 * Count the total number of nested replies in a comment tree
 */
export const countNestedReplies = (comment: RedditComment): number => {
  if (!comment.replies || comment.replies.length === 0) {
    return 0;
  }

  let count = comment.replies.length;

  // Recursively count replies to replies
  comment.replies.forEach(reply => {
    count += countNestedReplies(reply);
  });

  return count;
};

/**
 * Recursively process comment data to extract comments and their replies
 */
export const processCommentData = (commentData: any, depth: number = 0): RedditComment | null => {
  // Skip if not a valid comment or if it's been removed/deleted
  if (!commentData || !commentData.data || commentData.kind !== 't1' ||
      commentData.data.body === '[removed]' || commentData.data.body === '[deleted]') {
    return null;
  }

  const data = commentData.data;

  // Create the comment object with basic info
  const comment: RedditComment = {
    author: data.author,
    body: data.body,
    score: data.score || data.ups || 0,
    created_utc: data.created_utc,
    permalink: data.permalink || ''
  };

  // Process replies if they exist
  if (data.replies && typeof data.replies === 'object' && data.replies.kind === 'Listing') {
    const replies: RedditComment[] = [];

    // Process each child comment
    if (data.replies.data && Array.isArray(data.replies.data.children)) {
      console.log(`Found ${data.replies.data.children.length} replies at depth ${depth} for comment by ${data.author}`);

      data.replies.data.children.forEach((childComment: any) => {
        // Skip "more" type comments
        if (childComment.kind === 'more') {
          console.log(`Skipping 'more' comment with ${childComment.data.count} additional replies`);
          return;
        }

        const processedReply = processCommentData(childComment, depth + 1);
        if (processedReply) {
          replies.push(processedReply);
        }
      });

      // Sort replies by score (highest first)
      if (replies.length > 0) {
        replies.sort((a, b) => b.score - a.score);
        comment.replies = replies;
        console.log(`Added ${replies.length} sorted replies to comment by ${data.author}`);
      }
    }
  }
  // else if (data.replies === '') {
  //   console.log(`Comment by ${data.author} has no replies (empty string)`);
  // } else if (!data.replies) {
  //   console.log(`Comment by ${data.author} has no replies (undefined/null)`);
  // } else {
  //   console.log(`Comment by ${data.author} has replies in unexpected format:`, typeof data.replies);
  // }

  return comment;
};

/** One canonical output per run; no global latest-file aliases. */
export const saveResultsToFile = (results: unknown, baseFilename = 'reddit_posts', _runId?: string, _subreddits?: string[]): string => {
  const target = path.join(process.cwd(), 'reports', `${baseFilename}.json`);
  writeJson(target, results);
  return target;
};
