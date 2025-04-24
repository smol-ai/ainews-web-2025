/**
 * A remark plugin to transform standalone YouTube links in markdown into lite-youtube embeds
 */
import { visit } from 'unist-util-visit';

// Regular expressions for different YouTube URL patterns
const YOUTUBE_REGEX = [
  // Standard youtube.com URLs
  /^https?:\/\/(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})(?:&.*)?$/,
  // Short youtu.be URLs
  /^https?:\/\/(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]{11})(?:\?.*)?$/,
  // Embed URLs
  /^https?:\/\/(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})(?:\?.*)?$/,
];

/**
 * Extract YouTube video ID from a URL
 * @param {string} url - YouTube URL
 * @returns {string|null} Video ID or null if not a valid YouTube URL
 */
function extractVideoId(url) {
  for (const regex of YOUTUBE_REGEX) {
    const match = url.match(regex);
    if (match && match[1]) {
      return match[1];
    }
  }
  return null;
}

/**
 * Extract parameters from a YouTube URL
 * @param {string} url - YouTube URL
 * @returns {string} Parameter string or empty string
 */
function extractParams(url) {
  try {
    const urlObj = new URL(url);
    const params = new URLSearchParams(urlObj.search);
    
    // Skip the 'v' parameter as it's used for the video ID
    params.delete('v');
    
    // Convert the remaining parameters to a string
    return params.toString();
  } catch (e) {
    return '';
  }
}

/**
 * Remark plugin to transform standalone YouTube links to lite-youtube embeds
 * @returns {function} Transformer function
 */
export function remarkYouTube() {
  return function transformer(tree) {
    // First, handle paragraphs that contain a single link
    visit(tree, 'paragraph', (node, index, parent) => {
      // Case 1: Paragraph with a single link node
      if (
        node.children.length === 1 &&
        node.children[0].type === 'link'
      ) {
        const link = node.children[0];
        const url = link.url;
        const videoId = extractVideoId(url);
        
        if (videoId) {
          const params = extractParams(url);
          
          // Create a new node for the lite-youtube embed
          const embedNode = {
            type: 'html',
            value: `<lite-youtube videoid="${videoId}" ${params ? `params="${params}"` : ''}></lite-youtube>`
          };
          
          // Replace the paragraph node with the embed node
          parent.children.splice(index, 1, embedNode);
          return [visit.SKIP, index];
        }
      }
      
      // Case 2: Paragraph with a single text node containing only a YouTube URL
      if (
        node.children.length === 1 &&
        node.children[0].type === 'text'
      ) {
        const text = node.children[0].value.trim();
        const videoId = extractVideoId(text);
        
        if (videoId) {
          const params = extractParams(text);
          
          // Create a new node for the lite-youtube embed
          const embedNode = {
            type: 'html',
            value: `<lite-youtube videoid="${videoId}" ${params ? `params="${params}"` : ''}></lite-youtube>`
          };
          
          // Replace the paragraph node with the embed node
          parent.children.splice(index, 1, embedNode);
          return [visit.SKIP, index];
        }
      }
    });
  };
}

export default remarkYouTube; 