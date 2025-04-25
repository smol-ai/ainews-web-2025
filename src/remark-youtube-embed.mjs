import { visit } from 'unist-util-visit';

// Based on the YouTube matcher from astro-embed-youtube
// https://github.com/delucis/astro-embed/blob/main/packages/astro-embed-youtube/matcher.ts
const urlPattern = /(?=(\s*))\1(?:<a [^>]*?>)??(?=(\s*))\2(?:https?:\/\/)??(?:w{3}\.)??(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|shorts\/)??([A-Za-z0-9-_]{11})(?:[^\s<>]*)(?=(\s*))\4(?:<\/a>)??(?=(\s*))\5/;

/**
 * Extract a YouTube ID from a URL if it matches the pattern.
 * @param {string} url - URL to test
 * @returns {string|undefined} A YouTube video ID or undefined if none matched
 */
function extractVideoId(url) {
  const match = url.match(urlPattern);
  return match?.[3];
}

/**
 * Remark plugin to transform standalone YouTube links to YouTube embeds
 */
export function remarkYouTubeEmbed() {
  return function transformer(tree) {
    visit(tree, 'paragraph', (node, index, parent) => {
      // Only process paragraphs that contain a single text or link node
      if (node.children.length !== 1) return;
      
      let url = '';
      let isLink = false;
      
      // Check if it's a plain URL text node
      if (node.children[0].type === 'text') {
        url = node.children[0].value.trim();
      } 
      // Check if it's a link node
      else if (node.children[0].type === 'link') {
        url = node.children[0].url;
        isLink = true;
      }
      else {
        return;
      }
      
      // Extract video ID using the pattern matcher
      const videoId = extractVideoId(url);
      
      if (videoId) {
        // Create the YouTube component element
        const embedNode = {
          type: 'html',
          value: `<div class="youtube-embed-container">
            <lite-youtube 
              videoid="${videoId}" 
              style="background-image: url('https://i.ytimg.com/vi/${videoId}/hqdefault.jpg');"
            >
              <a href="https://youtube.com/watch?v=${videoId}" class="lty-playbtn">
                <span class="lyt-visually-hidden">Play</span>
              </a>
            </lite-youtube>
          </div>
          <style>
            .youtube-embed-container {
              margin: 1.5em 0;
            }
            lite-youtube {
              width: 100%;
              max-width: 720px;
              aspect-ratio: 16 / 9;
              border-radius: 0.5rem;
              overflow: hidden;
              position: relative;
              display: block;
              contain: content;
              background-position: center;
              background-size: cover;
              cursor: pointer;
            }
          </style>`
        };
        
        // Replace the paragraph node with the embed node
        parent.children.splice(index, 1, embedNode);
        return [visit.SKIP, index];
      }
    });
  };
}

export default remarkYouTubeEmbed; 