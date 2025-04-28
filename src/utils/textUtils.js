/**
 * Removes null characters (\u0000) from a string.
 * Handles null or undefined inputs gracefully.
 * @param {string | null | undefined} str The input string.
 * @returns {string} The sanitized string, or an empty string if input is null/undefined.
 */
export function sanitizeString(str) {
  if (str == null) {
    return '';
  }
  // Ensure input is treated as a string before replacing
  return String(str).replace(/\u0000/g, '');
}

/**
 * Simple markdown to HTML converter for inline formatting.
 * Converts bold, italics, code, and links.
 * If fullConversion is true, also converts images, blockquotes, headers, paragraphs, and bullet points. (things that often take more vertical space)
 * @param {string | null | undefined} text The input markdown string.
 * @param {boolean} [fullConversion=false] Whether to perform full conversion.
 * @returns {string} The converted HTML string.
 */
export function markdownToHtml(text, fullConversion = false) {
  if (!text) return '';

  let html = String(text);

  // Convert bold (**text**)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Convert italics (*text*)
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Convert code blocks (`text`)
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');


  if (fullConversion) {
    // Convert images ![alt](url)
    html = html.replace(/!\[(.*?)\]\((.*?)\)/g, '<img src="$2" alt="$1">');

    // Convert blockquotes
    html = html.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

    // Convert headers
    html = html.replace(/^#{1,6} (.+)$/gm, (match, p1) => {
      const level = match.trim().indexOf(' ');
      return `<h${level}>${p1}</h${level}>`;
    });

    // Convert paragraphs
    html = html.replace(/^(?!<[a-z]|\s*$)(.+)$/gm, '<p>$1</p>');

    // Convert bullet points
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/<li>.*?<\/li>/gs, '<ul>$&</ul>');

    // Convert horizontal rules
    html = html.replace(/^---+$/gm, '<hr>');
  }

  // Convert links [text](url). must come after images
  html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>');

  return html;
}