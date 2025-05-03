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
 * @param {string | null | undefined} text The input markdown string.
 * @returns {string} The converted HTML string.
 */
export function markdownToHtml(text) {
  if (!text) return '';

  let html = String(text);

  // Convert bold (**text**)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Convert italics (*text*)
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Convert code blocks (`text`)
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');

  // Convert links [text](url). must come after images
  html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2">$1</a>');

  return html;
}