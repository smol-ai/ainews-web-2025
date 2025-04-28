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