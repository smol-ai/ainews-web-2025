/**
 * Utility for generating Open Graph image URLs
 */

export interface OgImageParams {
  title: string;
  description?: string;
  type?: 'issue' | 'homepage' | 'tag';
  issueNumber?: string;
  date?: string | Date;
  secret?: string;
  companies?: string[];
  models?: string[];
}

/**
 * Generate a URL for dynamic Open Graph images
 */
export function generateOgImageUrl(params: OgImageParams): string {
  // Base URL for the OG image API (site-relative)
  const baseUrl = '/api/og';
  
  // Build query parameters
  const queryParams = new URLSearchParams();
  
  // Always include title
  if (params.title) {
    queryParams.set('title', params.title.slice(0, params.title.length - 9));
  }
  
  // Optional parameters
  if (params.description) {
    queryParams.set('description', params.description);
  }
  
  if (params.type) {
    queryParams.set('type', params.type);
  }
  
  if (params.issueNumber) {
    queryParams.set('issueNumber', params.issueNumber);
  }
  
  if (params.date) {
    const dateStr = params.date instanceof Date ? params.date.toISOString() : params.date;
    queryParams.set('date', dateStr);
  }
  
  // Add company and model tags
  if (params.companies && params.companies.length > 0) {
    queryParams.set('companies', params.companies.join(','));
  }
  
  if (params.models && params.models.length > 0) {
    queryParams.set('models', params.models.join(','));
  }
  
  // Secret for cache busting
  if (params.secret) {
    queryParams.set('secret', params.secret);
  }
  
  return `${baseUrl}?${queryParams.toString()}`;
} 