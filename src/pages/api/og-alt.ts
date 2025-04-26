import type { APIRoute } from 'astro';
import { ImageResponse } from '@vercel/og';
import { renderIssueOgImage, renderHomepageOgImage } from '../../lib/renderOgImage';

// Alternative approach using Astro's URL directly
export const GET: APIRoute = async ({ url, request }) => {
  try {
    // Debug logging to help identify issues
    console.log('OG-ALT Image Request URL:', request.url);
    console.log('OG-ALT Astro URL:', url.toString());
    
    // Debug log all search parameters
    console.log('OG-ALT Direct Search Params:');
    for (const [key, value] of url.searchParams.entries()) {
      console.log(`  ${key}: ${value}`);
    }
    
    // Extract parameters directly from Astro's URL object
    const title = url.searchParams.get('title') || 'AI News';
    const description = url.searchParams.get('description') || 'Weekday recaps of top News for AI Engineers';
    const type = url.searchParams.get('type') || 'homepage';
    const issueNumber = url.searchParams.get('issueNumber') || undefined;
    const date = url.searchParams.get('date');
    
    // Get company and model tags if available
    const companiesParam = url.searchParams.get('companies') || '';
    const modelsParam = url.searchParams.get('models') || '';
    
    // Debug log extracted array parameters
    console.log('OG-ALT Companies param:', companiesParam);
    console.log('OG-ALT Models param:', modelsParam);
    
    // Safely split and filter
    const companyTags = companiesParam 
      ? companiesParam.split(',').filter(tag => tag && tag.trim() !== '')
      : [];
      
    const modelTags = modelsParam
      ? modelsParam.split(',').filter(tag => tag && tag.trim() !== '')
      : [];
    
    // Format date nicely if available
    const formattedDate = date 
      ? new Date(date).toLocaleDateString('en-US', {
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        })
      : '';
    
    // Check caching headers
    const headers = new Headers();
    headers.set('Cache-Control', 'public, max-age=86400'); // 1 day cache
    
    console.log(`OG-ALT Rendering OG image for type: ${type}`);
    
    // Different layout for issue vs other types
    if (type === 'issue') {
      // Issue-specific layout with more prominent date and title
      return new ImageResponse(
        renderIssueOgImage({
          title,
          description,
          formattedDate,
          companyTags,
          modelTags,
          issueNumber
        }),
        {
          width: 1200,
          height: 630,
          headers,
        }
      );
    } else {
      // Standard homepage layout
      return new ImageResponse(
        renderHomepageOgImage({
          title,
          description
        }),
        {
          width: 1200,
          height: 630,
          headers,
        }
      );
    }
  } catch (error) {
    console.error('OG-ALT Error generating OG image:', error);
    
    // Return a basic error message
    return new Response(`Error generating image: ${error instanceof Error ? error.message : 'Unknown error'}`, {
      status: 500,
      headers: {
        'Content-Type': 'text/plain'
      }
    });
  }
}; 