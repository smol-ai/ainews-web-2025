/** @jsxImportSource react */
import React from 'react';
import { ImageResponse } from '@vercel/og';
import type { APIRoute } from 'astro';
import { renderIssueOgImage, renderHomepageOgImage } from '../../lib/renderOgImage';

// https://og-playground.vercel.app/
export const prerender = false;

export const GET: APIRoute = async ({ request }) => {
  try {
    const url = new URL(request.url);
    console.log('request', request.url)
    console.log('title', url.searchParams.get('title'))
    console.log('type', url.searchParams.get('type'))
    
    // Extract parameters
    const title = url.searchParams.get('title') || 'AI News';
    const description = url.searchParams.get('description') || 'Weekday recaps of top News for AI Engineers';
    const type = url.searchParams.get('type') || 'homepage';
    const issueNumber = url.searchParams.get('issueNumber') || undefined;
    const date = url.searchParams.get('date');
    const secret = url.searchParams.get('secret');
    
    // Get company and model tags if available
    const companyTags = url.searchParams.get('companies')?.split(',').filter(Boolean) || [];
    const modelTags = url.searchParams.get('models')?.split(',').filter(Boolean) || [];
    
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
    
    if (secret !== "nocache") {
      headers.set('Cache-Control', 'public, max-age=86400'); // 1 day cache
    } else {
      headers.set('Cache-Control', 'no-cache, no-store');
    }
    
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
    console.error('Error generating OG image:', error);
    return new Response('Error generating image', { status: 500 });
  }
}; 