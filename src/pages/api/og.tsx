/** @jsxImportSource react */
import React from 'react';
import { ImageResponse } from '@vercel/og';
import type { APIRoute } from 'astro';
import { renderIssueOgImage, renderHomepageOgImage } from '../../lib/renderOgImage';

// https://og-playground.vercel.app/

export const GET: APIRoute = async ({ request }) => {
  try {
    // Debug logging to help identify issues
    console.log('OG Image Request URL:', request.url);
    
    const url = new URL(request.url);
    
    // Debug log all search parameters
    console.log('Search Params:');
    for (const [key, value] of url.searchParams.entries()) {
      console.log(`  ${key}: ${value}`);
    }
    
    // Extract parameters
    const title = url.searchParams.get('title') || 'AI News';
    const description = url.searchParams.get('description') || 'Weekday recaps of top News for AI Engineers';
    const type = url.searchParams.get('type') || 'homepage';
    const issueNumber = url.searchParams.get('issueNumber') || undefined;
    const date = url.searchParams.get('date');
    const secret = url.searchParams.get('secret');
    
    // Get company and model tags if available
    const companiesParam = url.searchParams.get('companies') || '';
    const modelsParam = url.searchParams.get('models') || '';
    
    // Debug log extracted array parameters
    console.log('Companies param:', companiesParam);
    console.log('Models param:', modelsParam);
    
    // Safely split and filter
    const companyTags = companiesParam 
      ? companiesParam.split(',').filter(tag => tag && tag.trim() !== '')
      : [];
      
    const modelTags = modelsParam
      ? modelsParam.split(',').filter(tag => tag && tag.trim() !== '')
      : [];
      
    // Debug log processed tags
    console.log('Company tags:', companyTags);
    console.log('Model tags:', modelTags);
    
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
    
    // Don't cache if secret is provided
    if (!secret) {
      headers.set('Cache-Control', 'public, max-age=86400'); // 1 day cache
    } else {
      headers.set('Cache-Control', 'no-cache, no-store');
    }
    
    console.log(`Rendering OG image for type: ${type}`);
    
    // Different layout for issue vs other types
    if (type === 'issue') {
      // Issue-specific layout with more prominent date and title
      console.log('Rendering issue OG image with:', { 
        title, 
        description: description.substring(0, 50) + '...', 
        issueNumber,
        companyTags: companyTags.length,
        modelTags: modelTags.length
      });
      
      return new ImageResponse(
        renderIssueOgImage({
          title,
          description,
          formattedDate,
          companyTags: companyTags || [],
          modelTags: modelTags || [],
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
      console.log('Rendering homepage OG image with:', { 
        title, 
        description: description.substring(0, 50) + '...'
      });
      
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
    
    // Return a fallback image or an error message
    return new Response(`Error generating image: ${error instanceof Error ? error.message : 'Unknown error'}`, {
      status: 500,
      headers: {
        'Content-Type': 'text/plain'
      }
    });
  }
}; 