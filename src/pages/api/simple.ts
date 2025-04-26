import type { APIRoute } from 'astro';

// Ultra-simple endpoint to verify basic parameter parsing
export const GET: APIRoute = async ({ request, url }) => {
  const title = url.searchParams.get('title');
  
  return new Response(`
Simple API endpoint test
=======================
Request URL: ${request.url}
Astro URL: ${url}
Title parameter: ${title || 'NOT FOUND'}
All parameters:
${Array.from(url.searchParams.entries())
  .map(([key, value]) => `  ${key}: ${value}`)
  .join('\n')}
`,
    {
      status: 200,
      headers: {
        'Content-Type': 'text/plain'
      }
    }
  );
}; 