import type { APIRoute } from 'astro';

// Simple diagnostic endpoint
export const GET: APIRoute = async ({ request, url }) => {
  try {
    const requestUrl = new URL(request.url);
    const directUrl = url;
    
    // Collect debug information
    const debugInfo = {
      requestMethod: request.method,
      fullRequestUrl: request.url,
      parsedFromRequest: {
        pathname: requestUrl.pathname,
        search: requestUrl.search,
        searchParams: Object.fromEntries(requestUrl.searchParams.entries()),
      },
      directAstroUrl: {
        pathname: directUrl.pathname,
        search: directUrl.search,
        searchParams: Object.fromEntries(directUrl.searchParams.entries()),
      },
      headers: Object.fromEntries(request.headers.entries()),
    };
    
    // Return debug information as JSON
    return new Response(JSON.stringify(debugInfo, null, 2), {
      status: 200,
      headers: {
        'Content-Type': 'application/json'
      }
    });
  } catch (error) {
    console.error('Debug endpoint error:', error);
    return new Response(
      JSON.stringify({ error: error instanceof Error ? error.message : 'Unknown error' }),
      { status: 500, headers: { 'Content-Type': 'application/json' } }
    );
  }
}; 