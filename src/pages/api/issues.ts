import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export const GET: APIRoute = async ({ url }) => {
  const page = parseInt(url.searchParams.get('page') || '1');
  const itemsPerPage = 30;
  console.log(`[API] Processing request for page ${page}`);

  try {
    // Get all issues with logging, only fetching essential metadata
    const allIssuesRaw = await getCollection('issues', ({ data }) => {
      // Only fetch the data we need for the listing page
      return {
        id: true,
        data: {
          title: true,
          date: true, 
          tags: true,
          models: true,
          companies: true,
          topics: true,
          description: true,
          draft: true
        }
      };
    });
    console.log(`[API] Total raw issues before filtering: ${allIssuesRaw.length}`);
    
    // Log the years represented in the content
    const years = new Set(allIssuesRaw.map(post => new Date(post.data.date).getFullYear()));
    console.log(`[API] Years found in content: ${[...years].sort().join(', ')}`);
    
    // Count 2025 issues specifically
    const issues2025 = allIssuesRaw.filter(post => new Date(post.data.date).getFullYear() === 2025);
    console.log(`[API] Number of 2025 issues found: ${issues2025.length}`);
    
    // Apply filtering with logging
    const allIssues = allIssuesRaw
      .filter(post => {
        const isDraft = !!post.data.draft;
        if (isDraft) {
          console.log(`[API] Filtered out draft post: ${post.id}`);
        }
        return !isDraft;
      })
      .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
    
    console.log(`[API] Total issues after filtering: ${allIssues.length}`);

    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedIssues = allIssues.slice(start, end).map(issue => ({
      id: issue.id,
      title: issue.data.title,
      date: issue.data.date,
      models: issue.data.models,
      companies: issue.data.companies,
      topics: issue.data.topics,
      description: issue.data.description
    }));

    return new Response(
      JSON.stringify({
        issues: paginatedIssues,
        totalPages: Math.ceil(allIssues.length / itemsPerPage),
        currentPage: page
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
  } catch (error) {
    console.error(`[API] Error processing issues:`, error);
    return new Response(
      JSON.stringify({
        error: 'Failed to fetch issues'
      }),
      {
        status: 500,
        headers: {
          'Content-Type': 'application/json'
        }
      }
    );
  }
} 