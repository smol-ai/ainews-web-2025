import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

export const GET: APIRoute = async ({ url }) => {
  const page = parseInt(url.searchParams.get('page') || '1');
  const itemsPerPage = 30;

  try {
    const allIssues = (await getCollection('issues'))
      .filter(post => !post.data.draft)
      .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());

    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedIssues = allIssues.slice(start, end).map(issue => ({
      id: issue.id,
      title: issue.data.title,
      date: issue.data.date,
      tags: issue.data.tags,
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