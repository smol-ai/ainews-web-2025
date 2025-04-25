import { ImageResponse } from '@vercel/og';
import type { APIRoute } from 'astro';

export const GET: APIRoute = async ({ request }) => {
  try {
    const url = new URL(request.url);
    
    // Extract parameters
    const title = url.searchParams.get('title') || 'AI News';
    const description = url.searchParams.get('description') || 'Weekday recaps of top News for AI Engineers';
    const type = url.searchParams.get('type') || 'homepage';
    const issueNumber = url.searchParams.get('issueNumber');
    const date = url.searchParams.get('date');
    const secret = url.searchParams.get('secret');
    
    // Check caching headers
    const headers = new Headers();
    
    // Don't cache if secret is provided
    if (!secret) {
      headers.set('Cache-Control', 'public, max-age=86400'); // 1 day cache
    } else {
      headers.set('Cache-Control', 'no-cache, no-store');
    }
    
    // For proper TypeScript support, we need to handle the JSX differently
    // We'll create the HTML structure as an object that @vercel/og can handle
    return new ImageResponse(
      {
        type: 'div',
        props: {
          style: {
            height: '100%',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            backgroundColor: '#fafafa',
            color: '#111',
            padding: '60px',
            position: 'relative',
            fontFamily: 'Geist Sans, system-ui, -apple-system, sans-serif',
          },
          children: [
            // Top border line
            {
              type: 'div',
              props: {
                style: {
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: '6px',
                  backgroundColor: '#111'
                }
              }
            },
            
            // Issue label in top right corner (conditional)
            ...(type === 'issue' ? [{
              type: 'div',
              props: {
                style: {
                  position: 'absolute',
                  top: '24px',
                  right: '60px',
                  fontSize: 18,
                  color: '#555',
                  display: 'flex',
                  alignItems: 'center',
                },
                children: [
                  ...(issueNumber ? [{
                    type: 'span',
                    props: {
                      style: {
                        marginRight: 10,
                        fontWeight: 'bold'
                      },
                      children: `ISSUE #${issueNumber}`
                    }
                  }] : []),
                  ...(date ? [{
                    type: 'span',
                    props: {
                      children: new Date(date).toLocaleDateString('en-US', {
                        year: 'numeric',
                        month: 'short',
                        day: 'numeric'
                      })
                    }
                  }] : [])
                ]
              }
            }] : []),
            
            // Newspaper title - like a masthead
            {
              type: 'div',
              props: {
                style: {
                  marginBottom: 16,
                  borderBottom: '1px solid #ccc',
                  paddingBottom: 16,
                  display: 'flex',
                  flexDirection: 'column',
                },
                children: [
                  {
                    type: 'div',
                    props: {
                      style: {
                        fontSize: 28,
                        fontWeight: 'bold',
                        letterSpacing: '-0.02em',
                      },
                      children: 'AI NEWS'
                    }
                  },
                  {
                    type: 'div',
                    props: {
                      style: {
                        fontSize: 16,
                        color: '#555',
                      },
                      children: 'WITH SMOL.AI'
                    }
                  }
                ]
              }
            },
            
            // Main headline
            {
              type: 'div',
              props: {
                style: {
                  marginTop: 20,
                  marginBottom: 30,
                  maxWidth: '90%',
                  display: 'flex',
                  flexDirection: 'column',
                },
                children: [
                  {
                    type: 'h1',
                    props: {
                      style: {
                        fontSize: 60,
                        fontWeight: 800,
                        lineHeight: 1.1,
                        letterSpacing: '-0.02em',
                        margin: 0,
                        color: '#000',
                      },
                      children: title
                    }
                  }
                ]
              }
            },
            
            // Description with text overflow handling
            {
              type: 'div',
              props: {
                style: {
                  fontSize: 24,
                  lineHeight: 1.4,
                  color: '#444',
                  maxHeight: '200px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: '-webkit-box',
                  WebkitLineClamp: 3,
                  WebkitBoxOrient: 'vertical',
                },
                children: description
              }
            },
            
            // Bottom footer
            {
              type: 'div',
              props: {
                style: {
                  position: 'absolute',
                  bottom: '24px',
                  left: '60px',
                  right: '60px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  borderTop: '1px solid #ccc',
                  paddingTop: '16px',
                },
                children: [
                  {
                    type: 'div',
                    props: {
                      style: {
                        fontSize: 16,
                        color: '#555'
                      },
                      children: 'ainews.page'
                    }
                  },
                  {
                    type: 'div',
                    props: {
                      style: {
                        fontSize: 16,
                        color: '#555'
                      },
                      children: '@smol_ai'
                    }
                  }
                ]
              }
            }
          ]
        }
      },
      {
        width: 1200,
        height: 630,
        headers,
      }
    );
  } catch (error) {
    console.error('Error generating OG image:', error);
    return new Response('Error generating image', { status: 500 });
  }
}; 