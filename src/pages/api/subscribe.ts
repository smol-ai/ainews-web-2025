import { Resend } from 'resend';
import type { APIRoute } from 'astro';

export const prerender = false;

export const POST: APIRoute = async ({ request }) => {
  try {
    const resend = new Resend(import.meta.env.RESEND_API_KEY);
    const data = await request.json();
    
    const { email, firstName, lastName } = data;
    
    // Validate required fields
    if (!email) {
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: 'Email is required' 
        }),
        { status: 400 }
      );
    }
    
    // Create contact in Resend
    const response = await resend.contacts.create({
      email,
      firstName: firstName || '',
      lastName: lastName || '',
      unsubscribed: false,
      audienceId: import.meta.env.RESEND_AUDIENCE_ID,
    });
    
    if (response.error) {
      console.error('Error creating contact:', response.error);
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: response.error.message 
        }),
        { status: 400 }
      );
    }
    
    return new Response(
      JSON.stringify({ 
        success: true, 
        message: 'Subscription successful' 
      }),
      { status: 200 }
    );
  } catch (error) {
    console.error('Error in subscribe API:', error);
    return new Response(
      JSON.stringify({ 
        success: false, 
        message: 'An error occurred while processing your request' 
      }),
      { status: 500 }
    );
  }
} 