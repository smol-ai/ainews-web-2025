import { Resend } from 'resend';
import type { APIRoute } from 'astro';

export const prerender = false;

export const POST: APIRoute = async ({ request }) => {
  try {
    const resend = new Resend(import.meta.env.RESEND_API_KEY);
    const data = await request.json();
    
    const { email, firstName, lastName } = data;
    
    // Log incoming request with timestamp
    console.log(`[INFO ${new Date().toISOString()}] Subscribe request received for email: ${email}`);
    
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
    
    // Disallow @qq.com emails from signup
    if (email.toLowerCase().endsWith('@qq.com')) {
      console.warn(`[WARN ${new Date().toISOString()}] Disallowed email domain attempted: ${email}`);
      return new Response(
        JSON.stringify({
          success: false,
          message: 'This email unfortunately does not meet our requirements. Please use a work email.'
        }),
        { status: 400 }
      );
    }
    
    // Log creating contact
    console.log(`[INFO ${new Date().toISOString()}] Creating contact for email: ${email}`);
    
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
    
    // Log successful subscription
    console.log(`[INFO ${new Date().toISOString()}] Subscription successful for email: ${email}`);
    
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