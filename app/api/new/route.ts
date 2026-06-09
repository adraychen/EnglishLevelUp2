import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';

export async function POST() {
  try {
    const cookieStore = await cookies();

    // Get current style before clearing
    let currentStyle = 'casual';
    const sessionCookie = cookieStore.get('session');
    if (sessionCookie) {
      try {
        const session = JSON.parse(sessionCookie.value);
        currentStyle = session.style || 'casual';
      } catch {
        // Use default
      }
    }

    // Clear session but preserve style preference
    cookieStore.set(
      'session',
      JSON.stringify({
        style: currentStyle,
        history: [],
        turns: [],
        exchanges: 0,
        taughtWords: [],
        lastQuestion: '',
      }),
      {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'lax',
        maxAge: 60 * 60 * 24,
      }
    );

    return NextResponse.json({ redirect: '/' });
  } catch (error) {
    console.error('New conversation error:', error);
    return NextResponse.json({ redirect: '/' });
  }
}

// Also support GET for simple redirects
export async function GET() {
  return POST();
}
