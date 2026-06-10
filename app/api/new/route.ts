import { NextResponse } from 'next/server';
import { getChatState, setChatState } from '@/lib/chatSession';

export async function POST() {
  try {
    // Get current style before clearing
    const currentSession = await getChatState();
    const currentStyle = currentSession.style || 'casual';

    // Clear session but preserve style preference
    await setChatState({
      style: currentStyle,
      history: [],
      turns: [],
      exchanges: 0,
      lastQuestion: '',
      userName: currentSession.userName || 'Ray',
    });

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
