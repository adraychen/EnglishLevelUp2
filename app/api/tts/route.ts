import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { generateSpeech, VOICES } from '@/services/tts';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const text = (body.text || '').trim();

    if (!text) {
      return NextResponse.json({ error: 'No text provided' }, { status: 400 });
    }

    // Get style from session to determine voice
    const cookieStore = await cookies();
    const sessionCookie = cookieStore.get('session');
    let style = 'clear';

    if (sessionCookie) {
      try {
        const session = JSON.parse(sessionCookie.value);
        style = session.style || 'clear';
      } catch {
        // Use default
      }
    }

    // Use different voices for each coach
    const voiceConfig = style === 'casual' ? VOICES.dora : VOICES.morgan;

    const audio = await generateSpeech({
      text,
      ...voiceConfig,
    });

    return NextResponse.json({ audio });
  } catch (error) {
    console.error('TTS route error:', error);
    return NextResponse.json({ audio: '' });
  }
}
