import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

/**
 * Generate TTS audio using Google Translate TTS.
 * Tries multiple URL patterns as fallbacks.
 */
async function generateTTS(text: string, lang: string = 'en'): Promise<string> {
  if (!text || text.length === 0) {
    return '';
  }

  // Truncate to prevent issues with long text
  const truncatedText = text.slice(0, 500);
  const encodedText = encodeURIComponent(truncatedText);

  // Try multiple URL patterns
  const urls = [
    `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodedText}&tl=${lang}&client=tw-ob&ttsspeed=1`,
    `https://translate.googleapis.com/translate_tts?ie=UTF-8&q=${encodedText}&tl=${lang}&client=gtx&ttsspeed=1`,
  ];

  for (const url of urls) {
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Referer': 'https://translate.google.com/',
        },
      });

      if (response.ok) {
        const arrayBuffer = await response.arrayBuffer();
        if (arrayBuffer.byteLength > 0) {
          return Buffer.from(arrayBuffer).toString('base64');
        }
      }
    } catch {
      // Try next URL
    }
  }

  console.log('TTS: All endpoints failed, returning empty audio');
  return '';
}

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

    // Use Canadian English for Dora (casual), US English for Morgan (clear)
    const lang = style === 'casual' ? 'en-ca' : 'en';

    const audio = await generateTTS(text, lang);

    return NextResponse.json({ audio });
  } catch (error) {
    console.error('TTS route error:', error);
    return NextResponse.json({ audio: '' });
  }
}
