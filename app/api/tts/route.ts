import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

/**
 * Generate TTS audio for a single line using Google Translate TTS.
 * This is what gTTS uses under the hood.
 */
async function generateTTS(text: string, lang: string = 'en'): Promise<string> {
  if (!text || text.length === 0) {
    return '';
  }

  // Truncate to prevent issues with long text
  const truncatedText = text.slice(0, 500);

  // Google Translate TTS endpoint (same as what gTTS uses)
  const encodedText = encodeURIComponent(truncatedText);
  const url = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodedText}&tl=${lang}&client=tw-ob`;

  const maxRetries = 3;
  const retryDelay = 2000; // 2 seconds

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        },
      });

      if (!response.ok) {
        console.error(`TTS fetch failed (attempt ${attempt}/${maxRetries}):`, response.status);
        if (attempt < maxRetries) {
          await new Promise((r) => setTimeout(r, retryDelay));
          continue;
        }
        return '';
      }

      const arrayBuffer = await response.arrayBuffer();
      const base64 = Buffer.from(arrayBuffer).toString('base64');
      return base64;
    } catch (error) {
      console.error(`TTS error (attempt ${attempt}/${maxRetries}):`, error);
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, retryDelay));
        continue;
      }
      return '';
    }
  }

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
