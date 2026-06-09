import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { getNextTopic } from '@/lib/db';
import { CoachStyle, CoachName, Topic } from '@/types';

async function setSession(data: Record<string, unknown>) {
  const cookieStore = await cookies();
  cookieStore.set('session', JSON.stringify(data), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24, // 24 hours
  });
}

/**
 * Generate TTS audio (reuse logic from tts route)
 */
async function generateTTS(text: string, lang: string = 'en'): Promise<string> {
  if (!text || text.length === 0) {
    return '';
  }

  const truncatedText = text.slice(0, 500);
  const encodedText = encodeURIComponent(truncatedText);
  const url = `https://translate.google.com/translate_tts?ie=UTF-8&q=${encodedText}&tl=${lang}&client=tw-ob`;

  try {
    const response = await fetch(url, {
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
      },
    });

    if (!response.ok) {
      return '';
    }

    const arrayBuffer = await response.arrayBuffer();
    return Buffer.from(arrayBuffer).toString('base64');
  } catch {
    return '';
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const style: CoachStyle = body.style === 'casual' ? 'casual' : 'clear';
    const coachName: CoachName = style === 'casual' ? 'Dora' : 'Morgan';

    let opening = 'Say anything to start chatting!';
    let topic: (Topic & { already_taught: string[] }) | null = null;
    let topicId: number | null = null;
    let taughtWords: string[] = [];

    if (style === 'clear') {
      // Morgan — load topic
      topic = await getNextTopic();

      if (topic) {
        topicId = topic.id;
        taughtWords = topic.already_taught || [];

        if (taughtWords.length > 0) {
          // Continuing session
          const topicNameLower = (topic.name || '').toLowerCase();
          if (topicNameLower.includes('feeling')) {
            opening = `Welcome back! Last time we talked about feelings like ${taughtWords.slice(0, 3).join(', ')}. Today let's keep going with a few more. To start — how are you feeling right now?`;
          } else {
            opening = `Welcome back! Let's continue with ${topic.name || 'our topic'}. How are you doing today?`;
          }
        } else {
          // First session for this topic
          // Handle opening in various formats:
          // - Plain string: "Hello..."
          // - JSON array (parsed by Supabase): [{opening: "..."}]
          // - JSON string: '[{"opening": "..."}]'
          const rawOpening = topic.opening;
          let openingText = '';

          // If it's already an array (JSONB parsed by Supabase)
          if (Array.isArray(rawOpening) && rawOpening[0]?.opening) {
            openingText = rawOpening[0].opening;
          }
          // If it's a JSON string
          else if (typeof rawOpening === 'string' && rawOpening.trim().startsWith('[')) {
            try {
              const parsed = JSON.parse(rawOpening);
              if (Array.isArray(parsed) && parsed[0]?.opening) {
                openingText = parsed[0].opening;
              } else {
                openingText = rawOpening;
              }
            } catch {
              openingText = rawOpening;
            }
          }
          // Plain string
          else if (typeof rawOpening === 'string') {
            openingText = rawOpening;
          }
          opening = openingText || 'Say anything to start chatting!';
        }
      }
    }

    // Reset session with new style
    const sessionData: Record<string, unknown> = {
      style,
      history: [],
      turns: [],
      exchanges: 0,
      taughtWords,
      lastQuestion: '',
    };

    if (topic) {
      sessionData.topic = topic;
      sessionData.topicId = topicId;
    }

    await setSession(sessionData);

    // Generate opening audio for Morgan
    let openingAudio = '';
    if (style === 'clear') {
      openingAudio = await generateTTS(opening, 'en');
    }

    return NextResponse.json({
      style,
      coach_name: coachName,
      opening,
      opening_audio: openingAudio,
    });
  } catch (error) {
    console.error('Set style error:', error);
    return NextResponse.json(
      {
        style: 'casual',
        coach_name: 'Dora',
        opening: 'Say anything to start chatting!',
        opening_audio: '',
      },
      { status: 200 }
    );
  }
}
