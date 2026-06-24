import { NextRequest, NextResponse } from 'next/server';
import { getNextTopic, hasCompletedTopic } from '@/lib/db';
import { getServerSupabase } from '@/lib/supabase';
import { getSession as getAuthSession } from '@/lib/auth';
import { setChatState } from '@/lib/chatSession';
import { CoachStyle, CoachName, Topic } from '@/types';
import { generateSpeech, VOICES } from '@/services/tts';

/**
 * Extract plain text from opening field (handles various formats)
 */
function parseOpeningText(rawOpening: unknown): string {
  // If it's already an array (JSONB parsed by Supabase)
  if (Array.isArray(rawOpening) && rawOpening[0]?.opening) {
    return rawOpening[0].opening;
  }
  // If it's a JSON string
  if (typeof rawOpening === 'string' && rawOpening.trim().startsWith('[')) {
    try {
      const parsed = JSON.parse(rawOpening);
      if (Array.isArray(parsed) && parsed[0]?.opening) {
        return parsed[0].opening;
      }
      return rawOpening;
    } catch {
      return rawOpening;
    }
  }
  // Plain string
  if (typeof rawOpening === 'string') {
    return rawOpening;
  }
  return '';
}

/**
 * Pick a random opening line from a newline-separated list.
 * If there's only one line (or no newlines), returns it as-is.
 */
function pickRandomOpening(openingText: string): string {
  if (!openingText) return '';

  // Split by newlines, trim each line, and remove empty lines
  const lines = openingText
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length === 0) return '';
  if (lines.length === 1) return lines[0];

  // Pick one at random
  const randomIndex = Math.floor(Math.random() * lines.length);
  return lines[randomIndex];
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const style: CoachStyle = body.style === 'casual' ? 'casual' : 'clear';
    const requestedTopicId: number | undefined = body.topicId;
    const coachName: CoachName = style === 'casual' ? 'Dora' : 'Morgan';

    // Get user info for topic completion tracking
    const authSession = await getAuthSession();
    const userName = authSession?.name || 'Ray';

    let opening = 'Say anything to start chatting!';
    let topic: (Topic & { is_first_visit: boolean }) | null = null;
    let topicId: number | null = null;

    if (style === 'clear') {
      // Morgan — load topic (either requested or next in sequence)
      if (requestedTopicId) {
        // Load specific topic by ID (dashboard selection)
        const supabase = getServerSupabase();
        const { data: requestedTopic } = await supabase
          .from('eec_topics')
          .select('*')
          .eq('id', requestedTopicId)
          .single();

        if (requestedTopic) {
          // Check if user has completed this topic before
          const completed = await hasCompletedTopic(requestedTopicId, userName);
          topic = { ...requestedTopic, is_first_visit: !completed };
        }
      }

      // Fall back to auto-advance if no specific topic requested or found
      if (!topic) {
        topic = await getNextTopic(userName);
      }

      console.log('DEBUG topic:', JSON.stringify(topic, null, 2));

      if (topic) {
        topicId = topic.id;
        const rawOpeningText = parseOpeningText(topic.opening);
        // Pick a random opening if there are multiple (newline-separated)
        const openingText = pickRandomOpening(rawOpeningText);

        // Opening is always just the chosen opening line.
        // The topic intro is delivered separately as intro_script audio — never glue it on.
        opening = openingText || 'Say anything to start chatting!';
      }
    }

    console.log('DEBUG opening to return:', opening);

    // Reset session with new style (server-side, no size limit)
    // Parse coach_views if it's in JSONB array format
    let coachViewsStr = '';
    if (topic?.coach_views) {
      if (Array.isArray(topic.coach_views) && topic.coach_views[0]?.coach_views) {
        coachViewsStr = topic.coach_views[0].coach_views;
      } else if (typeof topic.coach_views === 'string') {
        if (topic.coach_views.trim().startsWith('[')) {
          try {
            const parsed = JSON.parse(topic.coach_views);
            if (Array.isArray(parsed) && parsed[0]?.coach_views) {
              coachViewsStr = parsed[0].coach_views;
            } else {
              coachViewsStr = topic.coach_views;
            }
          } catch {
            coachViewsStr = topic.coach_views;
          }
        } else {
          coachViewsStr = topic.coach_views;
        }
      }
    }

    await setChatState({
      style,
      history: [],
      turns: [],
      exchanges: 0,
      lastQuestion: opening,
      userName,
      topic: topic
        ? {
            id: topic.id,
            name: topic.name,
            level: topic.level,
            vocabulary_pool: topic.vocabulary_pool,
            coach_views: coachViewsStr,
            focus_keyword: topic.focus_keyword,
          }
        : undefined,
      topicId: topicId || undefined,
    });

    // Generate audio - intro_script (if Morgan) and opening
    const voiceConfig = style === 'casual' ? VOICES.dora : VOICES.morgan;

    // For Morgan, generate intro_script audio and opening audio in parallel
    let introScriptAudio = '';
    let openingAudio = '';

    if (style === 'clear' && topic?.intro_script) {
      // Generate both in parallel for speed
      const [introAudio, opAudio] = await Promise.all([
        generateSpeech({ text: topic.intro_script, ...voiceConfig }),
        generateSpeech({ text: opening, ...voiceConfig }),
      ]);
      introScriptAudio = introAudio;
      openingAudio = opAudio;
    } else {
      // Dora or no intro_script - just generate opening
      openingAudio = await generateSpeech({
        text: opening,
        ...voiceConfig,
      });
    }

    return NextResponse.json({
      style,
      coach_name: coachName,
      opening,
      opening_audio: openingAudio,
      intro_script: topic?.intro_script || '',
      intro_script_audio: introScriptAudio,
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
