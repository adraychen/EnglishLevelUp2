import { NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { buildReview, wordsUsedInText } from '@/services/review';
import { logLearning } from '@/lib/db';
import { PracticeTurn, Topic, ChatMessage } from '@/types';

async function getSession() {
  const cookieStore = await cookies();
  const sessionCookie = cookieStore.get('session');
  if (sessionCookie) {
    try {
      return JSON.parse(sessionCookie.value);
    } catch {
      return null;
    }
  }
  return null;
}

export async function POST() {
  try {
    const session = (await getSession()) || {};
    const style: string = session.style || 'casual';
    const history: ChatMessage[] = session.history || [];
    const turns: PracticeTurn[] = session.turns || [];
    const topic: Topic | null = session.topic || null;
    const topicId: number | null = session.topicId || null;
    const topicName = topic?.name || '';
    const pool = topic?.vocabulary_pool || '';

    // Build the review markdown
    const summary = buildReview(turns, style, topicName);

    // Log vocabulary words Morgan used (for topic progression)
    if (style === 'clear' && pool && topicId) {
      const morganText = history
        .filter((m) => m.role === 'coach')
        .map((m) => m.content)
        .join(' ');

      const taughtWords = wordsUsedInText(morganText, pool);

      if (taughtWords.length > 0) {
        try {
          await logLearning(topicId, taughtWords, false);
        } catch (error) {
          console.error('Learning log error:', error);
        }
      }
    }

    // Return practice turns for the replay feature
    const practiceTurns: PracticeTurn[] = turns.map((t) => ({
      morgan: t.morgan || '',
      student: t.student || '',
      corrected: t.corrected || '',
      reply: t.reply || '',
    }));

    return NextResponse.json({
      summary,
      practice_turns: practiceTurns,
    });
  } catch (error) {
    console.error('Summary error:', error);
    return NextResponse.json(
      {
        summary:
          '**Sorry, the review could not be generated this time.** Please try again, or start a new one.',
        practice_turns: [],
      },
      { status: 200 }
    );
  }
}
