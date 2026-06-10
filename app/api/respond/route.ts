import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { getChatResponse } from '@/services/coach';
import { correctSentence } from '@/services/correction';
import { generateSpeech, VOICES } from '@/services/tts';
import { ChatMessage, PracticeTurn, Topic, CoachStyle } from '@/types';

const MAX_EXCHANGES = 6;

// Simple session storage using cookies (JSON encoded)
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

async function setSession(data: Record<string, unknown>) {
  const cookieStore = await cookies();
  cookieStore.set('session', JSON.stringify(data), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 60 * 60 * 24, // 24 hours
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const studentText = (body.text || '').trim();

    if (!studentText) {
      return NextResponse.json({ error: 'No text provided' }, { status: 400 });
    }

    const session = (await getSession()) || {};
    const style: CoachStyle = session.style || 'casual';
    // History is stored compressed (no id) to save cookie space
    const storedHistory = session.history || [];
    const history: ChatMessage[] = storedHistory.map((m: { role: string; content: string }) => ({
      id: crypto.randomUUID(),
      role: m.role as 'coach' | 'student',
      content: m.content,
    }));
    const topic: Topic | null = session.topic || null;
    const turns: PracticeTurn[] = session.turns || [];
    let exchanges: number = session.exchanges || 0;
    const lastQuestion: string = session.lastQuestion || '';

    let reply: string;
    let sessionComplete = false;

    if (style === 'clear' && topic) {
      // Morgan — topic-led practice
      exchanges += 1;
      const isClosing = exchanges >= MAX_EXCHANGES;

      reply = await getChatResponse({
        studentText,
        history,
        style,
        topic,
        isClosing,
      });

      // Correct the student's sentence
      const corrected = await correctSentence(studentText);

      // Store turn for review/practice
      turns.push({
        morgan: lastQuestion,
        student: studentText,
        corrected,
        reply,
      });

      sessionComplete = isClosing;
    } else {
      // Dora — free chat
      reply = await getChatResponse({
        studentText,
        history,
        style,
      });
    }

    // Update history
    history.push({ id: crypto.randomUUID(), role: 'student', content: studentText });
    history.push({ id: crypto.randomUUID(), role: 'coach', content: reply });

    // Compress history to avoid cookie size limit (~4KB)
    // Only store role and content (no id), last 12 messages
    const compressedHistory = history.slice(-12).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Save session
    const newSession = {
      ...session,
      history: compressedHistory,
      turns,
      exchanges,
      lastQuestion: reply,
    };

    const sessionSize = JSON.stringify(newSession).length;
    console.log('DEBUG session size:', sessionSize, 'bytes, exchanges:', exchanges);

    if (sessionSize > 3500) {
      console.warn('WARNING: Session size approaching cookie limit (4KB), size:', sessionSize);
    }

    await setSession(newSession);

    console.log('DEBUG respond exchanges:', exchanges, 'sessionComplete:', sessionComplete);

    // Generate TTS audio in the same request (like Flask did)
    const voiceConfig = style === 'casual' ? VOICES.dora : VOICES.morgan;
    const replyAudio = await generateSpeech({
      text: reply,
      ...voiceConfig,
    });

    return NextResponse.json({
      reply,
      reply_audio: replyAudio,
      session_complete: sessionComplete,
    });
  } catch (error) {
    console.error('Respond error:', error);
    return NextResponse.json({ error: 'Server error' }, { status: 500 });
  }
}
