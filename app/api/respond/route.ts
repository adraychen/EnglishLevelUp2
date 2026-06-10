import { NextRequest, NextResponse } from 'next/server';
import { getChatResponse } from '@/services/coach';
import { correctSentence } from '@/services/correction';
import { generateSpeech, VOICES } from '@/services/tts';
import { getChatState, setChatState } from '@/lib/chatSession';
import { ChatMessage, PracticeTurn, Topic, CoachStyle } from '@/types';

const MAX_EXCHANGES = 6;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const studentText = (body.text || '').trim();

    if (!studentText) {
      return NextResponse.json({ error: 'No text provided' }, { status: 400 });
    }

    // Get session from database
    const session = await getChatState();
    const style: CoachStyle = (session.style as CoachStyle) || 'casual';
    const history: ChatMessage[] = (session.history || []).map((m) => ({
      id: crypto.randomUUID(),
      role: m.role as 'coach' | 'student',
      content: m.content,
    }));
    // Topic is stored with only fields needed for chat (slim version)
    const topic = session.topic as Topic | null;
    const turns: PracticeTurn[] = session.turns || [];
    let exchanges: number = session.exchanges || 0;
    const lastQuestion: string = session.lastQuestion || '';

    let reply: string;
    let sessionComplete = false;

    if (style === 'clear' && topic) {
      // Morgan — topic-led practice
      exchanges += 1;
      const isClosing = exchanges >= MAX_EXCHANGES;

      console.log('DEBUG exchanges:', exchanges, 'MAX:', MAX_EXCHANGES, 'isClosing:', isClosing);

      reply = await getChatResponse({
        studentText,
        history,
        style,
        topic,
        isClosing,
      });

      // Run correction and TTS in parallel for faster response
      const voiceConfig = VOICES.morgan;
      const [corrected, replyAudio] = await Promise.all([
        correctSentence(studentText),
        generateSpeech({ text: reply, ...voiceConfig }),
      ]);

      // Store turn for review/practice
      turns.push({
        morgan: lastQuestion,
        student: studentText,
        corrected,
        reply,
      });

      sessionComplete = isClosing;

      // Update history
      history.push({ id: crypto.randomUUID(), role: 'student', content: studentText });
      history.push({ id: crypto.randomUUID(), role: 'coach', content: reply });

      // Keep last 12 messages for context
      const trimmedHistory = history.slice(-12).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Save session to database
      await setChatState({
        ...session,
        history: trimmedHistory,
        turns,
        exchanges,
        lastQuestion: reply,
      });

      console.log('DEBUG respond exchanges:', exchanges, 'sessionComplete:', sessionComplete);

      return NextResponse.json({
        reply,
        reply_audio: replyAudio,
        session_complete: sessionComplete,
      });
    } else {
      // Dora — free chat
      reply = await getChatResponse({
        studentText,
        history,
        style,
      });
    }

    // Update history (for Dora)
    history.push({ id: crypto.randomUUID(), role: 'student', content: studentText });
    history.push({ id: crypto.randomUUID(), role: 'coach', content: reply });

    // Keep last 12 messages for context
    const trimmedHistory = history.slice(-12).map((m) => ({
      role: m.role,
      content: m.content,
    }));

    // Save session to database
    await setChatState({
      ...session,
      history: trimmedHistory,
      turns,
      exchanges,
      lastQuestion: reply,
    });

    // Generate TTS audio for Dora
    const voiceConfig = VOICES.dora;
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
