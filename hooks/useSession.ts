'use client';

import { useState, useCallback } from 'react';
import { CoachStyle, CoachName, Topic, PracticeTurn } from '@/types';

export interface SessionState {
  style: CoachStyle;
  coachName: CoachName;
  opening: string;
  topic: Topic | null;
  taughtWords: string[];
  isLoading: boolean;
}

const DEFAULT_SESSION: SessionState = {
  style: 'casual',
  coachName: 'Dora',
  opening: 'Say anything to start chatting!',
  topic: null,
  taughtWords: [],
  isLoading: false,
};

export function useSession(initialStyle: CoachStyle = 'casual') {
  const [session, setSession] = useState<SessionState>({
    ...DEFAULT_SESSION,
    style: initialStyle,
    coachName: initialStyle === 'casual' ? 'Dora' : 'Morgan',
  });

  const switchStyle = useCallback(async (newStyle: CoachStyle, topicId?: number) => {
    if (newStyle === session.style && !topicId) return;

    setSession((prev) => ({ ...prev, isLoading: true }));

    try {
      const res = await fetch('/api/set-style', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ style: newStyle, topicId }),
      });

      const data = await res.json();

      setSession({
        style: data.style || newStyle,
        coachName: data.coach_name || (newStyle === 'casual' ? 'Dora' : 'Morgan'),
        opening: data.opening || 'Say anything to start chatting!',
        topic: null, // Topic is stored server-side
        taughtWords: [],
        isLoading: false,
      });

      // Return audio for the opening (Morgan greets out loud)
      return data.opening_audio || '';
    } catch (error) {
      console.error('Switch style error:', error);
      setSession((prev) => ({
        ...prev,
        style: newStyle,
        coachName: newStyle === 'casual' ? 'Dora' : 'Morgan',
        isLoading: false,
      }));
      return '';
    }
  }, [session.style]);

  const resetSession = useCallback(async () => {
    setSession((prev) => ({ ...prev, isLoading: true }));

    try {
      await fetch('/api/new', { method: 'POST' });

      // Re-initialize with current style
      const res = await fetch('/api/set-style', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ style: session.style }),
      });

      const data = await res.json();

      setSession({
        style: data.style || session.style,
        coachName: data.coach_name || session.coachName,
        opening: data.opening || 'Say anything to start chatting!',
        topic: null,
        taughtWords: [],
        isLoading: false,
      });

      return data.opening_audio || '';
    } catch (error) {
      console.error('Reset session error:', error);
      setSession((prev) => ({ ...prev, isLoading: false }));
      return '';
    }
  }, [session.style, session.coachName]);

  return {
    session,
    switchStyle,
    resetSession,
  };
}
