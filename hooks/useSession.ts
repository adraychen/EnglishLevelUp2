'use client';

import { useState, useCallback } from 'react';
import { CoachStyle, CoachName, Topic } from '@/types';

export interface SessionState {
  style: CoachStyle;
  coachName: CoachName;
  opening: string;
  introScript: string;
  topic: Topic | null;
  isLoading: boolean;
}

export interface SessionAudio {
  introScriptAudio: string;
  openingAudio: string;
}

const DEFAULT_SESSION: SessionState = {
  style: 'casual',
  coachName: 'Dora',
  opening: 'Say anything to start chatting!',
  introScript: '',
  topic: null,
  isLoading: false,
};

export function useSession(initialStyle: CoachStyle = 'casual') {
  const [session, setSession] = useState<SessionState>({
    ...DEFAULT_SESSION,
    style: initialStyle,
    coachName: initialStyle === 'casual' ? 'Dora' : 'Morgan',
  });

  const switchStyle = useCallback(async (newStyle: CoachStyle, topicId?: number): Promise<SessionAudio | null> => {
    if (newStyle === session.style && !topicId) return null;

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
        introScript: data.intro_script || '',
        topic: null, // Topic is stored server-side
        isLoading: false,
      });

      // Return audio for intro_script and opening
      return {
        introScriptAudio: data.intro_script_audio || '',
        openingAudio: data.opening_audio || '',
      };
    } catch (error) {
      console.error('Switch style error:', error);
      setSession((prev) => ({
        ...prev,
        style: newStyle,
        coachName: newStyle === 'casual' ? 'Dora' : 'Morgan',
        isLoading: false,
      }));
      return null;
    }
  }, [session.style]);

  const resetSession = useCallback(async (): Promise<SessionAudio | null> => {
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
        introScript: data.intro_script || '',
        topic: null,
        isLoading: false,
      });

      return {
        introScriptAudio: data.intro_script_audio || '',
        openingAudio: data.opening_audio || '',
      };
    } catch (error) {
      console.error('Reset session error:', error);
      setSession((prev) => ({ ...prev, isLoading: false }));
      return null;
    }
  }, [session.style, session.coachName]);

  return {
    session,
    switchStyle,
    resetSession,
  };
}
