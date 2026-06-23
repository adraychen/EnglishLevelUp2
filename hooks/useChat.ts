'use client';

import { useState, useCallback } from 'react';
import { ChatMessage, CoachName, PracticeTurn } from '@/types';

export interface UseChatOptions {
  coachName: CoachName;
  initialMessage?: string;
}

export interface UseChatReturn {
  messages: ChatMessage[];
  isLoading: boolean;
  sessionComplete: boolean;
  sendMessage: (text: string) => Promise<{ reply: string; sessionComplete: boolean } | null>;
  getSummary: () => Promise<{ summary: string; practiceTurns: PracticeTurn[] } | null>;
  clearMessages: () => void;
  addCoachMessage: (content: string) => void;
}

export function useChat({ coachName, initialMessage }: UseChatOptions): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (initialMessage) {
      return [
        {
          id: crypto.randomUUID(),
          role: 'coach',
          content: initialMessage,
          coachName,
        },
      ];
    }
    return [];
  });

  const [isLoading, setIsLoading] = useState(false);
  const [sessionComplete, setSessionComplete] = useState(false);

  const sendMessage = useCallback(
    async (text: string): Promise<{ reply: string; sessionComplete: boolean } | null> => {
      const trimmed = text.trim();
      if (!trimmed || isLoading) return null;

      // Add student message
      const studentMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'student',
        content: trimmed,
      };
      setMessages((prev) => [...prev, studentMessage]);
      setIsLoading(true);

      try {
        const res = await fetch('/api/respond', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: trimmed }),
        });

        if (!res.ok) {
          throw new Error('Response failed');
        }

        const data = await res.json();

        // Add coach reply
        const coachMessage: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'coach',
          content: data.reply,
          coachName,
        };
        setMessages((prev) => [...prev, coachMessage]);

        if (data.session_complete) {
          setSessionComplete(true);
        }

        // Return reply text and session completion status
        return { reply: data.reply, sessionComplete: data.session_complete || false };
      } catch (error) {
        console.error('Send message error:', error);
        // Remove the student message on error
        setMessages((prev) => prev.filter((m) => m.id !== studentMessage.id));
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [coachName, isLoading]
  );

  const getSummary = useCallback(async (): Promise<{
    summary: string;
    practiceTurns: PracticeTurn[];
  } | null> => {
    try {
      const res = await fetch('/api/summary', { method: 'POST' });
      const data = await res.json();
      return {
        summary: data.summary || '',
        practiceTurns: data.practice_turns || [],
      };
    } catch (error) {
      console.error('Get summary error:', error);
      return null;
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setSessionComplete(false);
  }, []);

  const addCoachMessage = useCallback(
    (content: string) => {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'coach',
          content,
          coachName,
        },
      ]);
    },
    [coachName]
  );

  return {
    messages,
    isLoading,
    sessionComplete,
    sendMessage,
    getSummary,
    clearMessages,
    addCoachMessage,
  };
}
