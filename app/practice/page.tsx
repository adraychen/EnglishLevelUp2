'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { PracticeTurn } from '@/types';
import { ShadowingPractice } from '@/components/practice';

export default function PracticePage() {
  const router = useRouter();
  const [turns, setTurns] = useState<PracticeTurn[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load practice turns from sessionStorage on mount
  useEffect(() => {
    try {
      const stored = sessionStorage.getItem('practiceTurns');
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setTurns(parsed);
        } else {
          setError('No practice data found');
        }
      } else {
        setError('No practice data found');
      }
    } catch (e) {
      console.error('Error loading practice turns:', e);
      setError('Could not load practice data');
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleComplete = useCallback(() => {
    // Clear the stored turns
    sessionStorage.removeItem('practiceTurns');
    // Go back to chat
    router.push('/chat');
  }, [router]);

  const handleExit = useCallback(() => {
    // Keep the turns in case they want to try again
    router.push('/chat');
  }, [router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="flex items-center gap-3">
          <span className="w-6 h-6 border-2 border-blue-600/30 border-t-blue-600 rounded-full animate-spin" />
          <span className="text-slate-600">Loading practice...</span>
        </div>
      </div>
    );
  }

  if (error || turns.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-4">
        <div className="max-w-sm w-full text-center space-y-4">
          <div className="text-4xl">📝</div>
          <h1 className="text-xl font-bold text-slate-800">No Practice Available</h1>
          <p className="text-slate-600">
            {error || 'Complete a conversation with Morgan first to unlock shadowing practice.'}
          </p>
          <button
            onClick={() => router.push('/chat?style=clear')}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
          >
            Start a conversation
          </button>
        </div>
      </div>
    );
  }

  return (
    <ShadowingPractice
      turns={turns}
      onComplete={handleComplete}
      onExit={handleExit}
    />
  );
}
