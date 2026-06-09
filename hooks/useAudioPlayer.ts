'use client';

import { useState, useCallback, useRef } from 'react';

export interface UseAudioPlayerReturn {
  isPlaying: boolean;
  playMp3: (base64Audio: string) => Promise<void>;
  stop: () => void;
}

export function useAudioPlayer(): UseAudioPlayerReturn {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const playMp3 = useCallback(async (base64Audio: string): Promise<void> => {
    if (!base64Audio) return;

    return new Promise((resolve) => {
      // Stop any currently playing audio
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
      audioRef.current = audio;

      audio.onplay = () => setIsPlaying(true);
      audio.onended = () => {
        setIsPlaying(false);
        audioRef.current = null;
        resolve();
      };
      audio.onerror = () => {
        setIsPlaying(false);
        audioRef.current = null;
        resolve();
      };

      audio.play().catch(() => {
        setIsPlaying(false);
        resolve();
      });
    });
  }, []);

  const stop = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
      setIsPlaying(false);
    }
  }, []);

  return {
    isPlaying,
    playMp3,
    stop,
  };
}
