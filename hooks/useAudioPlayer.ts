'use client';

import { useState, useCallback, useRef } from 'react';

export interface UseAudioPlayerReturn {
  isPlaying: boolean;
  playMp3: (base64Audio: string) => Promise<void>;
  speak: (text: string) => Promise<void>;
  stop: () => void;
}

export function useAudioPlayer(): UseAudioPlayerReturn {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  // Browser-based TTS using Web Speech API
  const speak = useCallback(async (text: string): Promise<void> => {
    if (!text || typeof window === 'undefined' || !window.speechSynthesis) {
      return;
    }

    return new Promise((resolve) => {
      // Stop any currently playing audio/speech
      window.speechSynthesis.cancel();
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utteranceRef.current = utterance;

      // Try to find a good English voice
      const voices = window.speechSynthesis.getVoices();
      const englishVoice = voices.find(
        (v) => v.lang.startsWith('en') && v.name.includes('Female')
      ) || voices.find(
        (v) => v.lang.startsWith('en-US')
      ) || voices.find(
        (v) => v.lang.startsWith('en')
      );

      if (englishVoice) {
        utterance.voice = englishVoice;
      }

      utterance.rate = 0.9; // Slightly slower for learners
      utterance.pitch = 1;
      utterance.volume = 1;

      utterance.onstart = () => setIsPlaying(true);
      utterance.onend = () => {
        setIsPlaying(false);
        utteranceRef.current = null;
        resolve();
      };
      utterance.onerror = () => {
        setIsPlaying(false);
        utteranceRef.current = null;
        resolve();
      };

      window.speechSynthesis.speak(utterance);
    });
  }, []);

  const playMp3 = useCallback(async (base64Audio: string): Promise<void> => {
    if (!base64Audio) return;

    return new Promise((resolve) => {
      // Stop any currently playing audio/speech
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
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
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    utteranceRef.current = null;
    setIsPlaying(false);
  }, []);

  return {
    isPlaying,
    playMp3,
    speak,
    stop,
  };
}
