'use client';

import { useState, useCallback, useRef } from 'react';

export interface UseAudioPlayerReturn {
  isPlaying: boolean;
  playMp3: (base64Audio: string) => Promise<void>;
  playChunked: (text: string) => Promise<void>;
  speak: (text: string) => Promise<void>;
  stop: () => void;
}

export function useAudioPlayer(): UseAudioPlayerReturn {
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isPlayingQueueRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

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

  // Play next audio in queue
  const playNextInQueue = useCallback(async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingQueueRef.current = false;
      setIsPlaying(false);
      return;
    }

    const nextAudio = audioQueueRef.current.shift()!;
    await playMp3(nextAudio);
    playNextInQueue();
  }, [playMp3]);

  // Chunked streaming playback - starts playing first chunk while fetching rest
  const playChunked = useCallback(async (text: string): Promise<void> => {
    if (!text) return;

    // Stop any current playback
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingQueueRef.current = false;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    return new Promise(async (resolve) => {
      try {
        const response = await fetch('/api/tts-stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          resolve();
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE messages
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;

            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'chunk' && data.audio) {
                // Add to queue
                audioQueueRef.current.push(data.audio);

                // Start playing if not already
                if (!isPlayingQueueRef.current) {
                  isPlayingQueueRef.current = true;
                  setIsPlaying(true);
                  playNextInQueue();
                }
              } else if (data.type === 'done') {
                // Wait for queue to finish
                const checkDone = () => {
                  if (!isPlayingQueueRef.current && audioQueueRef.current.length === 0) {
                    resolve();
                  } else {
                    setTimeout(checkDone, 100);
                  }
                };
                checkDone();
              } else if (data.type === 'error') {
                console.error('TTS stream error:', data.message);
                resolve();
              }
            } catch {
              // Skip invalid JSON
            }
          }
        }
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          console.error('Chunked playback error:', error);
        }
        resolve();
      }
    });
  }, [playNextInQueue]);

  const stop = useCallback(() => {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    audioQueueRef.current = [];
    isPlayingQueueRef.current = false;
    utteranceRef.current = null;
    setIsPlaying(false);
  }, []);

  return {
    isPlaying,
    playMp3,
    playChunked,
    speak,
    stop,
  };
}
