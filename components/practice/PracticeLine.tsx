'use client';

import React from 'react';
import { Play, Mic, Square, Check, Volume2 } from 'lucide-react';

export type LineType = 'morgan' | 'student';

interface PracticeLineProps {
  type: LineType;
  text: string;
  originalText?: string; // For student lines, show "you said:" if different
  isActive: boolean;
  isCompleted: boolean;
  isPlaying?: boolean;
  isRecording?: boolean;
  onPlay?: () => void;
  onRecord?: () => void;
  onStopRecording?: () => void;
}

export const PracticeLine: React.FC<PracticeLineProps> = ({
  type,
  text,
  originalText,
  isActive,
  isCompleted,
  isPlaying,
  isRecording,
  onPlay,
  onRecord,
  onStopRecording,
}) => {
  const isMorgan = type === 'morgan';
  const showOriginal = originalText && originalText.toLowerCase() !== text.toLowerCase();

  return (
    <div
      className={`p-4 rounded-xl transition-all duration-200 border-2 ${
        isActive
          ? 'border-blue-500 bg-blue-50 shadow-sm'
          : isCompleted
          ? 'border-green-200 bg-green-50/50'
          : 'border-transparent bg-slate-50'
      }`}
    >
      {/* Speaker label */}
      <div className="flex items-center justify-between mb-2">
        <span
          className={`text-xs font-bold uppercase ${
            isActive
              ? 'text-blue-600'
              : isCompleted
              ? 'text-green-600'
              : 'text-slate-400'
          }`}
        >
          {isMorgan ? 'Morgan' : 'You'}
        </span>

        {isCompleted && (
          <Check className="w-4 h-4 text-green-500" />
        )}
      </div>

      {/* Main text */}
      <p
        className={`text-base leading-relaxed ${
          isActive ? 'text-slate-900 font-medium' : 'text-slate-600'
        }`}
      >
        {text}
      </p>

      {/* Original text (for corrected student lines) */}
      {showOriginal && (
        <p className="text-xs text-slate-400 mt-2 italic">
          You said: "{originalText}"
        </p>
      )}

      {/* Action buttons (only for active line) */}
      {isActive && (
        <div className="mt-4 flex items-center gap-3">
          {isMorgan ? (
            // Morgan line: play button
            <button
              onClick={onPlay}
              disabled={isPlaying}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                isPlaying
                  ? 'bg-blue-100 text-blue-600'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isPlaying ? (
                <>
                  <Volume2 className="w-4 h-4 animate-pulse" />
                  Playing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Listen
                </>
              )}
            </button>
          ) : (
            // Student line: record button
            <button
              onClick={isRecording ? onStopRecording : onRecord}
              className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
                isRecording
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isRecording ? (
                <>
                  <Square className="w-4 h-4 fill-current" />
                  Stop
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4" />
                  Practice
                </>
              )}
            </button>
          )}
        </div>
      )}
    </div>
  );
};
