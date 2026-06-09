'use client';

import React, { useState, useCallback } from 'react';
import { Mic, Square, Send } from 'lucide-react';

interface ChatInputProps {
  onSend: (text: string) => void;
  onStartRecording: () => void;
  onStopRecording: () => void;
  isRecording: boolean;
  isTranscribing: boolean;
  disabled?: boolean;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  onStartRecording,
  onStopRecording,
  isRecording,
  isTranscribing,
  disabled,
}) => {
  const [text, setText] = useState('');

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (trimmed) {
      onSend(trimmed);
      setText('');
    }
  }, [text, onSend]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend]
  );

  const handleMicClick = useCallback(() => {
    if (isRecording) {
      onStopRecording();
    } else {
      onStartRecording();
    }
  }, [isRecording, onStartRecording, onStopRecording]);

  const isDisabled = disabled || isTranscribing;

  return (
    <div className="p-4 bg-white border-t border-slate-200">
      <div className="flex items-center gap-3">
        {/* Mic button */}
        <button
          onClick={handleMicClick}
          disabled={isDisabled}
          className={`w-12 h-12 rounded-full flex items-center justify-center transition-all flex-shrink-0 ${
            isRecording
              ? 'bg-red-500 text-white animate-pulse-recording'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          aria-label={isRecording ? 'Stop recording' : 'Start recording'}
        >
          {isTranscribing ? (
            <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : isRecording ? (
            <Square className="w-5 h-5 fill-current" />
          ) : (
            <Mic className="w-5 h-5" />
          )}
        </button>

        {/* Text input */}
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your reply here..."
          disabled={isDisabled || isRecording}
          className="flex-1 px-4 py-3 border border-slate-300 rounded-xl text-sm outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:bg-slate-50 disabled:cursor-not-allowed"
        />

        {/* Send button */}
        <button
          onClick={handleSend}
          disabled={isDisabled || !text.trim() || isRecording}
          className="px-5 py-3 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
        >
          <Send className="w-5 h-5" />
        </button>
      </div>

      <p className="text-xs text-slate-500 text-center mt-2">
        {isRecording
          ? 'Recording... tap to stop'
          : isTranscribing
          ? 'Transcribing...'
          : 'Tap mic to speak, or type your reply'}
      </p>
    </div>
  );
};
