'use client';

import React from 'react';
import { Mic, Square } from 'lucide-react';

interface RecordButtonProps {
  isRecording: boolean;
  onToggle: () => void;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const RecordButton: React.FC<RecordButtonProps> = ({
  isRecording,
  onToggle,
  disabled,
  size = 'lg'
}) => {
  const sizes = {
    sm: 'w-10 h-10',
    md: 'w-14 h-14',
    lg: 'w-20 h-20',
  };

  const iconSizes = {
    sm: 'w-5 h-5',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
  };

  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative ${sizes[size]} rounded-full flex items-center justify-center transition-all duration-300 ${
        isRecording
          ? 'bg-red-500 text-white ring-2 ring-red-300 animate-pulse'
          : 'bg-blue-600 text-white hover:bg-blue-700'
      } disabled:opacity-50 disabled:grayscale`}
    >
      {isRecording ? (
        <Square className={`${iconSizes[size]} fill-current`} />
      ) : (
        <Mic className={iconSizes[size]} />
      )}
    </button>
  );
};
