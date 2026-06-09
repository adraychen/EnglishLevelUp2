'use client';

import React from 'react';
import { CoachStyle, CoachName } from '@/types';

interface CoachToggleProps {
  currentStyle: CoachStyle;
  coachName: CoachName;
  onSwitch: (style: CoachStyle) => void;
  disabled?: boolean;
}

export const CoachToggle: React.FC<CoachToggleProps> = ({
  currentStyle,
  coachName,
  onSwitch,
  disabled,
}) => {
  return (
    <div className="flex items-center gap-3 flex-wrap">
      <span className="text-xs text-slate-500">
        Coach: <strong className="text-slate-700">{coachName}</strong>
      </span>

      <div className="flex gap-1">
        <button
          onClick={() => onSwitch('casual')}
          disabled={disabled}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
            currentStyle === 'casual'
              ? 'bg-blue-600 text-white'
              : 'bg-transparent text-slate-600 border border-slate-300 hover:bg-slate-50'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          Dora
        </button>

        <button
          onClick={() => onSwitch('clear')}
          disabled={disabled}
          className={`px-3 py-1.5 rounded-full text-xs font-medium transition-all ${
            currentStyle === 'clear'
              ? 'bg-blue-600 text-white'
              : 'bg-transparent text-slate-600 border border-slate-300 hover:bg-slate-50'
          } disabled:opacity-50 disabled:cursor-not-allowed`}
        >
          Morgan
        </button>
      </div>
    </div>
  );
};
