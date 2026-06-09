'use client';

import React from 'react';
import { ChatMessage, CoachName } from '@/types';

interface ChatBubbleProps {
  message: ChatMessage;
}

export const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isCoach = message.role === 'coach';

  return (
    <div className={`flex flex-col ${isCoach ? 'items-start' : 'items-end'}`}>
      {isCoach && message.coachName && (
        <span className="text-xs text-slate-500 mb-1 ml-1">
          {message.coachName}
        </span>
      )}
      <div
        className={`px-4 py-3 rounded-2xl max-w-[85%] text-sm leading-relaxed ${
          isCoach
            ? 'bg-slate-100 text-slate-800 rounded-tl-sm'
            : 'bg-blue-600 text-white rounded-tr-sm'
        }`}
      >
        {message.content}
      </div>
    </div>
  );
};
