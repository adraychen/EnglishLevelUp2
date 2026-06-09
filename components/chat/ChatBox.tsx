'use client';

import React, { useEffect, useRef } from 'react';
import { ChatMessage } from '@/types';
import { ChatBubble } from './ChatBubble';

interface ChatBoxProps {
  messages: ChatMessage[];
  isLoading?: boolean;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ messages, isLoading }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  return (
    <div
      ref={containerRef}
      className="flex-1 overflow-y-auto p-4 space-y-3 chat-scroll"
    >
      {messages.map((message) => (
        <ChatBubble key={message.id} message={message} />
      ))}

      {isLoading && (
        <div className="flex flex-col items-start">
          <div className="px-4 py-3 rounded-2xl rounded-tl-sm bg-slate-100">
            <div className="flex items-center gap-1">
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
