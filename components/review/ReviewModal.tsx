'use client';

import React from 'react';
import { X } from 'lucide-react';

interface ReviewModalProps {
  isOpen: boolean;
  content: string;
  isLoading: boolean;
  hasPracticeTurns: boolean;
  onClose: () => void;
  onNewConversation: () => void;
  onStartPractice: () => void;
}

export const ReviewModal: React.FC<ReviewModalProps> = ({
  isOpen,
  content,
  isLoading,
  hasPracticeTurns,
  onClose,
  onNewConversation,
  onStartPractice,
}) => {
  if (!isOpen) return null;

  // Simple markdown-to-HTML conversion
  const renderContent = (markdown: string) => {
    return markdown
      .replace(/^### (.+)$/gm, '<h3 class="text-sm font-semibold text-slate-800 mt-4 mb-2">$1</h3>')
      .replace(/^## (.+)$/gm, '<h2 class="text-base font-semibold text-blue-600 mt-5 mb-2">$1</h2>')
      .replace(/^---$/gm, '<hr class="my-4 border-slate-200" />')
      .replace(/\*\*(.+?)\*\*/g, '<strong class="text-blue-600">$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br />');
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl max-w-xl w-full max-h-[80vh] flex flex-col shadow-xl">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200">
          <h2 className="text-lg font-semibold text-slate-800">
            Conversation Review
          </h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-slate-100 rounded-lg transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-slate-500" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <span className="w-6 h-6 border-2 border-blue-600/30 border-t-blue-600 rounded-full animate-spin" />
              <span className="ml-3 text-slate-600">Generating your review...</span>
            </div>
          ) : (
            <div
              className="text-sm leading-relaxed text-slate-700"
              dangerouslySetInnerHTML={{ __html: renderContent(content) }}
            />
          )}
        </div>

        {/* Footer */}
        <div className="flex flex-wrap gap-3 px-5 py-4 border-t border-slate-200">
          {hasPracticeTurns && !isLoading && (
            <button
              onClick={onStartPractice}
              className="px-5 py-2.5 bg-green-600 text-white rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors"
            >
              Shadowing practice
            </button>
          )}

          <button
            onClick={onNewConversation}
            disabled={isLoading}
            className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            New conversation
          </button>

          <button
            onClick={onClose}
            className="px-5 py-2.5 bg-slate-100 text-slate-700 rounded-lg text-sm font-medium hover:bg-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
