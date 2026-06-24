'use client';

import React, { Suspense, useState, useEffect, useCallback, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { ChatBox, ChatInput, CoachToggle } from '@/components/chat';
import { useSession, useChat, useAudioRecorder, useAudioPlayer, SessionAudio } from '@/hooks';
import { CoachStyle, PracticeTurn } from '@/types';
import { ReviewModal } from '@/components/review/ReviewModal';

function ChatContent() {
  const searchParams = useSearchParams();
  const initialStyle = (searchParams.get('style') as CoachStyle) || 'casual';
  const topicIdParam = searchParams.get('topicId');
  const initialTopicId = topicIdParam ? parseInt(topicIdParam, 10) : undefined;

  const { session, switchStyle, resetSession } = useSession(initialStyle);
  const { messages, isLoading, sendMessage, getSummary, clearMessages, addCoachMessage } = useChat({
    coachName: session.coachName,
  });

  const { isRecording, isTranscribing, startRecording, stopRecording } = useAudioRecorder();
  const { playMp3, playChunked } = useAudioPlayer();

  const [showReview, setShowReview] = useState(false);
  const [reviewContent, setReviewContent] = useState('');
  const [practiceTurns, setPracticeTurns] = useState<PracticeTurn[]>([]);
  const [isLoadingReview, setIsLoadingReview] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [pendingAudio, setPendingAudio] = useState<SessionAudio | null>(null);

  // Guard to prevent double opening
  const openingAddedRef = useRef(false);

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      openingAddedRef.current = false; // Reset guard on init
      const audio = await switchStyle(initialStyle, initialTopicId);
      setPendingAudio(audio);
      setInitialized(true);
    };
    initSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Play intro audio first (audio only, no text), then show opening bubble + play its audio
  useEffect(() => {
    if (!initialized || !session.opening || session.isLoading || openingAddedRef.current) {
      return;
    }

    // Mark as added immediately to prevent double-fire
    openingAddedRef.current = true;

    const playIntroThenOpening = async () => {
      const audio = pendingAudio;
      setPendingAudio(null);

      // Step 1: Play intro_script audio (audio only, no text on screen)
      if (audio?.introScriptAudio) {
        try {
          await playMp3(audio.introScriptAudio);
        } catch {
          // Intro audio failed - continue to opening
        }
      }

      // Step 2: Show opening bubble and play its audio
      addCoachMessage(session.opening);
      if (audio?.openingAudio) {
        try {
          await playMp3(audio.openingAudio);
        } catch {
          // Opening audio failed - bubble is already shown
        }
      }
    };

    playIntroThenOpening();
  }, [initialized, session.opening, session.isLoading, addCoachMessage, pendingAudio, playMp3]);

  const handleEndConversation = useCallback(async () => {
    setShowReview(true);
    setIsLoadingReview(true);

    const summary = await getSummary();
    if (summary) {
      setReviewContent(summary.summary);
      setPracticeTurns(summary.practiceTurns);
    } else {
      setReviewContent('Unable to generate review. Please try again.');
    }
    setIsLoadingReview(false);
  }, [getSummary]);

  const handleSend = useCallback(
    async (text: string) => {
      const result = await sendMessage(text);
      if (result?.reply) {
        if (result.sessionComplete) {
          // Closing turn: wait for audio to finish, then show modal
          try {
            await playChunked(result.reply);
          } catch {
            // Audio failed - continue to modal anyway
          }
          // Small delay for smoother transition
          await new Promise((resolve) => setTimeout(resolve, 300));
          handleEndConversation();
        } else {
          // Normal turn: play audio without waiting
          playChunked(result.reply);
        }
      }
    },
    [sendMessage, playChunked, handleEndConversation]
  );

  const handleStartRecording = useCallback(async () => {
    try {
      await startRecording();
    } catch (error) {
      console.error('Could not start recording:', error);
    }
  }, [startRecording]);

  const handleStopRecording = useCallback(async () => {
    const transcription = await stopRecording();
    if (transcription) {
      handleSend(transcription);
    }
  }, [stopRecording, handleSend]);

  const handleSwitchStyle = useCallback(
    async (style: CoachStyle) => {
      openingAddedRef.current = false; // Reset guard for new session
      clearMessages();
      const audio = await switchStyle(style);
      setPendingAudio(audio);
      // Opening will be added by the useEffect watching session.opening
    },
    [switchStyle, clearMessages]
  );

  const handleCloseReview = useCallback(() => {
    setShowReview(false);
  }, []);

  const handleNewConversation = useCallback(async () => {
    setShowReview(false);
    setReviewContent('');
    setPracticeTurns([]);
    openingAddedRef.current = false; // Reset guard for new session
    clearMessages();
    const audio = await resetSession();
    setPendingAudio(audio);
    // Opening will be added by the useEffect watching session.opening
  }, [resetSession, clearMessages]);

  const handleStartPractice = useCallback(() => {
    if (practiceTurns.length > 0) {
      sessionStorage.setItem('practiceTurns', JSON.stringify(practiceTurns));
      window.location.href = '/practice';
    }
  }, [practiceTurns]);

  const inputDisabled = isLoading || session.isLoading;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <nav className="bg-white border-b border-slate-200 px-4 py-3">
        <div className="max-w-2xl mx-auto flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-slate-600 hover:text-slate-800 text-sm">
              Dashboard
            </Link>
            <span className="font-semibold text-slate-800">
              English Conversation Coach
            </span>
          </div>
          <CoachToggle
            currentStyle={session.style}
            coachName={session.coachName}
            onSwitch={handleSwitchStyle}
            disabled={isLoading || isRecording}
          />
        </div>
      </nav>

      {/* Chat container */}
      <div className="flex-1 flex flex-col max-w-2xl mx-auto w-full bg-white border-x border-slate-200">
        <ChatBox messages={messages} isLoading={isLoading} />

        <ChatInput
          onSend={handleSend}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          isRecording={isRecording}
          isTranscribing={isTranscribing}
          disabled={inputDisabled}
        />

        {/* End conversation button */}
        <div className="px-4 pb-4">
          <button
            onClick={handleEndConversation}
            disabled={messages.length < 2 || isLoading}
            className="w-full py-3 bg-green-600 text-white rounded-xl text-sm font-medium hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            End conversation & see review
          </button>
        </div>
      </div>

      {/* Review modal */}
      <ReviewModal
        isOpen={showReview}
        content={reviewContent}
        isLoading={isLoadingReview}
        hasPracticeTurns={practiceTurns.length > 0}
        onClose={handleCloseReview}
        onNewConversation={handleNewConversation}
        onStartPractice={handleStartPractice}
      />
    </div>
  );
}

function ChatLoading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="flex items-center gap-3">
        <span className="w-6 h-6 border-2 border-blue-600/30 border-t-blue-600 rounded-full animate-spin" />
        <span className="text-slate-600">Loading...</span>
      </div>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<ChatLoading />}>
      <ChatContent />
    </Suspense>
  );
}
