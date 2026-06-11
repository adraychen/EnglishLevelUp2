'use client';

import React, { Suspense, useState, useEffect, useCallback } from 'react';
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
  const { messages, isLoading, sessionComplete, sendMessage, getSummary, clearMessages, addCoachMessage } = useChat({
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

  // Initialize session on mount
  useEffect(() => {
    const initSession = async () => {
      const audio = await switchStyle(initialStyle, initialTopicId);
      setPendingAudio(audio);
      setInitialized(true);
    };
    initSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Add opening message when session opening changes and no messages exist
  // Play intro_script audio first (if present), then opening audio
  useEffect(() => {
    if (initialized && session.opening && messages.length === 0 && !session.isLoading) {
      addCoachMessage(session.opening);
      if (pendingAudio) {
        const playSequentially = async () => {
          // Play intro_script audio first (if present)
          if (pendingAudio.introScriptAudio) {
            await playMp3(pendingAudio.introScriptAudio);
          }
          // Then play opening audio
          if (pendingAudio.openingAudio) {
            await playMp3(pendingAudio.openingAudio);
          }
        };
        playSequentially();
        setPendingAudio(null);
      }
    }
  }, [initialized, session.opening, session.isLoading, messages.length, addCoachMessage, pendingAudio, playMp3]);

  // Handle session complete
  useEffect(() => {
    if (sessionComplete) {
      addCoachMessage("That's a great place to pause! Let's look at what we covered today.");
      setTimeout(() => {
        handleEndConversation();
      }, 1000);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionComplete]);

  const handleSend = useCallback(
    async (text: string) => {
      const result = await sendMessage(text);
      if (result?.reply) {
        // Stream TTS audio in chunks for faster playback
        playChunked(result.reply);
      }
    },
    [sendMessage, playChunked]
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
      clearMessages();
      const audio = await switchStyle(style);
      setPendingAudio(audio);
      // Opening will be added by the useEffect watching session.opening
    },
    [switchStyle, clearMessages]
  );

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

  const handleCloseReview = useCallback(() => {
    setShowReview(false);
  }, []);

  const handleNewConversation = useCallback(async () => {
    setShowReview(false);
    setReviewContent('');
    setPracticeTurns([]);
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
