'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { ArrowLeft, ArrowRight, X, CheckCircle } from 'lucide-react';
import { PracticeTurn } from '@/types';
import { PracticeLine, LineType } from './PracticeLine';
import { ProgressBar } from '@/components/ui';
import { useAudioRecorder, useAudioPlayer } from '@/hooks';

interface PracticeStep {
  type: LineType;
  text: string;
  originalText?: string;
}

interface ShadowingPracticeProps {
  turns: PracticeTurn[];
  onComplete: () => void;
  onExit: () => void;
}

export const ShadowingPractice: React.FC<ShadowingPracticeProps> = ({
  turns,
  onComplete,
  onExit,
}) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [completedIndices, setCompletedIndices] = useState<Set<number>>(new Set());
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);

  const { isRecording, startRecording, stopRecording, cancelRecording } = useAudioRecorder();
  const { isPlaying, playMp3, stop: stopAudio } = useAudioPlayer();

  // Flatten turns into a sequence of practice steps
  const steps = useMemo<PracticeStep[]>(() => {
    const result: PracticeStep[] = [];

    turns.forEach((turn, index) => {
      // First turn: include Morgan's opening line
      if (index === 0 && turn.morgan) {
        result.push({ type: 'morgan', text: turn.morgan });
      }

      // Student's corrected line
      if (turn.corrected) {
        result.push({
          type: 'student',
          text: turn.corrected,
          originalText: turn.student,
        });
      }

      // Morgan's reply
      if (turn.reply) {
        result.push({ type: 'morgan', text: turn.reply });
      }
    });

    return result;
  }, [turns]);

  const currentStep = steps[currentIndex];
  const isLastStep = currentIndex === steps.length - 1;
  const allCompleted = completedIndices.size === steps.length;

  // Fetch TTS for a line
  const fetchTTS = useCallback(async (text: string): Promise<string> => {
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      return data.audio || '';
    } catch {
      return '';
    }
  }, []);

  // Handle playing Morgan's line
  const handlePlayMorgan = useCallback(async () => {
    if (!currentStep || currentStep.type !== 'morgan' || isPlaying) return;

    setIsLoadingAudio(true);
    const audio = await fetchTTS(currentStep.text);
    setIsLoadingAudio(false);

    if (audio) {
      await playMp3(audio);
    }

    // Mark as completed after playing
    setCompletedIndices((prev) => new Set(prev).add(currentIndex));
  }, [currentStep, currentIndex, isPlaying, fetchTTS, playMp3]);

  // Handle recording student's line
  const handleStartRecording = useCallback(async () => {
    try {
      await startRecording();
    } catch (error) {
      console.error('Could not start recording:', error);
    }
  }, [startRecording]);

  const handleStopRecording = useCallback(async () => {
    await stopRecording();
    // Mark as completed after recording (we don't grade it)
    setCompletedIndices((prev) => new Set(prev).add(currentIndex));
  }, [stopRecording, currentIndex]);

  // Navigation
  const goNext = useCallback(() => {
    stopAudio();
    cancelRecording();

    if (isLastStep) {
      onComplete();
    } else {
      setCurrentIndex((prev) => Math.min(prev + 1, steps.length - 1));
    }
  }, [isLastStep, steps.length, stopAudio, cancelRecording, onComplete]);

  const goPrev = useCallback(() => {
    stopAudio();
    cancelRecording();
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  }, [stopAudio, cancelRecording]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio();
      cancelRecording();
    };
  }, [stopAudio, cancelRecording]);

  if (!currentStep) {
    return null;
  }

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-3">
        <div className="max-w-xl mx-auto flex items-center justify-between">
          <button
            onClick={onExit}
            className="flex items-center gap-2 text-slate-600 hover:text-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
            <span className="text-sm font-medium">Exit</span>
          </button>

          <span className="text-sm font-medium text-slate-600">
            Line {currentIndex + 1} of {steps.length}
          </span>
        </div>
      </header>

      {/* Progress bar */}
      <div className="bg-white px-4 py-3 border-b border-slate-100">
        <div className="max-w-xl mx-auto">
          <ProgressBar current={completedIndices.size} total={steps.length} />
        </div>
      </div>

      {/* Main content */}
      <main className="flex-1 flex flex-col justify-center p-4">
        <div className="max-w-xl mx-auto w-full space-y-4">
          {/* Title */}
          <div className="text-center mb-6">
            <h1 className="text-xl font-bold text-slate-800">Shadowing Practice</h1>
            <p className="text-sm text-slate-500 mt-1">
              {currentStep.type === 'morgan'
                ? 'Listen to Morgan, then tap Next'
                : 'Say this line out loud, then tap Next'}
            </p>
          </div>

          {/* Current line */}
          <PracticeLine
            type={currentStep.type}
            text={currentStep.text}
            originalText={currentStep.originalText}
            isActive={true}
            isCompleted={completedIndices.has(currentIndex)}
            isPlaying={isPlaying || isLoadingAudio}
            isRecording={isRecording}
            onPlay={handlePlayMorgan}
            onRecord={handleStartRecording}
            onStopRecording={handleStopRecording}
          />

          {/* Navigation */}
          <div className="flex items-center justify-between pt-6">
            <button
              onClick={goPrev}
              disabled={currentIndex === 0}
              className="flex items-center gap-2 px-4 py-2 text-slate-600 hover:text-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>

            <button
              onClick={goNext}
              className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 text-white rounded-full font-medium hover:bg-blue-700 transition-colors"
            >
              {isLastStep ? (
                <>
                  Finish
                  <CheckCircle className="w-4 h-4" />
                </>
              ) : (
                <>
                  Next
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>
        </div>
      </main>

      {/* Completion overlay */}
      {allCompleted && isLastStep && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-sm w-full text-center space-y-4">
            <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
            <h2 className="text-xl font-bold text-slate-800">Great job!</h2>
            <p className="text-slate-600">
              You've completed the shadowing practice. Keep practicing to build your fluency!
            </p>
            <button
              onClick={onComplete}
              className="w-full py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 transition-colors"
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
