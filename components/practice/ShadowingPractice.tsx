'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { ArrowLeft, Volume2, CheckCircle, RefreshCw, ChevronRight } from 'lucide-react';
import { PracticeTurn } from '@/types';
import { ProgressBar, RecordButton } from '@/components/ui';
import { useAudioRecorder, useAudioPlayer } from '@/hooks';
import { scorePronunciation, ScoreResult } from '@/utils/scoringUtils';

interface PracticeStep {
  type: 'morgan' | 'student';
  text: string;
  originalText?: string; // For student lines, the original (uncorrected) text
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
  const [transcript, setTranscript] = useState<string>('');
  const [scoreResult, setScoreResult] = useState<ScoreResult | null>(null);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);

  const { isRecording, isTranscribing, startRecording, stopRecording } = useAudioRecorder();
  const { isPlaying, playMp3 } = useAudioPlayer();

  // Flatten turns into a sequence of practice steps
  const steps = useMemo<PracticeStep[]>(() => {
    const result: PracticeStep[] = [];

    turns.forEach((turn, index) => {
      // First turn: include Morgan's opening line
      if (index === 0 && turn.morgan) {
        result.push({ type: 'morgan', text: turn.morgan });
      }

      // Student's corrected line (what they should say)
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

  // Play the current line
  const handlePlay = useCallback(async () => {
    if (!currentStep || isPlaying || isLoadingAudio) return;

    setIsLoadingAudio(true);
    const audio = await fetchTTS(currentStep.text);
    setIsLoadingAudio(false);

    if (audio) {
      await playMp3(audio);
    }
  }, [currentStep, isPlaying, isLoadingAudio, fetchTTS, playMp3]);

  // Auto-play when step changes (for Morgan lines)
  useEffect(() => {
    if (currentStep?.type === 'morgan') {
      // Small delay to let UI render first
      const timer = setTimeout(() => {
        handlePlay();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [currentIndex]); // Only trigger on index change, not handlePlay

  // Handle recording
  const handleToggleRecord = useCallback(async () => {
    if (isRecording) {
      const result = await stopRecording();
      if (result) {
        setTranscript(result);
        const score = scorePronunciation(currentStep?.text || '', result);
        setScoreResult(score);
      }
    } else {
      setTranscript('');
      setScoreResult(null);
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording, currentStep]);

  // Handle Try Again
  const handleTryAgain = useCallback(() => {
    setTranscript('');
    setScoreResult(null);
  }, []);

  // Handle Next
  const handleNext = useCallback(() => {
    setTranscript('');
    setScoreResult(null);

    if (isLastStep) {
      onComplete();
    } else {
      setCurrentIndex((prev) => prev + 1);
    }
  }, [isLastStep, onComplete]);

  if (!currentStep) {
    return null;
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-4 py-3 sticky top-0 z-10">
        <div className="max-w-xl mx-auto flex items-center justify-between">
          <button
            onClick={onExit}
            className="flex items-center text-slate-500 hover:text-slate-800 transition-colors font-medium text-sm"
          >
            <ArrowLeft className="w-5 h-5 mr-1" /> Exit
          </button>
          <div className="flex flex-col items-center">
            <span className="font-bold text-slate-700 text-sm uppercase tracking-wide">
              Shadowing Practice
            </span>
            <span className="text-[10px] text-slate-400 font-medium">
              {currentIndex + 1} of {steps.length}
            </span>
          </div>
          <div className="w-16" />
        </div>
      </header>

      {/* Progress */}
      <div className="bg-white px-4 py-3 border-b border-slate-100">
        <div className="max-w-xl mx-auto">
          <ProgressBar current={currentIndex + 1} total={steps.length} />
        </div>
      </div>

      <main className="flex-1 p-6 max-w-xl mx-auto w-full flex flex-col gap-6">
        {/* Target Sentence */}
        <section className="space-y-3">
          <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">
            {currentStep.type === 'morgan' ? 'Morgan says' : 'Your turn to say'}
          </p>

          <div className="bg-white p-6 rounded-2xl border-2 border-slate-100 shadow-sm min-h-[120px] flex items-center">
            {scoreResult ? (
              <div className="text-xl font-medium leading-relaxed flex flex-wrap gap-1">
                {scoreResult.wordResults.map((wr, idx) => (
                  <span
                    key={idx}
                    className={
                      wr.status === 'hit'
                        ? 'text-green-600 bg-green-50 px-1 rounded'
                        : 'text-red-400 line-through decoration-red-300 decoration-2'
                    }
                  >
                    {wr.word}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-xl font-medium text-slate-800 leading-relaxed">
                {currentStep.text}
              </p>
            )}
          </div>

          <button
            onClick={handlePlay}
            disabled={isPlaying || isLoadingAudio}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition-colors disabled:opacity-50"
          >
            <Volume2 className="w-5 h-5" />
            {isLoadingAudio ? 'Loading...' : isPlaying ? 'Playing...' : 'Listen'}
          </button>
        </section>

        {/* Recording Section */}
        <section className="space-y-4 pt-4 border-t border-slate-200">
          <div className="flex items-center justify-between">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">
              Your Turn
            </p>
            {isRecording && (
              <span className="text-red-500 text-xs font-bold animate-pulse">
                Recording...
              </span>
            )}
            {isTranscribing && (
              <span className="text-blue-500 text-xs font-bold animate-pulse">
                Processing...
              </span>
            )}
          </div>

          <div className="flex flex-col items-center gap-4">
            <RecordButton
              isRecording={isRecording}
              onToggle={handleToggleRecord}
              disabled={isTranscribing}
            />
            <p className="text-sm text-slate-500">
              {isRecording
                ? 'Say the sentence out loud...'
                : isTranscribing
                ? 'Checking your pronunciation...'
                : 'Tap microphone to start speaking'}
            </p>
          </div>
        </section>

        {/* Transcription display */}
        {transcript && !scoreResult && (
          <div className="bg-slate-100 p-4 rounded-xl">
            <p className="text-sm text-slate-600">{transcript}</p>
          </div>
        )}

        {/* Score Result */}
        {scoreResult && (
          <section className="animate-in slide-in-from-bottom-4 fade-in duration-300">
            <div
              className={`p-5 rounded-2xl border ${scoreResult.bg} ${scoreResult.color} border-transparent flex flex-col md:flex-row items-center justify-between gap-4 shadow-sm`}
            >
              <div>
                <p className="font-bold text-xl">{scoreResult.label}</p>
                <p className="text-sm opacity-80">Accuracy: {scoreResult.accuracy}%</p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleTryAgain}
                  className="flex items-center gap-1 px-4 py-2 bg-white/50 rounded-xl text-sm font-medium hover:bg-white/70 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" /> Try Again
                </button>
                <button
                  onClick={handleNext}
                  className="flex items-center gap-1 px-4 py-2 bg-white rounded-xl text-sm font-medium hover:bg-white/90 transition-colors"
                >
                  {isLastStep ? 'Finish' : 'Next'} <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </section>
        )}

        {/* Completion Modal */}
        {isLastStep && scoreResult && scoreResult.accuracy >= 70 && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm">
            <div className="bg-white rounded-3xl p-8 max-w-sm w-full shadow-2xl space-y-6 text-center">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto text-green-600">
                <CheckCircle className="w-10 h-10" />
              </div>
              <div>
                <h3 className="text-2xl font-bold text-slate-900">Great job!</h3>
                <p className="text-slate-500 mt-2">
                  You've completed the shadowing practice.
                </p>
              </div>
              <button
                onClick={onComplete}
                className="w-full py-3 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 transition-colors"
              >
                Done
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
