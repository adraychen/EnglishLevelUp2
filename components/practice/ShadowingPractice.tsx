'use client';

import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { ArrowLeft, Volume2, CheckCircle, RefreshCw, ChevronRight, Mic, Square } from 'lucide-react';
import { PracticeTurn } from '@/types';
import { ProgressBar } from '@/components/ui';
import { useAudioRecorder, useAudioPlayer } from '@/hooks';
import { scorePronunciation, ScoreResult } from '@/utils/scoringUtils';

interface ShadowingPracticeProps {
  turns: PracticeTurn[];
  onComplete: () => void;
  onExit: () => void;
}

type ActiveLine = 'morgan' | 'student' | null;

interface LineScore {
  transcript: string;
  result: ScoreResult;
}

export const ShadowingPractice: React.FC<ShadowingPracticeProps> = ({
  turns,
  onComplete,
  onExit,
}) => {
  const [currentExchange, setCurrentExchange] = useState(0);
  const [activeLine, setActiveLine] = useState<ActiveLine>(null);
  const [morganScore, setMorganScore] = useState<LineScore | null>(null);
  const [studentScore, setStudentScore] = useState<LineScore | null>(null);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [hasAutoPlayed, setHasAutoPlayed] = useState(false);

  const { isRecording, isTranscribing, startRecording, stopRecording } = useAudioRecorder();
  const { isPlaying, playMp3 } = useAudioPlayer();

  // Build exchanges from turns
  // Each exchange has Morgan's line and the student's response
  const exchanges = useMemo(() => {
    return turns.map((turn) => ({
      morgan: turn.morgan || '',
      student: turn.student || '',
      corrected: turn.corrected || turn.student || '',
      reply: turn.reply || '',
    }));
  }, [turns]);

  const currentTurn = exchanges[currentExchange];
  const isLastExchange = currentExchange === exchanges.length - 1;

  // Check if student line has a correction (different from original)
  const hasCorrectedVersion = currentTurn &&
    currentTurn.corrected.toLowerCase().trim() !== currentTurn.student.toLowerCase().trim();

  // Fetch TTS for a line
  const fetchTTS = useCallback(async (text: string): Promise<string> => {
    if (!text) return '';
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

  // Play Morgan's line
  const handlePlayMorgan = useCallback(async () => {
    if (!currentTurn?.morgan || isPlaying || isLoadingAudio) return;

    setIsLoadingAudio(true);
    const audio = await fetchTTS(currentTurn.morgan);
    setIsLoadingAudio(false);

    if (audio) {
      await playMp3(audio);
    }
  }, [currentTurn, isPlaying, isLoadingAudio, fetchTTS, playMp3]);

  // Auto-play Morgan's line when exchange changes
  useEffect(() => {
    if (currentTurn?.morgan && !hasAutoPlayed) {
      const timer = setTimeout(() => {
        handlePlayMorgan();
        setHasAutoPlayed(true);
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [currentExchange, hasAutoPlayed]);

  // Handle recording for a specific line
  const handleToggleRecord = useCallback(async (line: 'morgan' | 'student') => {
    if (isRecording) {
      const result = await stopRecording();
      if (result) {
        const targetText = line === 'morgan' ? currentTurn.morgan : currentTurn.corrected;
        const score = scorePronunciation(targetText, result);
        if (score) {
          const lineScore = { transcript: result, result: score };
          if (line === 'morgan') {
            setMorganScore(lineScore);
          } else {
            setStudentScore(lineScore);
          }
        }
      }
      setActiveLine(null);
    } else {
      // Clear previous score for this line
      if (line === 'morgan') {
        setMorganScore(null);
      } else {
        setStudentScore(null);
      }
      setActiveLine(line);
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording, currentTurn]);

  // Retry a specific line
  const handleRetry = useCallback((line: 'morgan' | 'student') => {
    if (line === 'morgan') {
      setMorganScore(null);
    } else {
      setStudentScore(null);
    }
  }, []);

  // Go to next exchange
  const handleNext = useCallback(() => {
    setMorganScore(null);
    setStudentScore(null);
    setActiveLine(null);
    setHasAutoPlayed(false);

    if (isLastExchange) {
      onComplete();
    } else {
      setCurrentExchange((prev) => prev + 1);
    }
  }, [isLastExchange, onComplete]);

  if (!currentTurn) {
    return null;
  }

  const renderScoreResult = (score: ScoreResult) => (
    <div className="text-lg font-medium leading-relaxed flex flex-wrap gap-1">
      {score.wordResults.map((wr, idx) => (
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
  );

  const renderScoreBadge = (score: ScoreResult) => (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium ${score.bg} ${score.color}`}>
      {score.label} ({score.accuracy}%)
    </div>
  );

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
              Exchange {currentExchange + 1} of {exchanges.length}
            </span>
          </div>
          <div className="w-16" />
        </div>
      </header>

      {/* Progress */}
      <div className="bg-white px-4 py-3 border-b border-slate-100">
        <div className="max-w-xl mx-auto">
          <ProgressBar current={currentExchange + 1} total={exchanges.length} />
        </div>
      </div>

      <main className="flex-1 p-4 max-w-xl mx-auto w-full flex flex-col gap-4">
        {/* Morgan's Line */}
        <section className="bg-white p-5 rounded-2xl border-2 border-blue-100 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">
              Morgan says
            </span>
            {morganScore && renderScoreBadge(morganScore.result)}
          </div>

          {/* Morgan's text or score result */}
          <div className="min-h-[60px] mb-4">
            {morganScore ? (
              renderScoreResult(morganScore.result)
            ) : (
              <p className="text-lg font-medium text-slate-800 leading-relaxed">
                {currentTurn.morgan}
              </p>
            )}
          </div>

          {/* Morgan line actions */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={handlePlayMorgan}
              disabled={isPlaying || isLoadingAudio}
              className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition-colors disabled:opacity-50 text-sm font-medium"
            >
              <Volume2 className="w-4 h-4" />
              {isLoadingAudio ? 'Loading...' : isPlaying ? 'Playing...' : 'Listen'}
            </button>

            {morganScore ? (
              <button
                onClick={() => handleRetry('morgan')}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition-colors text-sm font-medium"
              >
                <RefreshCw className="w-4 h-4" /> Try Again
              </button>
            ) : (
              <button
                onClick={() => handleToggleRecord('morgan')}
                disabled={isTranscribing || (activeLine !== null && activeLine !== 'morgan')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  isRecording && activeLine === 'morgan'
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                } disabled:opacity-50`}
              >
                {isRecording && activeLine === 'morgan' ? (
                  <>
                    <Square className="w-4 h-4 fill-current" /> Stop
                  </>
                ) : isTranscribing && activeLine === 'morgan' ? (
                  'Processing...'
                ) : (
                  <>
                    <Mic className="w-4 h-4" /> Shadow
                  </>
                )}
              </button>
            )}
          </div>
        </section>

        {/* Student's Line */}
        <section className="bg-white p-5 rounded-2xl border-2 border-green-100 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-bold text-green-600 uppercase tracking-wider">
              Your turn
            </span>
            {studentScore && renderScoreBadge(studentScore.result)}
          </div>

          {/* Student's corrected text (or original if no correction) */}
          <div className="min-h-[60px] mb-2">
            {studentScore ? (
              renderScoreResult(studentScore.result)
            ) : (
              <p className="text-lg font-medium text-slate-800 leading-relaxed">
                {currentTurn.corrected}
              </p>
            )}
          </div>

          {/* Original text if different from corrected */}
          {hasCorrectedVersion && !studentScore && (
            <p className="text-sm text-slate-400 mb-4 italic">
              You said: "{currentTurn.student}"
            </p>
          )}

          {/* Student line actions */}
          <div className="flex items-center gap-2 flex-wrap mt-3">
            {studentScore ? (
              <button
                onClick={() => handleRetry('student')}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 text-slate-700 rounded-xl hover:bg-slate-200 transition-colors text-sm font-medium"
              >
                <RefreshCw className="w-4 h-4" /> Try Again
              </button>
            ) : (
              <button
                onClick={() => handleToggleRecord('student')}
                disabled={isTranscribing || (activeLine !== null && activeLine !== 'student')}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-colors ${
                  isRecording && activeLine === 'student'
                    ? 'bg-red-500 text-white animate-pulse'
                    : 'bg-green-600 text-white hover:bg-green-700'
                } disabled:opacity-50`}
              >
                {isRecording && activeLine === 'student' ? (
                  <>
                    <Square className="w-4 h-4 fill-current" /> Stop
                  </>
                ) : isTranscribing && activeLine === 'student' ? (
                  'Processing...'
                ) : (
                  <>
                    <Mic className="w-4 h-4" /> Practice
                  </>
                )}
              </button>
            )}
          </div>
        </section>

        {/* Next Button */}
        <div className="flex justify-end pt-2">
          <button
            onClick={handleNext}
            className="flex items-center gap-2 px-6 py-3 bg-slate-800 text-white rounded-xl font-medium hover:bg-slate-900 transition-colors"
          >
            {isLastExchange ? 'Finish' : 'Next'} <ChevronRight className="w-5 h-5" />
          </button>
        </div>

        {/* Completion Modal */}
        {isLastExchange && (morganScore || studentScore) && (
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
