import { getGroqClient, MODELS } from '@/lib/groq';
import { SessionAnalysis, ProgressReport, Turn } from '@/types';

/**
 * Convert score to label
 */
export function scoreToLabel(score: number): string {
  if (score <= 3) return 'Beginner';
  if (score <= 5) return 'Developing';
  if (score <= 7) return 'Intermediate';
  if (score <= 9) return 'Fluent';
  return 'Mastery';
}

/**
 * Analyze a conversation session and return scores
 */
export async function analyzeSession(
  turns: Array<{
    app_question: string;
    student_speech: string;
    fluency_comment: string;
  }>
): Promise<Omit<SessionAnalysis, 'id' | 'session_id' | 'created_at'>> {
  const client = getGroqClient();

  // Build turns text
  let turnsText = '';
  for (let i = 0; i < turns.length; i++) {
    const t = turns[i];
    turnsText += `Turn ${i + 1}:\n`;
    turnsText += `  Question: ${t.app_question}\n`;
    turnsText += `  Student said: ${t.student_speech}\n`;
    turnsText += `  Coach comment: ${t.fluency_comment}\n\n`;
  }

  const prompt = `Analyze the following spoken English conversation turns from a student:

${turnsText}

Assess the student across three categories.
Respond ONLY with a JSON object in this exact format:
{
  "vocabulary_score": <1-10>,
  "vocabulary_note": "<2-3 sentence assessment>",
  "phrasing_score": <1-10>,
  "phrasing_note": "<2-3 sentence assessment>",
  "structure_score": <1-10>,
  "structure_note": "<2-3 sentence assessment>",
  "overall_score": <1-10>,
  "overall_note": "<2-3 sentence overall summary>",
  "suggestion": "<one specific thing to focus on next session>"
}`;

  try {
    const response = await client.chat.completions.create({
      model: MODELS.MORGAN,
      messages: [
        {
          role: 'system',
          content:
            'You are an expert English language assessor specializing in spoken fluency for non-native speakers. You assess speech across three categories: vocabulary, phrasing & expression, and sentence structure. You are precise, fair, and constructive in your feedback. Always respond with valid JSON only.',
        },
        { role: 'user', content: prompt },
      ],
      max_tokens: 800,
      temperature: 0.3,
    });

    const raw = response.choices[0]?.message?.content || '';

    // Extract JSON from response
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        const data = JSON.parse(match[0]);
        return {
          vocabulary_score: data.vocabulary_score || 5,
          vocabulary_note: data.vocabulary_note || 'Unable to analyze.',
          phrasing_score: data.phrasing_score || 5,
          phrasing_note: data.phrasing_note || 'Unable to analyze.',
          structure_score: data.structure_score || 5,
          structure_note: data.structure_note || 'Unable to analyze.',
          overall_score: data.overall_score || 5,
          overall_note: data.overall_note || 'Unable to analyze.',
          suggestion: data.suggestion || 'Keep practicing!',
        };
      } catch {
        // JSON parse failed
      }
    }
  } catch (error) {
    console.error('Session analysis error:', error);
  }

  // Default fallback
  return {
    vocabulary_score: 5,
    vocabulary_note: 'Unable to analyze.',
    phrasing_score: 5,
    phrasing_note: 'Unable to analyze.',
    structure_score: 5,
    structure_note: 'Unable to analyze.',
    overall_score: 5,
    overall_note: 'Unable to analyze.',
    suggestion: 'Keep practicing!',
  };
}

/**
 * Analyze progress across multiple sessions
 */
export async function analyzeProgress(
  sessionsData: Array<{
    topic: string;
    vocabulary_score: number;
    vocabulary_note: string;
    phrasing_score: number;
    phrasing_note: string;
    structure_score: number;
    structure_note: string;
    overall_score: number;
    overall_note: string;
  }>
): Promise<Omit<ProgressReport, 'id' | 'user_id' | 'report_number' | 'sessions_from' | 'sessions_to' | 'generated_at'>> {
  const client = getGroqClient();

  // Build sessions text
  let sessionsText = '';
  for (let i = 0; i < sessionsData.length; i++) {
    const s = sessionsData[i];
    sessionsText += `Session ${i + 1} (Topic: ${s.topic}):\n`;
    sessionsText += `  Vocabulary: ${s.vocabulary_score}/10 — ${s.vocabulary_note}\n`;
    sessionsText += `  Phrasing: ${s.phrasing_score}/10 — ${s.phrasing_note}\n`;
    sessionsText += `  Structure: ${s.structure_score}/10 — ${s.structure_note}\n`;
    sessionsText += `  Overall: ${s.overall_score}/10 — ${s.overall_note}\n\n`;
  }

  const prompt = `Here are ${sessionsData.length} sessions of English fluency data for a student:

${sessionsText}

Generate a progress report. Compare performance across the sessions and describe improvement or areas needing work.
Respond ONLY with a JSON object:
{
  "vocabulary_score": <average 1-10>,
  "vocabulary_description": "<3-4 sentence progress description>",
  "phrasing_score": <average 1-10>,
  "phrasing_description": "<3-4 sentence progress description>",
  "structure_score": <average 1-10>,
  "structure_description": "<3-4 sentence progress description>",
  "overall_score": <average 1-10>,
  "improvement_description": "<3-4 sentence overall progress summary>"
}`;

  try {
    const response = await client.chat.completions.create({
      model: MODELS.MORGAN,
      messages: [
        {
          role: 'system',
          content:
            'You are an expert English language assessor specializing in spoken fluency for non-native speakers. Generate progress reports that compare performance across sessions. Always respond with valid JSON only.',
        },
        { role: 'user', content: prompt },
      ],
      max_tokens: 1000,
      temperature: 0.3,
    });

    const raw = response.choices[0]?.message?.content || '';

    // Extract JSON from response
    const match = raw.match(/\{[\s\S]*\}/);
    if (match) {
      try {
        const data = JSON.parse(match[0]);
        return {
          vocabulary_score: data.vocabulary_score || 5,
          vocabulary_label: scoreToLabel(data.vocabulary_score || 5),
          vocabulary_description: data.vocabulary_description || '',
          phrasing_score: data.phrasing_score || 5,
          phrasing_label: scoreToLabel(data.phrasing_score || 5),
          phrasing_description: data.phrasing_description || '',
          structure_score: data.structure_score || 5,
          structure_label: scoreToLabel(data.structure_score || 5),
          structure_description: data.structure_description || '',
          overall_score: data.overall_score || 5,
          overall_label: scoreToLabel(data.overall_score || 5),
          improvement_description: data.improvement_description || '',
        };
      } catch {
        // JSON parse failed
      }
    }
  } catch (error) {
    console.error('Progress analysis error:', error);
  }

  // Calculate averages from input data
  const avgVocab =
    sessionsData.reduce((sum, s) => sum + s.vocabulary_score, 0) / sessionsData.length;
  const avgPhrasing =
    sessionsData.reduce((sum, s) => sum + s.phrasing_score, 0) / sessionsData.length;
  const avgStructure =
    sessionsData.reduce((sum, s) => sum + s.structure_score, 0) / sessionsData.length;
  const avgOverall =
    sessionsData.reduce((sum, s) => sum + s.overall_score, 0) / sessionsData.length;

  return {
    vocabulary_score: Math.round(avgVocab * 10) / 10,
    vocabulary_label: scoreToLabel(avgVocab),
    vocabulary_description: 'Analysis not available.',
    phrasing_score: Math.round(avgPhrasing * 10) / 10,
    phrasing_label: scoreToLabel(avgPhrasing),
    phrasing_description: 'Analysis not available.',
    structure_score: Math.round(avgStructure * 10) / 10,
    structure_label: scoreToLabel(avgStructure),
    structure_description: 'Analysis not available.',
    overall_score: Math.round(avgOverall * 10) / 10,
    overall_label: scoreToLabel(avgOverall),
    improvement_description: 'Analysis not available.',
  };
}
