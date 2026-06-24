import { getGroqClient, MODELS } from '@/lib/groq';

/**
 * Return a fully corrected version of the student's sentence, or the sentence
 * UNCHANGED if it is already natural and correct. Used for the review and the
 * replay/practice feature. Conservative: only fixes genuine errors.
 */
export async function correctSentence(studentText: string): Promise<string> {
  const text = studentText.trim();

  // Too short to meaningfully correct
  if (text.split(/\s+/).length < 2) {
    return text;
  }

  const systemPrompt = `You are a careful English editor. You are given one sentence from
an English learner. Return the most natural, correct version of that sentence.

RULES:
- Fix only genuine errors: wrong verb tense, subject-verb disagreement, wrong or missing
  articles, wrong prepositions, and clearly unnatural phrasing.
- Keep the student's meaning and their words wherever possible — make the smallest changes
  needed to make it sound natural and correct.
- If the sentence is already natural and correct, return it EXACTLY as it is, unchanged.
- Do NOT change style or word choice just because you prefer a different word.
- Do NOT add or remove information. Do NOT make it longer or fancier.
- Optional contractions ("I am" / "I'm") and informal-but-correct expressions are fine —
  leave them alone.

Return ONLY the corrected sentence as plain text — no quotes, no labels, no explanation.`;

  const userPrompt = `Student sentence:
${text}

Return the corrected sentence (or the same sentence if it is already correct).`;

  try {
    const client = getGroqClient();
    const response = await client.chat.completions.create({
      model: MODELS.DORA, // Use fast model for corrections
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      max_tokens: 200,
      temperature: 0.2,
    });

    let corrected = response.choices[0]?.message?.content?.trim() || '';
    // Strip accidental wrapping quotes
    corrected = corrected.replace(/^["']|["']$/g, '').trim();
    return corrected || text;
  } catch (error) {
    console.error('Correction error:', error);
    return text;
  }
}
