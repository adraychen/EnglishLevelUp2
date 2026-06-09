import { PracticeTurn } from '@/types';

/**
 * Return which single vocabulary words from the pool literally appear in the text.
 * Simple, reliable text matching — no LLM judgement. Sentence patterns (with [..])
 * are skipped since they don't match literally.
 */
export function wordsUsedInText(text: string, vocabularyPool: string): string[] {
  const poolItems = (vocabularyPool || '')
    .split('\n')
    .map((w) => w.trim())
    .filter(Boolean);

  const textLower = (text || '').toLowerCase();
  const used: string[] = [];

  for (const w of poolItems) {
    // Skip sentence patterns
    if (w.includes('[')) {
      continue;
    }
    // Check for word boundary match
    const regex = new RegExp(`\\b${escapeRegExp(w.toLowerCase())}\\b`);
    if (regex.test(textLower)) {
      used.push(w);
    }
  }

  return used;
}

/**
 * Escape special regex characters in a string
 */
function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Build a review of the whole conversation as markdown.
 * `turns` is a list of PracticeTurn objects.
 * Shows Morgan's lines, the student's original, and the correction where different.
 */
export function buildReview(
  turns: PracticeTurn[],
  style: string = 'clear',
  topicName: string = ''
): string {
  if (!turns || turns.length === 0) {
    return "**Nice chat!** There's nothing to review yet.";
  }

  const lines: string[] = ['## Conversation Review', ''];

  if (topicName) {
    lines.push(`*Topic: ${topicName}*`);
    lines.push('');
  }

  let anyCorrection = false;

  for (const t of turns) {
    const morganLine = (t.morgan || '').trim();
    const studentLine = (t.student || '').trim();
    const correctedLine = (t.corrected || '').trim();

    if (morganLine) {
      lines.push(`**Morgan:** ${morganLine}`);
    }

    if (studentLine) {
      lines.push(`**You:** ${studentLine}`);

      if (correctedLine && correctedLine.toLowerCase() !== studentLine.toLowerCase()) {
        lines.push(`**✓ Better:** ${correctedLine}`);
        anyCorrection = true;
      }
    }

    lines.push('');
  }

  lines.push('---');

  if (anyCorrection) {
    lines.push(
      'The **✓ Better** lines show a more natural way to say what you said. ' +
        'Try the practice round to say them out loud!'
    );
  } else {
    lines.push(
      'Your English was natural throughout — wonderful work! ' +
        'Try the practice round to say the conversation again.'
    );
  }

  return lines.join('\n');
}
