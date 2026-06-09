import { getServerSupabase } from './supabase';
import { Topic, LearningLogEntry } from '@/types';

const USER_NAME = 'Ray'; // Single user for now; becomes user_id with login

/**
 * Return only the single vocabulary words from a pool (excludes [..] patterns)
 */
function singleWords(vocabularyPool: string): string[] {
  return (vocabularyPool || '')
    .split('\n')
    .map((w) => w.trim())
    .filter((w) => w && !w.includes('['));
}

/**
 * Return the next topic for the user plus the words already taught for it.
 * Advances through topics in order: a topic is 'complete' when all its single
 * vocabulary words have been taught.
 */
export async function getNextTopic(): Promise<(Topic & { already_taught: string[] }) | null> {
  let supabase;
  try {
    supabase = getServerSupabase();
  } catch (error) {
    console.error('Supabase not configured:', error);
    return null;
  }

  // Get all topics ordered
  const { data: topics, error: topicsError } = await supabase
    .from('eec_topics')
    .select('*')
    .order('topic_order', { ascending: true });

  if (topicsError || !topics || topics.length === 0) {
    console.error('Error fetching topics:', topicsError);
    return null;
  }

  for (const topic of topics) {
    const words = singleWords(topic.vocabulary_pool);

    // Which of this topic's words has the user already been taught?
    const { data: logEntries, error: logError } = await supabase
      .from('eec_learning_log')
      .select('word_taught')
      .eq('user_name', USER_NAME)
      .eq('topic_id', topic.id);

    if (logError) {
      console.error('Error fetching learning log:', logError);
      continue;
    }

    const taught = new Set((logEntries || []).map((r) => r.word_taught));
    const taughtSingle = words.filter((w) => taught.has(w));

    // If not all single words are taught, this is the active topic
    if (taughtSingle.length < words.length) {
      return {
        ...topic,
        already_taught: taughtSingle,
      };
    }
  }

  // All topics fully taught — cycle back to the first, fresh
  return {
    ...topics[0],
    already_taught: [],
  };
}

/**
 * Write taught words to eec_learning_log for the user
 */
export async function logLearning(
  topicId: number,
  taughtWords: string[],
  errorsOccurred: boolean = false
): Promise<void> {
  if (!taughtWords || taughtWords.length === 0) {
    return;
  }

  let supabase;
  try {
    supabase = getServerSupabase();
  } catch (error) {
    console.error('Supabase not configured:', error);
    return;
  }

  const entries: Omit<LearningLogEntry, 'id' | 'created_at'>[] = taughtWords.map((word) => ({
    user_name: USER_NAME,
    topic_id: topicId,
    word_taught: word,
    had_error: errorsOccurred,
  }));

  const { error } = await supabase.from('eec_learning_log').insert(entries);

  if (error) {
    console.error('Error logging learning:', error);
  }
}
