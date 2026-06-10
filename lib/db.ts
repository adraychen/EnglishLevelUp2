import { getServerSupabase } from './supabase';
import { Topic } from '@/types';

const DEFAULT_USER_NAME = 'Ray'; // Fallback for unauthenticated users

/**
 * Get the set of completed topic IDs for a user.
 */
export async function getCompletedTopicIds(userName: string = DEFAULT_USER_NAME): Promise<Set<number>> {
  let supabase;
  try {
    supabase = getServerSupabase();
  } catch (error) {
    console.error('Supabase not configured:', error);
    return new Set();
  }

  const { data: logEntries, error: logError } = await supabase
    .from('eec_learning_log')
    .select('topic_id')
    .eq('user_name', userName)
    .eq('word_taught', '__session_complete__');

  if (logError) {
    console.error('Error fetching completed topics:', logError);
    return new Set();
  }

  return new Set((logEntries || []).map((r) => r.topic_id));
}

/**
 * Return the next uncompleted topic for the user.
 * Topics are ordered by topic_order. If all are completed, returns the first topic.
 * Also returns whether this is a first visit (for intro display).
 */
export async function getNextTopic(
  userName: string = DEFAULT_USER_NAME
): Promise<(Topic & { is_first_visit: boolean }) | null> {
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

  // Get completed topic IDs for user
  const completedIds = await getCompletedTopicIds(userName);

  // Find first uncompleted topic
  for (const topic of topics) {
    if (!completedIds.has(topic.id)) {
      return {
        ...topic,
        is_first_visit: true,
      };
    }
  }

  // All topics completed — cycle back to the first
  return {
    ...topics[0],
    is_first_visit: false,
  };
}

/**
 * Check if a user has completed a specific topic before.
 */
export async function hasCompletedTopic(
  topicId: number,
  userName: string = DEFAULT_USER_NAME
): Promise<boolean> {
  const completedIds = await getCompletedTopicIds(userName);
  return completedIds.has(topicId);
}

/**
 * Log topic completion for a user.
 * Inserts one row with '__session_complete__' marker.
 */
export async function logTopicCompletion(
  topicId: number,
  userName: string = DEFAULT_USER_NAME
): Promise<void> {
  let supabase;
  try {
    supabase = getServerSupabase();
  } catch (error) {
    console.error('Supabase not configured:', error);
    return;
  }

  const { error } = await supabase.from('eec_learning_log').insert({
    user_name: userName,
    topic_id: topicId,
    word_taught: '__session_complete__',
    had_error: false,
  });

  if (error) {
    console.error('Error logging topic completion:', error);
  }
}
