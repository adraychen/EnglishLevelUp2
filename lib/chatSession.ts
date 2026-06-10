import { getServerSupabase } from './supabase';
import { getSession as getAuthSession } from './auth';

export interface ChatState {
  style: string;
  history: Array<{ role: string; content: string }>;
  turns: Array<{
    morgan: string;
    student: string;
    corrected: string;
    reply: string;
  }>;
  exchanges: number;
  lastQuestion: string;
  userName: string;
  topic?: {
    id: number;
    name: string;
    level: string;
    vocabulary_pool: string;
    coach_views?: string;
    focus_keyword: string;
  };
  topicId?: number;
}

const DEFAULT_STATE: ChatState = {
  style: 'casual',
  history: [],
  turns: [],
  exchanges: 0,
  lastQuestion: '',
  userName: 'Ray',
};

/**
 * Get user identifier for session storage.
 * Uses authenticated user ID or falls back to 'anonymous'.
 */
async function getUserId(): Promise<string> {
  try {
    const authSession = await getAuthSession();
    if (authSession?.userId) {
      return `user_${authSession.userId}`;
    }
  } catch {
    // Not authenticated
  }
  return 'anonymous';
}

/**
 * Get chat state from database.
 */
export async function getChatState(): Promise<ChatState> {
  try {
    const userId = await getUserId();
    const supabase = getServerSupabase();

    const { data, error } = await supabase
      .from('chat_state')
      .select('state')
      .eq('user_id', userId)
      .single();

    if (error || !data) {
      return { ...DEFAULT_STATE };
    }

    return { ...DEFAULT_STATE, ...data.state };
  } catch (error) {
    console.error('getChatState error:', error);
    return { ...DEFAULT_STATE };
  }
}

/**
 * Save chat state to database (upsert).
 */
export async function setChatState(state: ChatState): Promise<void> {
  try {
    const userId = await getUserId();
    const supabase = getServerSupabase();

    const { error } = await supabase
      .from('chat_state')
      .upsert(
        {
          user_id: userId,
          state,
          updated_at: new Date().toISOString(),
        },
        { onConflict: 'user_id' }
      );

    if (error) {
      console.error('setChatState error:', error);
    }
  } catch (error) {
    console.error('setChatState error:', error);
  }
}

/**
 * Clear chat state (delete from database).
 */
export async function clearChatState(): Promise<void> {
  try {
    const userId = await getUserId();
    const supabase = getServerSupabase();

    await supabase.from('chat_state').delete().eq('user_id', userId);
  } catch (error) {
    console.error('clearChatState error:', error);
  }
}
