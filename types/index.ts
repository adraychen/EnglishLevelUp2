// Coach types
export type CoachStyle = 'casual' | 'clear';
export type CoachName = 'Dora' | 'Morgan';

// Chat message
export interface ChatMessage {
  id: string;
  role: 'coach' | 'student';
  content: string;
  coachName?: CoachName;
}

// Practice turn for shadowing/replay feature
export interface PracticeTurn {
  morgan: string;      // Morgan's line the student replied to
  student: string;     // Student's original sentence
  corrected: string;   // The corrected sentence
  reply: string;       // Morgan's reply after the student
}

// Session state
export interface SessionState {
  style: CoachStyle;
  coachName: CoachName;
  messages: ChatMessage[];
  turns: PracticeTurn[];
  exchanges: number;
  topicId?: number;
  taughtWords: string[];
}

// API response types
export interface RespondResponse {
  reply: string;
  reply_audio: string;
  session_complete: boolean;
}

export interface SummaryResponse {
  summary: string;
  practice_turns: PracticeTurn[];
}

export interface TranscribeResponse {
  text: string;
}

export interface TtsResponse {
  audio: string;
}

export interface SetStyleResponse {
  style: CoachStyle;
  coach_name: CoachName;
  opening: string;
  opening_audio: string;
}

// Database types
export interface Topic {
  id: number;
  topic_order: number;
  name: string;
  level: string;
  opening: string;
  vocabulary_pool: string;
  focus_keyword: string;
  sample_coach_views?: string;
}

export interface LearningLogEntry {
  id?: number;
  user_name: string;
  topic_id: number;
  word_taught: string;
  had_error: boolean;
  created_at?: string;
}
