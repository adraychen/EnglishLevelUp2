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
  opening: string | Array<{ opening: string }>;
  intro?: string; // Topic introduction shown on first visit
  vocabulary_pool: string;
  focus_keyword: string;
  coach_views?: string | Array<{ coach_views: string }>;
}

export interface LearningLogEntry {
  id?: number;
  user_name: string;
  topic_id: number;
  word_taught: string;
  had_error: boolean;
  created_at?: string;
}

// User and auth types
export type UserRole = 'student' | 'teacher';

export interface User {
  id: number;
  name: string;
  email: string;
  password_hash: string;
  role: UserRole;
  created_at: string;
}

export interface UserProfile {
  id: number;
  name: string;
  email: string;
  role: UserRole;
}

// Session tracking types
export interface DbSession {
  id: number;
  user_id: number;
  topic: string;
  topic_id?: number;
  session_number: number;
  date: string;
  analysis?: SessionAnalysis;
}

export interface Turn {
  id?: number;
  session_id: number;
  turn_number: number;
  app_question: string;
  student_speech: string;
  fluency_comment: string;
}

export interface SessionAnalysis {
  id?: number;
  session_id: number;
  vocabulary_score: number;
  vocabulary_note: string;
  phrasing_score: number;
  phrasing_note: string;
  structure_score: number;
  structure_note: string;
  overall_score: number;
  overall_note: string;
  suggestion: string;
  created_at?: string;
}

export interface ProgressReport {
  id?: number;
  user_id: number;
  report_number: number;
  sessions_from: number;
  sessions_to: number;
  vocabulary_score: number;
  vocabulary_label: string;
  vocabulary_description: string;
  phrasing_score: number;
  phrasing_label: string;
  phrasing_description: string;
  structure_score: number;
  structure_label: string;
  structure_description: string;
  overall_score: number;
  overall_label: string;
  improvement_description: string;
  generated_at?: string;
}

// Dashboard response types
export interface DashboardData {
  user: UserProfile;
  sessions: DbSession[];
  reports: ProgressReport[];
  topics: Topic[];
}

export interface StudentSummary {
  id: number;
  name: string;
  email: string;
  session_count: number;
  latest_score?: number;
  latest_session_date?: string;
}
