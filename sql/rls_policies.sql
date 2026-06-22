-- Row Level Security (RLS) policies for EnglishLevelUp
--
-- All tables have RLS enabled to protect against direct API access.
-- The app uses the Supabase service role key (which bypasses RLS) for all
-- database operations via server-side API routes.
--
-- Run this after creating all tables.

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE turns ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE progress_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE eec_learning_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE eec_topics ENABLE ROW LEVEL SECURITY;

-- Drop existing policies (safe to re-run)
DROP POLICY IF EXISTS "No direct access" ON users;
DROP POLICY IF EXISTS "No direct access" ON sessions;
DROP POLICY IF EXISTS "No direct access" ON turns;
DROP POLICY IF EXISTS "No direct access" ON session_analysis;
DROP POLICY IF EXISTS "No direct access" ON progress_reports;
DROP POLICY IF EXISTS "No direct access" ON chat_state;
DROP POLICY IF EXISTS "No direct access" ON eec_learning_log;
DROP POLICY IF EXISTS "Topics readable" ON eec_topics;
DROP POLICY IF EXISTS "No direct write" ON eec_topics;

-- Block direct API access for sensitive tables
CREATE POLICY "No direct access" ON users FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON sessions FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON turns FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON session_analysis FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON progress_reports FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON chat_state FOR ALL TO anon, authenticated USING (false);
CREATE POLICY "No direct access" ON eec_learning_log FOR ALL TO anon, authenticated USING (false);

-- Topics are public content, allow read access only
CREATE POLICY "Topics readable" ON eec_topics FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "No direct write" ON eec_topics FOR ALL TO anon, authenticated USING (false);
