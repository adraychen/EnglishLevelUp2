-- Chat state table for server-side sessions
-- One row per user, upserted on each action
CREATE TABLE IF NOT EXISTS chat_state (
  user_id TEXT PRIMARY KEY,
  state JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cleanup queries (optional)
CREATE INDEX IF NOT EXISTS idx_chat_state_updated_at ON chat_state(updated_at);

-- Enable Row Level Security
ALTER TABLE chat_state ENABLE ROW LEVEL SECURITY;

-- Block direct API access (app uses service role key which bypasses RLS)
DROP POLICY IF EXISTS "No direct access" ON chat_state;
CREATE POLICY "No direct access" ON chat_state FOR ALL TO anon, authenticated USING (false);
