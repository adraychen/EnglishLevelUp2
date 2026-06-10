-- Chat state table for server-side sessions
-- One row per user, upserted on each action
CREATE TABLE IF NOT EXISTS chat_state (
  user_id TEXT PRIMARY KEY,
  state JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cleanup queries (optional)
CREATE INDEX IF NOT EXISTS idx_chat_state_updated_at ON chat_state(updated_at);
