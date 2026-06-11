# English Level Up

An AI-powered conversational English practice app that helps learners build
fluency by speaking with an engaging AI coach. The app combines two
approaches: free casual conversation with silent error correction, and
topic-led practice where the coach models useful everyday phrases in a clear,
podcast-style conversation. After each topic session, learners can replay the
conversation in a practice round to rehearse the corrected sentences out loud.

Built with Next.js (React + TypeScript), Supabase, and the Groq API.

---

## The Two Coaches

Students switch between two coaches at any time using the toggle in the
navigation bar. Switching resets the conversation.

| Coach | Style | Voice | What it's for |
|---|---|---|---|
| Dora | Casual | US English | Free, natural conversation on any subject. Dora chats like a friend, recasts errors silently, and uses rich everyday vocabulary. Good for learners who want exposure to natural native speech. |
| Morgan | Clear, topic-led | US English | Structured practice on a set topic. Morgan hosts a clear, engaging conversation — like an English-learning podcast host — modelling useful words and phrases at the topic's difficulty level. Good for learners building everyday vocabulary and confidence. |

Both coaches use **recasting** — silently weaving the correct form into the
reply instead of pointing out mistakes — and both produce an error review at
the end of the session.

---

## How It Works

### Dora — free conversation
The student chats freely with Dora by voice or text. Dora keeps the
conversation natural and engaging, silently recasting any errors. The student
ends the conversation whenever they like and sees a review of the errors
caught during the chat.

### Morgan — topic-led practice
Morgan leads a clear, engaging conversation on a topic drawn from the
database. Each topic has a vocabulary pool of useful words and phrases, a
difficulty level, and a focus keyword that keeps the conversation on subject.
Morgan:

- Speaks like a warm, clever podcast host (think Leo & Tina) — not a classroom teacher
- Models the topic's words and phrases naturally so the student hears and absorbs them
- Uses a varied range of vocabulary — never repeating the same one or two words every turn
- Adapts language complexity to the topic's difficulty level (e.g. simple sentences for A1 Beginner)
- Acknowledges what the student said and keeps the conversation flowing on topic
- Recasts errors silently as they come up

A Morgan session runs for a set number of exchanges (6 turns), then ends with a review.

### Topic intro
The `eec_topics` table has two intro-related fields:
- **`intro`** — A short written description shown on the dashboard
- **`intro_script`** — A spoken introduction that Morgan reads aloud via TTS at
  the start of every topic session (first visit and revisits)

When the topic starts, Morgan speaks the `intro_script` first, then her opening
line. This happens every time the topic is selected.

### The review
At the end of a session the app shows a **Conversation Review** — the whole
conversation laid out turn by turn: Morgan's lines, the student's original
lines, and a **Better** line wherever a sentence can be said more naturally.
Dora sessions show a simpler error review.

### The practice round (shadowing)
From the review, the student can start a **practice round** that replays the
finished conversation as exchanges (Morgan's line + student's line together):

1. **Exchange view**: Each exchange shows Morgan's line and the student's
   corrected line (with original shown below in smaller text if different)
2. **Auto-play**: Morgan's line plays automatically via TTS when the exchange loads
3. **Dual recording**: Student can record on either line:
   - **Shadow** button on Morgan's line — practice shadowing what Morgan said
   - **Practice** button on student's line — practice the corrected sentence
4. **Transcribe & Score**: Speech is transcribed via Whisper and scored for accuracy
5. **Word highlighting**: Matched words appear green; missed words appear red with strikethrough
6. **Try Again**: Each line has its own retry button after scoring
7. **Next**: Advances to the next exchange

Accuracy scoring uses word matching with normalization (handles contractions,
numbers, punctuation). Scores ≥90% show "Excellent!", ≥70% show "Almost there",
below 70% show "Try again". A completion modal appears after the final exchange.

The practice data lives only in the browser session — it is never saved to the database.

---

## Topics & Progression (Morgan)

Topics live in the database and are taught in sequence by `topic_order`.
Progression is **per topic** (not per word):

- When a student completes a session on a topic, that topic is marked as completed
- Auto-advance picks the next uncompleted topic in order
- Students can also choose any topic from the dashboard (including revisiting completed ones)
- If all topics are completed, auto-advance cycles back to the first topic

Learning is tracked per user so progress carries across sessions.

### Level adaptation
Morgan adapts her language complexity based on the topic's `level` field:

| Level | Morgan's style |
|-------|----------------|
| **Beginner** | Short, simple sentences, very common words, no idioms |
| **Elementary** | Simple sentences with a little more range, no idioms |
| **Intermediate** | Natural everyday English, common expressions, soft language ("a bit", "quite") |
| **Advanced** | Fuller, more nuanced language, wider vocabulary range |

---

## User System

The app supports user accounts with two roles:

| Role | Access |
|---|---|
| **Student** | Practice with coaches, view personal dashboard with session history, progress reports, and topic selection |
| **Teacher** | View all students, see individual student progress and session details |

Progress reports are generated automatically every 5 sessions, analyzing
vocabulary, phrasing, and sentence structure with scores and personalized feedback.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js (App Router) |
| Language | TypeScript + React |
| Styling | Tailwind CSS |
| Database | PostgreSQL (Supabase) via the Supabase JS client |
| Authentication | Cookie-based sessions with bcrypt password hashing |
| Dora conversation + sentence correction | Groq API (llama-3.1-8b-instant) |
| Morgan conversation + review + analysis | Groq API (llama-3.3-70b-versatile) |
| Speech-to-text | Groq Whisper turbo |
| Text-to-speech | Edge TTS (Microsoft neural voices) — see note below |
| Hosting | Render.com (Node) |

**TTS Note:** Edge TTS provides high-quality neural voices without an API key.
However, Microsoft may block requests from cloud/datacenter IPs (403 errors).
If this occurs, alternatives include Google Cloud TTS (requires API key) or
gTTS (lower quality but reliable).

---

## Database Tables

| Table | Description |
|---|---|
| `users` | User accounts with name, email, password hash, and role (student/teacher) |
| `sessions` | Completed conversation sessions per user, with topic and session number |
| `turns` | Individual turns within a session (Morgan's question, student's response, correction) |
| `session_analysis` | AI-generated analysis per session: vocabulary, phrasing, structure scores and notes |
| `progress_reports` | Aggregated reports generated every 5 sessions with overall progress assessment |
| `eec_topics` | Topics for Morgan: order, name, level, intro (dashboard text), intro_script (spoken TTS), opening line, vocabulary pool, focus keyword, and sample coach views |
| `eec_learning_log` | Records topic completion per user for progression tracking |
| `chat_state` | Server-side session storage for active conversations (avoids cookie size limits) |

---

## Project Structure

```
english-level-up/
├── app/
│   ├── page.tsx                    # Home (redirects to login or dashboard)
│   ├── login/page.tsx              # Login page
│   ├── register/page.tsx           # Registration page
│   ├── dashboard/page.tsx          # Student/teacher dashboard
│   ├── dashboard/student/[id]/     # Teacher view of individual student
│   ├── chat/page.tsx               # Main chat interface
│   ├── practice/page.tsx           # Practice round (shadowing)
│   └── api/                        # API routes
│       ├── auth/                   # Login, register, logout
│       ├── respond/                # Coach reply + per-turn correction
│       ├── transcribe/             # Whisper speech-to-text
│       ├── summary/                # Builds review + saves session + analysis
│       ├── set-style/              # Switch coach (Dora / Morgan)
│       ├── new/                    # Reset the session
│       ├── topics/                 # List all topics
│       └── tts/                    # Google Cloud Text-to-speech
├── components/
│   ├── chat/                       # ChatBox, ChatBubble, ChatInput, CoachToggle
│   ├── review/                     # ReviewModal
│   ├── practice/                   # ShadowingPractice, PracticeLine
│   ├── ui/                         # Button, Card, ProgressBar, RecordButton
│   └── LogoutButton.tsx            # Logout button component
├── hooks/                          # useSession, useChat, useAudioRecorder, useAudioPlayer
├── services/
│   ├── coach.ts                    # Dora and Morgan conversation logic
│   ├── correction.ts               # Sentence correction
│   ├── review.ts                   # Build conversation review markdown
│   ├── analysis.ts                 # Session analysis and progress reports
│   └── tts.ts                      # Google Cloud TTS integration
├── lib/
│   ├── supabase.ts                 # Supabase client
│   ├── groq.ts                     # Groq API client
│   ├── db.ts                       # Topic progression helpers
│   ├── auth.ts                     # Authentication helpers
│   └── chatSession.ts              # Server-side session management
├── types/                          # TypeScript types
├── utils/                          # audioUtils.ts, scoringUtils.ts
├── middleware.ts                   # Route protection
└── .env.local.example              # Example environment variables
```

---

## Getting Started

### Prerequisites
- Node.js 18+ (LTS recommended)
- A [Groq](https://console.groq.com) API key (free)
- A [Supabase](https://supabase.com) project (free)

### Database setup
Create the following tables in Supabase:

```sql
-- Users table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'student',
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  topic TEXT,
  topic_id INTEGER,
  session_number INTEGER,
  date TIMESTAMPTZ DEFAULT NOW()
);

-- Turns table
CREATE TABLE turns (
  id SERIAL PRIMARY KEY,
  session_id INTEGER REFERENCES sessions(id),
  turn_number INTEGER,
  app_question TEXT,
  student_speech TEXT,
  fluency_comment TEXT
);

-- Session analysis table
CREATE TABLE session_analysis (
  id SERIAL PRIMARY KEY,
  session_id INTEGER REFERENCES sessions(id),
  vocabulary_score INTEGER,
  vocabulary_note TEXT,
  phrasing_score INTEGER,
  phrasing_note TEXT,
  structure_score INTEGER,
  structure_note TEXT,
  overall_score INTEGER,
  overall_note TEXT,
  suggestion TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Progress reports table
CREATE TABLE progress_reports (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  report_number INTEGER,
  sessions_from INTEGER,
  sessions_to INTEGER,
  vocabulary_score INTEGER,
  vocabulary_label TEXT,
  vocabulary_description TEXT,
  phrasing_score INTEGER,
  phrasing_label TEXT,
  phrasing_description TEXT,
  structure_score INTEGER,
  structure_label TEXT,
  structure_description TEXT,
  overall_score INTEGER,
  overall_label TEXT,
  improvement_description TEXT,
  generated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Topics table
CREATE TABLE eec_topics (
  id SERIAL PRIMARY KEY,
  topic_order INTEGER,
  name TEXT,
  level TEXT,
  intro TEXT,           -- Written description for dashboard
  intro_script TEXT,    -- Spoken introduction (TTS) for Morgan
  opening TEXT,
  vocabulary_pool TEXT,
  focus_keyword TEXT,
  coach_views TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Learning log (topic completion tracking)
CREATE TABLE eec_learning_log (
  id SERIAL PRIMARY KEY,
  user_name TEXT,
  topic_id INTEGER,
  word_taught TEXT,
  had_error BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Server-side session state (avoids cookie size limits)
CREATE TABLE chat_state (
  user_id TEXT PRIMARY KEY,
  state JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/english-level-up.git
   cd english-level-up
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Create a `.env.local` file** based on `.env.local.example`
   ```
   GROQ_API_KEY=your_groq_api_key
   NEXT_PUBLIC_SUPABASE_URL=your_supabase_project_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   ```

4. **Run the dev server**
   ```bash
   npm run dev
   ```

5. Open `http://localhost:3000` in your browser

---

## Deploying to Render

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) -> **New** -> **Web Service**
3. Connect your GitHub repo
4. Set these environment variables (see below)
5. Set the build and start commands:
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm run start -- -p $PORT`
6. Deploy

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for the LLMs and Whisper transcription |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL (REST API URL, not PostgreSQL connection string) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key for server-side API routes |

**Note:** Edge TTS does not require an API key. If you switch to Google Cloud TTS,
add `GOOGLE_CLOUD_CREDENTIALS` with your service account JSON.

---

## The Recasting Technique

Recasting is a well-established second-language teaching technique where the
coach reformulates a learner's incorrect sentence correctly — without
explicitly pointing out the error. Learners acquire correct forms more
naturally when they hear them in context rather than being told a rule.

**Example:**
- Student says: *"Yesterday I go to the market"*
- Coach says: *"Oh nice, you went to the market yesterday? What did you pick up?"*

The student hears *"went"* used correctly, and the conversation continues
without interruption.

---

## License

MIT
