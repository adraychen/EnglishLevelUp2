# 🗣️ English Level Up

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
| 💬 Dora | Casual | Canadian English | Free, natural conversation on any subject. Dora chats like a friend, recasts errors silently, and uses rich everyday vocabulary. Good for learners who want exposure to natural native speech. |
| 📖 Morgan | Clear, topic-led | US English | Structured practice on a set topic. Morgan hosts a clear, engaging conversation — like an English-learning podcast host — modelling useful words and phrases at the topic's difficulty level. Good for learners building everyday vocabulary and confidence. |

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
- Adapts language complexity to the topic's difficulty level (e.g. simple sentences for A1 Beginner)
- Acknowledges what the student said and keeps the conversation flowing on topic
- Recasts errors silently as they come up

A Morgan session runs for a set number of exchanges, then ends with a review.

### The review
At the end of a session the app shows a **Conversation Review** — the whole
conversation laid out turn by turn: Morgan's lines, the student's original
lines, and a **✓ Better** line wherever a sentence can be said more naturally.
Dora sessions show a simpler error review.

The words Morgan used are recorded behind the scenes so future sessions can
introduce new vocabulary rather than repeating what's already been covered.

### The practice round (shadowing)
From the review, the student can start a **practice round** that replays the
finished conversation line by line: Morgan's lines play as audio (more exposure
to natural phrasing), and the student practises saying their own lines using
the corrected version. This gives a second pass of exposure plus a chance to
rehearse the better sentences out loud. The practice data lives only in the
browser session — it is never saved to the database.

---

## Topics & Progression (Morgan)

Topics live in the database and are taught in sequence. A topic has a pool of
vocabulary, so one topic can span several sessions — each session introduces
fresh words. When a topic's vocabulary has been covered, Morgan moves on to
the next topic. Learning is tracked per user so progress carries across
sessions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | Next.js (App Router) |
| Language | TypeScript + React |
| Styling | Tailwind CSS |
| Database | PostgreSQL (Supabase) via the Supabase JS client |
| Dora conversation + sentence correction | Groq API (llama-3.1-8b-instant) |
| Morgan conversation + review | Groq API (llama-3.3-70b-versatile) |
| Speech-to-text | Groq Whisper turbo |
| Text-to-speech | Google Translate TTS (free, Canadian English for Dora, US English for Morgan) |
| Hosting | Render.com (Node) |

---

## Database Tables

| Table | Description |
|---|---|
| `eec_topics` | Topics for Morgan: order, name, level, opening line, vocabulary pool, focus keyword, and sample coach views |
| `eec_learning_log` | Records which words each user has practised, per topic, for cross-session progression |

---

## Project Structure

```
english-level-up/
├── app/
│   ├── page.tsx              # Home (coach selection)
│   ├── chat/page.tsx         # Main chat interface
│   ├── practice/page.tsx     # Practice round (shadowing)
│   └── api/                  # API routes
│       ├── respond/          # Coach reply + per-turn correction
│       ├── transcribe/       # Whisper speech-to-text
│       ├── summary/          # Builds the conversation review
│       ├── set-style/        # Switch coach (Dora / Morgan)
│       ├── new/              # Reset the session
│       └── tts/              # Text-to-speech audio
├── components/
│   ├── chat/                 # ChatBox, ChatBubble, ChatInput, CoachToggle
│   ├── review/               # ReviewModal
│   ├── practice/             # ShadowingPractice, PracticeLine
│   └── ui/                   # Button, Card, ProgressBar, RecordButton
├── hooks/                    # useSession, useChat, useAudioRecorder, useAudioPlayer
├── services/                 # coach.ts, correction.ts, review.ts
├── lib/                      # supabase.ts, groq.ts, db.ts
├── types/                    # TypeScript types
├── utils/                    # audioUtils.ts
├── sql/
│   ├── setup_tables.sql      # Creates eec_topics and eec_learning_log + first topic
│   └── add_focus_keyword.sql # Adds the focus_keyword column to eec_topics
└── .env.local.example        # Example environment variables
```

---

## Getting Started

### Prerequisites
- Node.js 18+ (LTS recommended)
- A [Groq](https://console.groq.com) API key (free)
- A [Supabase](https://supabase.com) project (free)

### Database setup
Run `sql/setup_tables.sql` in the Supabase SQL editor to create the tables and
the first topic, then run `sql/add_focus_keyword.sql` to add the focus keyword
column.

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
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Set these environment variables (see below)
5. Set the build and start commands:
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm start`
6. Deploy

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key for the LLMs and Whisper transcription |
| `NEXT_PUBLIC_SUPABASE_URL` | Your Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase key used by the server-side API routes to read topics and write the learning log |

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

## Roadmap

- User login (currently single-user; learning is tracked under one name)
- More topics across more difficulty levels
- Saving conversation history and practice progress per user (after login)

---

## License

MIT
