# Topic Evaluation Guideline — English Level Up

Use this to decide whether a podcast episode (or any source) will make a good Morgan
topic, and at what level. The goal of the app is **conversational fluency through
exposure** — Morgan models natural language in conversation; she does not drill or lecture.
A topic is "good" when it plays to that strength.

---

## The core test: is it a CONTENT topic or a SKILL topic?

This is the single most important question.

- ✅ **Content topics** describe a SUBJECT the student and Morgan can talk about — feelings,
  personality, food, travel, hobbies, daily routine, work, family, weather, shopping.
  Morgan can naturally model the vocabulary by chatting about the subject. **These fit the app.**

- ❌ **Skill / strategy topics** teach a TECHNIQUE for how to converse — "what to say when
  you're stuck," "how to buy time," "how to keep a conversation going," "how to pass the
  question back." These need the student in the *answerer's seat* practising the technique,
  which Morgan (as the host/asker) cannot demonstrate naturally. **These do NOT fit well.**

Quick way to tell: if the topic is about *a thing you talk about*, it's content (good).
If it's about *how to talk*, it's a skill (poor fit — Morgan can't demonstrate it from the
host chair, and forcing the phrases in produces awkward, misused language).

> Lesson learned: the "Speaking when you don't know what to say" topic failed this test.
> Morgan modelled the reaction words but couldn't authentically demonstrate the buy-time and
> pass-the-ball techniques, because those belong to the person being questioned.

---

## Checklist for a good topic

A strong topic should have most of these:

1. **It's a content/subject topic** (passes the core test above).
2. **Concrete vocabulary** — a clear set of words a learner can hear, absorb, and reuse.
   Single words and short phrases are ideal (e.g. happy, nervous; organized, easygoing).
3. **Natural sentence patterns** that fit real conversation, NOT mechanical frames.
   Good: "I'm [adjective] because [reason]", "When I was younger I was X, but now I'm Y."
   These are patterns Morgan can model by example without drilling.
4. **Morgan can demonstrate everything from her own seat** — she can describe her own
   feelings/personality/weekend and invite the student to do the same. If the key skill
   requires the STUDENT to be the one practising it live, it's a poor fit.
5. **Everyday usefulness** — the language is useful in normal daily conversation, not only
   in a narrow context (e.g. an exam). Exam-framed episodes can still work if the underlying
   language is everyday — just drop the exam framing.
6. **Level-appropriate** — the language sits cleanly at one of the app's levels (below).

If a topic is mostly a skill topic, either skip it or keep it but place it at a high
`topic_order` so it doesn't lead the sequence.

---

## Splitting an episode

- Split into multiple topics ONLY when an episode genuinely covers two subjects a learner
  would think of as separate (e.g. "ordering food" + "making a complaint" → two topics).
- Do NOT manufacture two thin topics from one coherent lesson. One good topic beats two weak
  ones.

---

## Levels (descriptive words, not CEFR codes)

Use these words in the database `level` field. The app shows them to learners and Morgan
adapts her language complexity to them.

- **Beginner** — very simple, very common words; short sentences; no idioms.
  (e.g. Talking about feelings)
- **Elementary** — simple everyday words; short, clear sentences; a little more range.
  (e.g. light everyday small talk)
- **Intermediate** — natural everyday English, common expressions, softening words
  ("a bit", "quite", "tend to"), simple contrast ("but", "however"); richer vocabulary.
  (e.g. Describing your personality)
- **Advanced** — fuller, more nuanced language and a wider vocabulary range, still clear
  and accessible. (reserved for future richer topics)

Pick the level by the LANGUAGE actually used, not by how the source is marketed. An
"IELTS" episode whose language is simple adjectives + softening words is **Intermediate**,
not advanced.

---

## Building the topic row (fields)

When a topic passes, create an `eec_topics` row with:

- **topic_order** — position in the auto-advance sequence (unique number; park weak topics high)
- **name** — short, learner-facing (drop exam/marketing framing)
- **level** — one of the descriptive words above
- **intro** — one descriptive sentence; used to frame the first session AND shown on the
  dashboard. NOT phrased "Today we're..." — describe what the topic is about.
- **opening** — Morgan's first spoken line that starts the chat
- **vocabulary_pool** — the words/phrases as GUIDANCE for Morgan (not a checklist), one per
  line; sentence patterns wrapped in [ ... ]
- **coach_views** — a few sample things Morgan might say, for inspiration
- **focus_keyword** — what Morgan keeps the conversation about (drives "keep the chat about
  ___" and her questions). For content topics this is the subject itself.

---

## One-line summary

**Good Morgan topic = a everyday SUBJECT with concrete vocabulary and natural patterns that
Morgan can model from her own seat. Avoid topics that teach a conversation TECHNIQUE the
student must practise live.**