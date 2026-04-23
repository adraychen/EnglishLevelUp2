import os
import io
import tempfile
import base64
import random
import streamlit as st
from groq import Groq
from gtts import gTTS
from crewai import Agent, Task, Crew, Process
from audio_recorder_streamlit import audio_recorder

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="English Fluency Coach", page_icon="🎓")
st.title("🎓 English Fluency Coach")
st.caption("Answer my question out loud — I'll help you sound more natural!")

# ── API key ───────────────────────────────────────────────────────────────────
groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("API Key missing! Please set GROQ_API_KEY in Streamlit Secrets.")
    st.stop()
os.environ["GROQ_API_KEY"] = groq_key
groq_client = Groq(api_key=groq_key)

# ── Agent ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_agent():
    return Agent(
        role="English Fluency Coach",
        goal="Help the student sound more natural and fluent in everyday spoken English.",
        backstory="""You are a friendly English fluency coach helping non-native speakers
        sound more natural in everyday conversation.

        Your focus is spoken fluency — how a fluent English speaker would naturally say something
        in real conversation. You are NOT a grammar checker.

        When the student gives a response:

        1. If the sentence sounds UNNATURAL or STIFF:
           - Start with "A more natural way to say this is: ..."
           - Add one sentence explaining why. Focus on word choice or phrasing, not grammar rules.
           - Maximum 3 sentences total.

        2. If the sentence already sounds NATURAL and FLUENT:
           - Give a short warm acknowledgement like "That sounds great!" or "Very natural!"
           - Do NOT suggest rewrites or alternatives.

        Examples:
        - "I ate rice for breakfast" → "I had rice for breakfast"
        - "I am going to the market for buying vegetables" → "I'm going to the market to buy some vegetables"
        - "I work in a school from 5 years" → "I've been working at a school for 5 years"

        Always be warm, encouraging, and brief.""",
        llm="groq/llama-3.3-70b-versatile",
        verbose=False,
    )

# ── Topic pool ────────────────────────────────────────────────────────────────
ALL_PROMPTS = [
    "Tell me what you did this morning.",
    "Describe your favourite food.",
    "Tell me about your school or workplace.",
    "What do you like to do on weekends?",
    "Describe a place you would like to visit.",
    "Tell me about a hobby you enjoy.",
    "Describe your neighbourhood.",
    "Tell me about a movie or show you watched recently.",
    "What did you have for dinner last night?",
    "Tell me about someone important in your life.",
    "Describe your daily morning routine.",
    "What do you usually do after work or school?",
    "Tell me about a trip you took.",
    "Describe your favourite season and why you like it.",
    "What kind of music do you enjoy?",
    "Tell me about a goal you have for this year.",
    "Describe a typical weekend for you.",
    "What is something you are good at?",
    "Tell me about a challenge you faced recently.",
    "Describe your favourite place to relax.",
    "What do you usually do when you feel stressed?",
    "Tell me about a book you read or are reading.",
    "Describe your best friend.",
    "What is something new you learned recently?",
    "Tell me about your hometown.",
]
QUESTIONS_PER_SESSION = 5

# ── Helpers ───────────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        with open(f.name, "rb") as af:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=af,
                language="en",
            )
    return result.text.strip()

def get_feedback(sentence: str) -> str:
    task = Task(
        description=(
            f"The student said: \"{sentence}\"\n"
            f"Assess whether this sounds natural in spoken English. "
            f"If it needs improvement, offer a more natural version and explain why in one sentence. "
            f"If it already sounds fluent and natural, give a brief warm acknowledgement only."
        ),
        expected_output=(
            "Either: 'A more natural way to say this is: <version>. <one sentence explanation>.' "
            "Or: A short warm acknowledgement if the sentence is already natural."
        ),
        agent=get_agent(),
    )
    crew = Crew(agents=[get_agent()], tasks=[task], process=Process.sequential)
    return crew.kickoff().raw

def make_audio_b64(text: str) -> str:
    tts = gTTS(text=text, lang="en", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def autoplay_audio(b64: str):
    st.markdown(
        f'<audio autoplay style="display:none">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
        unsafe_allow_html=True,
    )

# ── Session state ─────────────────────────────────────────────────────────────
if "prompts" not in st.session_state:
    st.session_state.prompts = random.sample(ALL_PROMPTS, QUESTIONS_PER_SESSION)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_index" not in st.session_state:
    st.session_state.prompt_index = 0
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = True
# States: "waiting_for_answer" | "showing_feedback"
if "state" not in st.session_state:
    st.session_state.state = "waiting_for_answer"
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None

# ── Audio toggle ──────────────────────────────────────────────────────────────
col_tog, _ = st.columns([1, 5])
with col_tog:
    label = "🔊 Audio ON" if st.session_state.audio_enabled else "🔇 Audio OFF"
    if st.button(label, use_container_width=True):
        st.session_state.audio_enabled = not st.session_state.audio_enabled
        st.rerun()

st.divider()

current_index = st.session_state.prompt_index
all_done = current_index >= QUESTIONS_PER_SESSION

# ── Load current question into chat if not yet shown ─────────────────────────
if not all_done and st.session_state.state == "waiting_for_answer":
    question = st.session_state.prompts[current_index]
    # Only add if not already the last message
    if not st.session_state.messages or st.session_state.messages[-1].get("content") != question:
        audio_b64 = make_audio_b64(question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": question,
            "audio_b64": audio_b64,
        })
        st.session_state.pending_audio = audio_b64

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Play pending audio (one shot) ────────────────────────────────────────────
if st.session_state.audio_enabled and st.session_state.pending_audio:
    autoplay_audio(st.session_state.pending_audio)
    st.session_state.pending_audio = None

# ── All done ──────────────────────────────────────────────────────────────────
if all_done:
    st.success("✅ Great practice! You've completed all questions.")
    if st.button("Start over with new questions"):
        for key in ["messages", "prompt_index", "state", "pending_audio", "prompts"]:
            del st.session_state[key]
        st.rerun()
    st.stop()

# ── Progress ──────────────────────────────────────────────────────────────────
st.caption(f"Question {current_index + 1} of {QUESTIONS_PER_SESSION}")
st.progress(current_index / QUESTIONS_PER_SESSION)
st.markdown("---")

# ── STATE: waiting for answer ─────────────────────────────────────────────────
if st.session_state.state == "waiting_for_answer":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**🎤 Tap to start · tap again to stop**")
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e85d04",
            neutral_color="#6c757d",
            icon_size="2x",
            pause_threshold=3.0,
            key=f"recorder_{current_index}",
        )
    with col2:
        st.markdown("**⌨️ Or type your answer:**")
        text_input = st.chat_input("Type your answer here...")

    def handle_answer(user_text: str):
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.spinner("Listening..."):
            feedback = get_feedback(user_text)
        audio_b64 = make_audio_b64(feedback)
        st.session_state.messages.append({
            "role": "assistant",
            "content": feedback,
            "audio_b64": audio_b64,
        })
        # Queue feedback audio and switch to feedback state
        st.session_state.pending_audio = audio_b64
        st.session_state.state = "showing_feedback"
        st.rerun()

    if audio_bytes and len(audio_bytes) > 1000:
        with st.spinner("Transcribing..."):
            try:
                user_text = transcribe(audio_bytes)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                user_text = None
        if user_text:
            handle_answer(user_text)

    elif text_input:
        handle_answer(text_input)

# ── STATE: showing feedback — wait for user to press Next ────────────────────
elif st.session_state.state == "showing_feedback":
    if st.button("Next question →", use_container_width=False, type="primary"):
        st.session_state.prompt_index += 1
        st.session_state.state = "waiting_for_answer"
        st.rerun()
