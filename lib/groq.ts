import Groq from 'groq-sdk';

// Initialize Groq client (server-side only)
export const getGroqClient = () => {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    throw new Error('GROQ_API_KEY environment variable is not set');
  }
  return new Groq({ apiKey });
};

// Model constants
export const MODELS = {
  // Fast model for Dora (casual conversation)
  DORA: 'llama-3.1-8b-instant',
  // More capable model for Morgan (topic-led practice)
  MORGAN: 'llama-3.3-70b-versatile',
  // Whisper model for transcription
  WHISPER: 'whisper-large-v3-turbo',
} as const;
