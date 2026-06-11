import { TextToSpeechClient } from '@google-cloud/text-to-speech';

let client: TextToSpeechClient | null = null;

function getClient(): TextToSpeechClient {
  if (!client) {
    const credentials = process.env.GOOGLE_CLOUD_CREDENTIALS;

    if (credentials) {
      const parsed = JSON.parse(credentials);
      client = new TextToSpeechClient({ credentials: parsed });
    } else if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      client = new TextToSpeechClient();
    } else {
      throw new Error('Google Cloud credentials not configured');
    }
  }
  return client;
}

export interface TTSOptions {
  text: string;
  languageCode?: string;
  voiceName?: string;
  speakingRate?: number;
}

/**
 * Generate speech audio using Google Cloud TTS.
 * Returns base64-encoded MP3 audio.
 */
export async function generateSpeech({
  text,
  languageCode = 'en-US',
  voiceName = 'en-US-Standard-C',
  speakingRate = 1.0,
}: TTSOptions): Promise<string> {
  if (!text || text.length === 0) {
    return '';
  }

  try {
    const ttsClient = getClient();

    const [response] = await ttsClient.synthesizeSpeech({
      input: { text },
      voice: {
        languageCode,
        name: voiceName,
      },
      audioConfig: {
        audioEncoding: 'MP3',
        speakingRate,
        pitch: 0,
      },
    });

    if (response.audioContent) {
      const buffer = Buffer.from(response.audioContent as Uint8Array);
      return buffer.toString('base64');
    }

    return '';
  } catch (error) {
    console.error('Google Cloud TTS error:', error);
    return '';
  }
}

/**
 * Split text into sentences for chunked generation.
 */
export function splitIntoSentences(text: string): string[] {
  // Split on sentence-ending punctuation followed by space or end
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 0);

  return sentences.length > 0 ? sentences : [text];
}

/**
 * Generate TTS for multiple sentences, returning as they complete.
 * Yields {index, audio} for each sentence.
 */
export async function* generateSpeechChunked(
  text: string,
  options: Omit<TTSOptions, 'text'>
): AsyncGenerator<{ index: number; audio: string; text: string }> {
  const sentences = splitIntoSentences(text);

  // Generate first sentence immediately for fast start
  if (sentences.length > 0) {
    const firstAudio = await generateSpeech({ text: sentences[0], ...options });
    yield { index: 0, audio: firstAudio, text: sentences[0] };
  }

  // Generate remaining sentences
  for (let i = 1; i < sentences.length; i++) {
    const audio = await generateSpeech({ text: sentences[i], ...options });
    yield { index: i, audio, text: sentences[i] };
  }
}

/**
 * Generate TTS for all sentences in parallel, return combined.
 * Faster than sequential but waits for all before returning.
 */
export async function generateSpeechParallel(
  text: string,
  options: Omit<TTSOptions, 'text'>
): Promise<string[]> {
  const sentences = splitIntoSentences(text);

  const audioPromises = sentences.map(sentence =>
    generateSpeech({ text: sentence, ...options })
  );

  return Promise.all(audioPromises);
}

// Voice presets - using Standard voices for speed
// Standard voices are faster to generate than WaveNet/Neural2/Journey
export const VOICES = {
  // Morgan - clear female voice (Standard for speed)
  morgan: {
    languageCode: 'en-US',
    voiceName: 'en-US-Standard-C', // Female, clear
    speakingRate: 0.9,
  },
  // Dora - casual female voice (Standard for speed)
  dora: {
    languageCode: 'en-US',
    voiceName: 'en-US-Standard-E', // Female, slightly different tone
    speakingRate: 0.95,
  },
};
