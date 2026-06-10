import { TextToSpeechClient } from '@google-cloud/text-to-speech';

let client: TextToSpeechClient | null = null;

function getClient(): TextToSpeechClient {
  if (!client) {
    // Check for credentials
    const credentials = process.env.GOOGLE_CLOUD_CREDENTIALS;

    if (credentials) {
      // Parse JSON credentials from environment variable
      const parsed = JSON.parse(credentials);
      client = new TextToSpeechClient({ credentials: parsed });
    } else if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      // Use file path (for local development)
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
  voiceName = 'en-US-Journey-F', // Natural female voice
  speakingRate = 0.95, // Slightly slower for learners
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
      // audioContent is Uint8Array, convert to base64
      const buffer = Buffer.from(response.audioContent as Uint8Array);
      return buffer.toString('base64');
    }

    return '';
  } catch (error) {
    console.error('Google Cloud TTS error:', error);
    return '';
  }
}

// Voice presets for different coaches
export const VOICES = {
  // Morgan - clear, professional female voice (slower for learners)
  morgan: {
    languageCode: 'en-US',
    voiceName: 'en-US-Journey-F',
    speakingRate: 0.85,
  },
  // Dora - casual, friendly female voice
  dora: {
    languageCode: 'en-US',
    voiceName: 'en-US-Studio-O',
    speakingRate: 0.9,
  },
};
