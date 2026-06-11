import { generateSpeech as edgeTTS } from '@bestcodes/edge-tts';

export interface TTSOptions {
  text: string;
  languageCode?: string;
  voiceName?: string;
  speakingRate?: number;
}

/**
 * Generate speech audio using Edge TTS (Microsoft's free neural TTS).
 * Returns base64-encoded MP3 audio.
 *
 * Much faster than Google Cloud TTS - no API key or authentication required.
 * Uses the same high-quality neural voices as Azure TTS.
 */
export async function generateSpeech({
  text,
  voiceName = 'en-US-JennyNeural',
  speakingRate = 1.0,
}: TTSOptions): Promise<string> {
  if (!text || text.length === 0) {
    return '';
  }

  try {
    // Convert speakingRate (0.5-2.0) to Edge TTS rate format
    // Edge TTS uses percentage: -50% to +100%
    const ratePercent = Math.round((speakingRate - 1) * 100);
    const rate = ratePercent >= 0 ? `+${ratePercent}%` : `${ratePercent}%`;

    const audioBuffer = await edgeTTS({
      text,
      voice: voiceName,
      rate,
    });

    // Convert Uint8Array to base64
    return Buffer.from(audioBuffer).toString('base64');
  } catch (error) {
    console.error('Edge TTS error:', error);
    return '';
  }
}

// Voice presets for different coaches
// Edge TTS US English voices: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
export const VOICES = {
  // Morgan - clear, professional female voice (slower for learners)
  morgan: {
    voiceName: 'en-US-JennyNeural', // Professional, clear female voice
    speakingRate: 0.85, // Slower for learners
  },
  // Dora - casual, friendly female voice
  dora: {
    voiceName: 'en-US-AriaNeural', // Casual, expressive female voice
    speakingRate: 0.95,
  },
};
