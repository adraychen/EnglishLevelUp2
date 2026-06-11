import { NextRequest } from 'next/server';
import { generateSpeech, splitIntoSentences, VOICES } from '@/services/tts';
import { getChatState } from '@/lib/chatSession';

/**
 * Streaming TTS endpoint using Server-Sent Events.
 * Sends audio chunks as they're generated so client can start playing immediately.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const text = (body.text || '').trim();

    if (!text) {
      return new Response('No text provided', { status: 400 });
    }

    // Get voice config based on session style
    const session = await getChatState();
    const style = session.style || 'clear';
    const voiceConfig = style === 'casual' ? VOICES.dora : VOICES.morgan;

    const sentences = splitIntoSentences(text);

    // Create a readable stream for SSE
    const stream = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();

        try {
          // Send total count first
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'start', total: sentences.length })}\n\n`)
          );

          // Generate and send each sentence's audio
          for (let i = 0; i < sentences.length; i++) {
            const audio = await generateSpeech({
              text: sentences[i],
              ...voiceConfig,
            });

            controller.enqueue(
              encoder.encode(
                `data: ${JSON.stringify({
                  type: 'chunk',
                  index: i,
                  audio,
                  text: sentences[i],
                })}\n\n`
              )
            );
          }

          // Signal completion
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'done' })}\n\n`)
          );
        } catch (error) {
          console.error('TTS stream error:', error);
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify({ type: 'error', message: 'TTS generation failed' })}\n\n`)
          );
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  } catch (error) {
    console.error('TTS stream route error:', error);
    return new Response('Server error', { status: 500 });
  }
}
