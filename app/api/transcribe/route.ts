import { NextRequest, NextResponse } from 'next/server';
import { getGroqClient, MODELS } from '@/lib/groq';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';
import { createReadStream } from 'fs';

export async function POST(request: NextRequest) {
  let tempFilePath: string | null = null;

  try {
    const contentType = request.headers.get('content-type') || '';

    let audioBuffer: Buffer;

    if (contentType.includes('audio/')) {
      // Raw audio bytes
      const arrayBuffer = await request.arrayBuffer();
      audioBuffer = Buffer.from(arrayBuffer);
    } else if (contentType.includes('multipart/form-data')) {
      // Form data with file
      const formData = await request.formData();
      const file = formData.get('audio') as File;
      if (!file) {
        return NextResponse.json({ error: 'No audio file provided' }, { status: 400 });
      }
      const arrayBuffer = await file.arrayBuffer();
      audioBuffer = Buffer.from(arrayBuffer);
    } else {
      return NextResponse.json({ error: 'Invalid content type' }, { status: 400 });
    }

    if (!audioBuffer || audioBuffer.length === 0) {
      return NextResponse.json({ error: 'No audio data' }, { status: 400 });
    }

    // Write to temp file (Groq SDK requires a file)
    const filename = `audio-${Date.now()}.wav`;
    tempFilePath = join(tmpdir(), filename);
    await writeFile(tempFilePath, audioBuffer);

    // Transcribe using Groq Whisper
    const client = getGroqClient();
    const transcription = await client.audio.transcriptions.create({
      model: MODELS.WHISPER,
      file: createReadStream(tempFilePath) as unknown as File,
      language: 'en',
    });

    // Clean up temp file
    await unlink(tempFilePath).catch(() => {});

    return NextResponse.json({ text: transcription.text.trim() });
  } catch (error) {
    console.error('Transcribe error:', error);

    // Clean up temp file on error
    if (tempFilePath) {
      await unlink(tempFilePath).catch(() => {});
    }

    return NextResponse.json({ error: 'Transcription failed' }, { status: 500 });
  }
}
