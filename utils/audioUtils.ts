// Helper to write string to DataView
const writeString = (view: DataView, offset: number, string: string) => {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
};

export const pcmToWavBlobUrl = (base64PCM: string): string => {
  const binaryString = atob(base64PCM);
  const len = binaryString.length;
  const buffer = new ArrayBuffer(44 + len);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + len, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 24000, true);
  view.setUint32(28, 24000 * 1 * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, len, true);

  const bytes = new Uint8Array(buffer, 44);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const blob = new Blob([view], { type: 'audio/wav' });
  return URL.createObjectURL(blob);
};

export const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      const base64 = result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

// Singleton AudioContext to prevent "Max AudioContexts" error
let globalAudioContext: AudioContext | null = null;

const getAudioContext = () => {
  if (!globalAudioContext) {
    globalAudioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  }
  return globalAudioContext;
};

export const playAudioFromBase64 = async (base64PCM: string): Promise<void> => {
  const ctx = getAudioContext();

  if (ctx.state === 'suspended') {
    await ctx.resume();
  }

  const binaryString = atob(base64PCM);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  const shorts = new Int16Array(bytes.buffer);
  const audioBuffer = ctx.createBuffer(1, shorts.length, 24000);
  const channel = audioBuffer.getChannelData(0);

  for (let i = 0; i < shorts.length; i++) {
    channel[i] = shorts[i] / 32768.0;
  }

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.start(0);
};

// Play MP3 audio from base64 (for gTTS output)
export const playMp3FromBase64 = (base64Mp3: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    if (!base64Mp3) {
      resolve();
      return;
    }
    const audio = new Audio(`data:audio/mp3;base64,${base64Mp3}`);
    audio.onended = () => resolve();
    audio.onerror = () => reject(new Error('Audio playback failed'));
    audio.play().catch(reject);
  });
};
