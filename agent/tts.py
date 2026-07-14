"""Text-to-speech.

Primary backend is Piper (fast, local, neural). edge-tts is the fallback for
when Piper's Windows install misbehaves -- same interface, decoded to PCM with
miniaudio so there's no ffmpeg dependency.

Both backends expose the same surface:
    tts.sample_rate            -> int, the rate of the returned audio
    await tts.synth(text)      -> float32 numpy in -1..1 at tts.sample_rate

``take_sentences`` is the streaming glue: the pipeline pushes LLM tokens into a
running buffer and pulls out complete sentences the moment they finish, so we
start speaking sentence one while the model is still generating sentence two.
"""

import asyncio
import re

import numpy as np

from . import config

# A sentence ends on . ! ? or ellipsis followed by whitespace or end-of-buffer.
_SENT_END = re.compile(r"[.!?\u2026]+(?=\s|$)")


def take_sentences(buffer: str):
    """Split completed sentences out of a growing text buffer.

    Returns ``(sentences, remainder)`` where ``remainder`` is the trailing
    partial sentence still being generated.
    """
    sentences, last = [], 0
    for m in _SENT_END.finditer(buffer):
        seg = buffer[last:m.end()].strip()
        if seg:
            sentences.append(seg)
        last = m.end()
    return sentences, buffer[last:]


class _Piper:
    def __init__(self, voice=config.PIPER_VOICE):
        from piper import PiperVoice

        # Accepts a bare name or a path; Piper wants the .onnx file.
        onnx = voice if voice.endswith(".onnx") else f"{voice}.onnx"
        self.voice = PiperVoice.load(onnx)
        self.sample_rate = self.voice.config.sample_rate

    def _synth_sync(self, text):
        pcm = bytearray()
        # Newer piper1-gpl: synthesize() yields AudioChunk objects.
        # Older piper-tts: synthesize_stream_raw() yields raw int16 bytes.
        if hasattr(self.voice, "synthesize"):
            try:
                for chunk in self.voice.synthesize(text):
                    pcm.extend(chunk.audio_int16_bytes)
            except (AttributeError, TypeError):
                pcm.clear()
                for raw in self.voice.synthesize_stream_raw(text):
                    pcm.extend(raw)
        else:
            for raw in self.voice.synthesize_stream_raw(text):
                pcm.extend(raw)
        return np.frombuffer(bytes(pcm), dtype=np.int16).astype("float32") / 32768.0

    async def synth(self, text):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._synth_sync, text)


class _Edge:
    def __init__(self, voice=config.EDGE_VOICE, sample_rate=config.EDGE_SAMPLE_RATE):
        self.voice = voice
        self.sample_rate = sample_rate

    async def synth(self, text):
        import edge_tts
        import miniaudio

        mp3 = bytearray()
        async for chunk in edge_tts.Communicate(text, self.voice).stream():
            if chunk["type"] == "audio":
                mp3.extend(chunk["data"])
        decoded = miniaudio.decode(
            bytes(mp3),
            output_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=self.sample_rate,
        )
        return np.asarray(decoded.samples, dtype="float32") / 32768.0


class TTS:
    def __init__(self, backend=config.TTS_BACKEND):
        self._impl = _Piper() if backend == "piper" else _Edge()
        self.sample_rate = self._impl.sample_rate

    async def synth(self, text):
        return await self._impl.synth(text)
