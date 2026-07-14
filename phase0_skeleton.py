"""Phase 0 smoke test: blocking, turn-based, one file.

Proves every dependency works end to end before you touch async. Record until
~1 s of silence, transcribe, ask the LLM, speak the reply. No barge-in, no
streaming -- just "say something, hear an answer."

    python phase0_skeleton.py
"""

import os
import queue

import numpy as np
import sounddevice as sd

from agent.config import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE
from agent.llm import LLM
from agent.stt import STT
from agent.tts import TTS
from agent.vad import VAD

SILENCE_MS_TO_STOP = 1000


def record_utterance(vad):
    """Block until the user speaks and then goes quiet for ~1 s."""
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def cb(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    frames, started, silence = [], False, 0
    need = int(SILENCE_MS_TO_STOP / FRAME_MS)
    print("(speak now...)")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES, callback=cb):
        while True:
            frame = q.get()
            speech = vad.is_speech(frame)
            if speech:
                started, silence = True, 0
                frames.append(frame)
            elif started:
                silence += 1
                frames.append(frame)
                if silence >= need:
                    break
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


def main():
    assert os.environ.get("GROQ_API_KEY"), "Set GROQ_API_KEY first."
    vad, stt, llm, tts = VAD(), STT(), LLM(), TTS()
    speaker = sd.OutputStream(samplerate=tts.sample_rate, channels=1, dtype="float32")
    speaker.start()

    import asyncio

    try:
        while True:
            audio = record_utterance(vad)
            text = stt.transcribe(audio)
            if not text:
                continue
            print(f"USER : {text}")
            reply = "".join(asyncio.run(_collect(llm.stream(text))))
            print(f"AGENT: {reply}")
            speaker.write(asyncio.run(tts.synth(reply)))
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        speaker.stop()
        speaker.close()


async def _collect(agen):
    return [tok async for tok in agen]


if __name__ == "__main__":
    main()
