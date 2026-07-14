"""Whisper silence-hallucination probe.

Fed pure silence or non-speech noise, Whisper famously invents phantom text
("Thank you.", "Thanks for watching."). This measures how often that happens
raw, then shows the Silero VAD gate suppresses it -- because silence/noise never
passes the gate, so it never reaches Whisper in the live pipeline.

    python -m eval.hallucination
"""

import numpy as np

from agent.config import FRAME_SAMPLES, SAMPLE_RATE
from agent.stt import STT
from agent.vad import VAD


def _has_speech(vad, audio):
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        if vad.is_speech(audio[i : i + FRAME_SAMPLES]):
            return True
    return False


def run(trials=30, dur_s=3.0):
    stt, vad = STT(), VAD()
    rng = np.random.default_rng(0)
    n = int(dur_s * SAMPLE_RATE)

    conditions = {
        "pure silence": lambda: np.zeros(n, dtype="float32"),
        "white noise": lambda: (rng.standard_normal(n) * 0.02).astype("float32"),
    }

    print(f"\nHallucination probe ({trials} trials x {dur_s:g}s each)")
    print(f"  {'condition':<14} {'raw whisper':>12} {'VAD-gated':>12}")
    for name, gen in conditions.items():
        raw = gated = 0
        for _ in range(trials):
            audio = gen()
            if stt.transcribe(audio).strip():
                raw += 1
            if _has_speech(vad, audio) and stt.transcribe(audio).strip():
                gated += 1
        print(f"  {name:<14} {f'{raw}/{trials}':>12} {f'{gated}/{trials}':>12}")


if __name__ == "__main__":
    run()
