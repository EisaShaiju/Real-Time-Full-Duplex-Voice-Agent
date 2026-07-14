"""Voice activity detection and turn endpointing.

``VAD`` wraps the Silero ONNX model (tiny, <1 ms per 512-sample frame on CPU).
``Endpointer`` turns a stream of speech/non-speech decisions into a single
"the user has finished their turn" signal based on trailing silence, with a
minimum-speech gate so coughs and clicks don't open a turn.
"""

import numpy as np
import torch
from silero_vad import load_silero_vad

from . import config


class VAD:
    def __init__(self, threshold=config.VAD_THRESHOLD, sample_rate=config.SAMPLE_RATE):
        self.model = load_silero_vad()
        self.threshold = threshold
        self.sample_rate = sample_rate

    def speech_prob(self, frame):
        """Probability that a 512-sample frame contains speech."""
        x = torch.from_numpy(np.asarray(frame, dtype="float32"))
        with torch.no_grad():
            return float(self.model(x, self.sample_rate).item())

    def is_speech(self, frame):
        return self.speech_prob(frame) >= self.threshold


class Endpointer:
    """Detects the end of a user turn from a run of trailing silence.

    Feed it one boolean per frame via ``update(is_speech)``. It returns True on
    the frame where enough silence has accumulated *after* real speech began.
    """

    def __init__(
        self,
        frame_ms=config.FRAME_MS,
        silence_ms=config.SILENCE_ENDPOINT_MS,
        min_speech_ms=config.MIN_SPEECH_MS,
    ):
        self.silence_frames_needed = max(1, int(silence_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.reset()

    def reset(self):
        self._speech_frames = 0
        self._silence_frames = 0
        self._triggered = False
        self._fired = False

    def update(self, is_speech: bool) -> bool:
        if self._fired:
            return False  # latched; caller must reset() for the next turn
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            if self._speech_frames >= self.min_speech_frames:
                self._triggered = True
        elif self._triggered:
            self._silence_frames += 1
            if self._silence_frames >= self.silence_frames_needed:
                self._fired = True
                return True
        return False
