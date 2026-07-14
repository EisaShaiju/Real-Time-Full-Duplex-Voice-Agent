"""Speech-to-text using faster-whisper (CTranslate2 backend, fully local).

Whisper is not truly streaming, so the pipeline hands us a complete utterance
buffer once the endpointer fires and we transcribe it in one shot. Two settings
matter for a live agent:

* ``beam_size=1``                    -> greedy decode, lowest latency.
* ``condition_on_previous_text=False`` -> stops Whisper's tendency to loop /
  hallucinate carried-over phantom text on short clips.
"""

import numpy as np
from faster_whisper import WhisperModel

from . import config


class STT:
    def __init__(
        self,
        model=config.STT_MODEL,
        device=config.STT_DEVICE,
        compute_type=config.STT_COMPUTE_TYPE,
    ):
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio) -> str:
        """audio: 1-D float32 numpy at 16 kHz. Returns the recognised text."""
        audio = np.asarray(audio, dtype="float32")
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,
            vad_filter=False,  # we already gate with Silero upstream
            condition_on_previous_text=False,
        )
        return "".join(seg.text for seg in segments).strip()
