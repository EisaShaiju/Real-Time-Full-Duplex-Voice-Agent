"""Low-level audio I/O built on sounddevice.

Two objects:

* ``Microphone`` - opens an input stream in fixed 512-sample blocks (one Silero
  VAD window) and exposes them as an async generator.
* ``Speaker``    - opens an output stream fed from an in-memory buffer.  The key
  method is ``flush()``: it drops the buffer instantly so barge-in can silence
  the agent mid-word.

sounddevice runs its callbacks on a PortAudio thread, so everything shared with
the asyncio loop is guarded by a plain ``threading`` primitive and bridged with
``run_in_executor``.
"""

import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd

from . import config


class Microphone:
    """Streams mono float32 frames of ``FRAME_SAMPLES`` samples at 16 kHz."""

    def __init__(self, sample_rate=config.SAMPLE_RATE, frame_samples=config.FRAME_SAMPLES):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):  # PortAudio thread
        # status carries xrun warnings; we keep going rather than crash a live call
        self._q.put(indata[:, 0].copy())

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def frames(self):
        """Yield frames forever. Runs the blocking queue get in a thread."""
        loop = asyncio.get_running_loop()
        while True:
            frame = await loop.run_in_executor(None, self._q.get)
            yield frame


class Speaker:
    """Buffered playback that can be flushed instantly for barge-in."""

    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._buf = np.zeros(0, dtype="float32")
        self._lock = threading.Lock()
        self._stream = None

    def _callback(self, outdata, frames, time_info, status):  # PortAudio thread
        with self._lock:
            n = min(frames, len(self._buf))
            outdata[:n, 0] = self._buf[:n]
            outdata[n:, 0] = 0.0
            self._buf = self._buf[n:]

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, samples):
        """Queue float32 samples (range -1..1) for playback."""
        samples = np.asarray(samples, dtype="float32")
        with self._lock:
            self._buf = np.concatenate([self._buf, samples])

    def flush(self):
        """Drop everything queued -> playback stops on the next callback (~ms)."""
        with self._lock:
            self._buf = np.zeros(0, dtype="float32")

    def is_active(self):
        with self._lock:
            return len(self._buf) > 0
