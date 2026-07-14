"""Async orchestration: mic -> VAD -> STT -> LLM -> TTS -> speaker, plus barge-in.

Design in one paragraph
-----------------------
A single coroutine (``_consume_mic``) reads every 512-sample frame and stays
live for the whole session -- this is what makes full-duplex possible. When the
endpointer fires, we launch the response as a *separate* task (``_respond``) so
the mic loop keeps running underneath it. While the agent is SPEAKING, sustained
user speech flushes the speaker, cancels the in-flight LLM/TTS task, and seeds a
fresh turn with the frames that triggered the interrupt. LLM tokens are batched
into sentences (``take_sentences``) and spoken one at a time, so we start talking
before the model has finished generating.

Barge-in fires only while SPEAKING (audio actually playing). Speech during
THINKING is treated as a *continuation* of the user's turn and buffered, which
pairs with semantic endpointing so a thinking pause doesn't get cut off.
"""

import asyncio
import logging
from collections import deque

import numpy as np

from . import config
from .audio import Microphone, Speaker
from .endpointing import SemanticEndpointer
from .llm import LLM
from .state import State, StateMachine
from .stt import STT
from .tts import TTS, take_sentences
from .vad import VAD, Endpointer
from metrics.latency import Timer, waterfall

log = logging.getLogger("pipeline")

_MAX_EXTENSIONS = 2  # how many times semantic endpointing may keep a turn open


class Pipeline:
    def __init__(self, save_waterfall=True):
        self.mic = Microphone()
        self.vad = VAD()
        self.endpointer = Endpointer()
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.speaker = Speaker(sample_rate=self.tts.sample_rate)
        self.sem = SemanticEndpointer()
        self.sm = StateMachine()
        self.timer = Timer()

        # capture state
        self._preroll = deque(maxlen=max(1, int(config.PREROLL_MS / config.FRAME_MS)))
        self._capturing = False
        self._capture = []          # frames of the current utterance
        self._extend_count = 0      # semantic-endpointing extensions used this turn

        # busy state (THINKING / SPEAKING)
        self._think_buffer = []     # frames heard while THINKING (possible continuation)
        self._barge_frames = 0      # consecutive speech frames heard while SPEAKING
        self._barge_seed = deque(maxlen=8)

        self._respond_task = None
        self._save_waterfall = save_waterfall
        self._saved_waterfall = False
        self._first_audio_reported = False

    # ------------------------------------------------------------------ run loop
    async def run(self):
        self.mic.start()
        self.speaker.start()
        try:
            async for frame in self.mic.frames():
                is_speech = self.vad.is_speech(frame)
                if self.sm.is_(State.LISTENING):
                    self._on_listen_frame(frame, is_speech)
                elif self.sm.is_(State.SPEAKING):
                    self._on_speaking_frame(frame, is_speech)
                elif self.sm.is_(State.THINKING):
                    self._think_buffer.append(frame)
                # INTERRUPTED is transient; handled synchronously in _trigger_barge_in
        finally:
            if self._respond_task and not self._respond_task.done():
                self._respond_task.cancel()
            self.mic.stop()
            self.speaker.stop()

    # ------------------------------------------------------------- LISTENING path
    def _on_listen_frame(self, frame, is_speech):
        self._preroll.append(frame)

        if not self._capturing:
            if is_speech:
                self._start_capture(list(self._preroll))
            return

        self._capture.append(frame)
        if self.endpointer.update(is_speech):
            self.timer.mark("endpoint")
            audio = np.concatenate(self._capture)
            self._capturing = False
            self.sm.to(State.THINKING)
            self._think_buffer = []
            self._respond_task = asyncio.create_task(self._respond(audio))

    def _start_capture(self, seed_frames):
        self._capturing = True
        self._capture = list(seed_frames)
        self.endpointer.reset()
        self.timer.reset()
        self.timer.mark("speech_start")

    # -------------------------------------------------------------- SPEAKING path
    def _on_speaking_frame(self, frame, is_speech):
        if is_speech:
            self._barge_seed.append(frame)
            self._barge_frames += 1
            if self._barge_frames * config.FRAME_MS >= config.BARGE_IN_SPEECH_MS:
                self._trigger_barge_in()
        else:
            self._barge_frames = 0
            self._barge_seed.clear()

    def _trigger_barge_in(self):
        log.info("BARGE-IN detected")
        self.sm.to(State.INTERRUPTED)
        self.speaker.flush()  # stop audio immediately
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        # Seed the new turn with the frames that triggered the interrupt.
        seed = list(self._barge_seed)
        self._barge_seed.clear()
        self._barge_frames = 0
        self._extend_count = 0
        self.sm.to(State.LISTENING)
        self._start_capture(seed)
        self._capturing = True

    # -------------------------------------------------------------- response path
    async def _respond(self, audio):
        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self.stt.transcribe, audio)
            self.timer.mark("stt_done")

            if not text:
                self._to_listening()
                return

            # Semantic endpointing: if the user seems mid-thought, keep listening.
            if config.USE_SEMANTIC_ENDPOINTING and self._extend_count < _MAX_EXTENSIONS:
                if not await self.sem.is_complete(text):
                    self._extend_turn(audio)
                    return

            print(f"USER: {text}")
            await self._speak_reply(text)
            await self._drain_playback()

            if self._save_waterfall and not self._saved_waterfall:
                path = waterfall(self.timer.marks)
                if path:
                    print(f"[saved latency waterfall -> {path}]")
                    self._saved_waterfall = True

            self._extend_count = 0
            self._to_listening()

        except asyncio.CancelledError:
            # Barge-in cancelled us; _trigger_barge_in already reset the state.
            raise

    async def _speak_reply(self, text):
        buffer = ""
        printed = ""
        async for token in self.llm.stream(text):
            self.timer.mark("llm_first_token")
            buffer += token
            sentences, buffer = take_sentences(buffer)
            for sentence in sentences:
                await self._speak(sentence)
                printed += sentence + " "
        tail = buffer.strip()
        if tail:
            await self._speak(tail)
            printed += tail
        if printed:
            print(f"AGENT: {printed.strip()}")

    async def _speak(self, sentence):
        audio = await self.tts.synth(sentence)
        # If a barge-in (or turn end) landed while we were synthesizing, drop it.
        if not (self.sm.is_(State.THINKING) or self.sm.is_(State.SPEAKING)):
            return
        if self.sm.is_(State.THINKING):
            self.sm.to(State.SPEAKING)
            self._think_buffer = []  # entering SPEAKING; buffered frames aren't continuation
        self.timer.mark("tts_first_audio")
        if not self._first_audio_reported:
            self.timer.print_report()
            self._first_audio_reported = True
        self.speaker.play(audio)

    async def _drain_playback(self):
        while self.speaker.is_active():
            await asyncio.sleep(0.02)

    # ------------------------------------------------------------------- helpers
    def _extend_turn(self, audio):
        """Semantic endpointing said 'not done' -> resume capturing this turn."""
        self._extend_count += 1
        log.info("semantic endpointing: incomplete, extending turn (%d)", self._extend_count)
        # Keep what we had, plus anything heard while we were transcribing.
        frames = [audio] + list(self._think_buffer)
        self._think_buffer = []
        self.timer.marks.pop("endpoint", None)  # re-measure from the final endpoint
        self.timer.marks.pop("stt_done", None)
        self.sm.to(State.LISTENING)
        self._start_capture([])
        self._capture = frames
        self._capturing = True

    def _to_listening(self):
        self._capturing = False
        self._capture = []
        self._think_buffer = []
        self._first_audio_reported = False
        self.sm.to(State.LISTENING)
