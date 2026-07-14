"""Central configuration for the voice agent.

Everything tunable lives here so the pipeline stays readable and the
latency/robustness knobs are in one place.
"""

# ---------------------------------------------------------------- audio format
SAMPLE_RATE = 16_000          # mic + STT + VAD all run at 16 kHz
CHANNELS = 1
FRAME_SAMPLES = 512           # 32 ms @ 16 kHz -- Silero VAD requires exactly this
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000.0

# ------------------------------------------------------------------ VAD / turns
VAD_THRESHOLD = 0.5           # speech probability above this counts as speech
SILENCE_ENDPOINT_MS = 700     # trailing silence that ends a user turn
MIN_SPEECH_MS = 200           # ignore blips shorter than this (clicks, breaths)
BARGE_IN_SPEECH_MS = 160      # sustained speech during playback -> interrupt
PREROLL_MS = 300              # audio kept before speech starts (avoids clipping)

# -------------------------------------------------------------------------- STT
STT_MODEL = "base.en"         # faster-whisper size; small.en for more accuracy
STT_DEVICE = "cpu"           # "cpu" | "cuda" | "auto"
STT_COMPUTE_TYPE = "int8"     # int8 on CPU, float16 on GPU

# -------------------------------------------------------------------------- LLM
# NOTE: as of Jun 2026 Groq announced deprecation of the llama-3.x chat models.
# They still respond today, but if you get a model-not-found error, switch to the
# recommended replacements below (they are current on Groq's models page):
#     LLM_MODEL          -> "openai/gpt-oss-120b"
#     ENDPOINT_LLM_MODEL -> "openai/gpt-oss-20b"
# Llama is the default here because it is a *non-reasoning* model with the
# sub-100 ms time-to-first-token this project's latency budget is built around.
LLM_MODEL = "llama-3.3-70b-versatile"
ENDPOINT_LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.6
SYSTEM_PROMPT = (
    "You are a concise, friendly voice assistant. Your replies are spoken aloud, "
    "so keep them short and conversational: one or two sentences, no lists, no "
    "markdown, no emoji. Get to the point."
)

# -------------------------------------------------------------------------- TTS
TTS_BACKEND = "piper"                 # "piper" | "edge"
PIPER_VOICE = "en_US-lessac-medium"   # expects <voice>.onnx (+ .onnx.json) on disk
EDGE_VOICE = "en-US-AriaNeural"
EDGE_SAMPLE_RATE = 22_050             # we decode edge-tts mp3 down to this rate

# --------------------------------------------------------------- endpointing
USE_SEMANTIC_ENDPOINTING = True       # confirm ambiguous pauses with a fast LLM call
