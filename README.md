# Real-Time Full-Duplex Voice Agent

A voice-in / voice-out assistant you can **interrupt mid-sentence**. Streaming
STT → LLM → TTS with a measured latency budget and an honest robustness eval.

- **Sub-second** voice-to-voice latency (LLM tokens piped into TTS sentence-by-sentence)
- **Barge-in**: talk over the agent and it stops, listens, and responds to the new input
- **Semantic endpointing**: a thinking pause mid-sentence doesn't cut you off
- **Robustness report**: WER across noise levels + Whisper silence-hallucination suppression

```
  mic ─▶ VAD/endpoint (Silero) ─▶ STT (faster-whisper) ─▶ LLM (Groq, stream)
                │                                              │
   barge-in ────┘                                              ▼
  speaker ◀──── TTS (Piper, sentence-by-sentence) ◀───────────┘
```

Forward path: mic → VAD → STT → LLM → TTS → speaker.
Control path: while TTS plays, mic + VAD stay live; user speech flushes the
speaker, cancels the in-flight LLM/TTS task, and starts a fresh turn.

---

## Stack

| Component     | Choice                        | Why                                        |
| ------------- | ----------------------------- | ------------------------------------------ |
| STT           | `faster-whisper` (`base.en`)  | Fast local CTranslate2 backend, no API     |
| VAD/endpoint  | `silero-vad`                  | Tiny ONNX model, <1 ms per 32 ms frame     |
| LLM           | Groq (streaming)              | Very low time-to-first-token               |
| TTS           | `piper` (fallback `edge-tts`) | Fast neural TTS, streams per sentence      |
| Audio I/O     | `sounddevice`                 | Clean mic/speaker access on Windows        |
| Orchestration | `asyncio`                     | Non-blocking, cancellable stage pipeline   |
| Eval          | `jiwer`, `numpy`, `matplotlib`| WER, noise injection, latency waterfall    |

---

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download a Piper voice (creates en_US-lessac-medium.onnx + .json here)
python -m piper.download_voices en_US-lessac-medium

# Groq API key
$env:GROQ_API_KEY = "your_key_here"
```

If Piper fights the install for more than an hour, set `TTS_BACKEND = "edge"` in
`agent/config.py` and skip the voice download — `edge-tts` needs no local model.

> **Model note:** `agent/config.py` defaults to `llama-3.3-70b-versatile`, which
> is a non-reasoning model with the low time-to-first-token this budget needs.
> Groq announced deprecation of its Llama chat models in mid-2026; if you hit a
> "model not found" error, switch `LLM_MODEL` to `openai/gpt-oss-120b` and
> `ENDPOINT_LLM_MODEL` to `openai/gpt-oss-20b` (both current on Groq's models
> page). One line each.

---

## Run

```powershell
python phase0_skeleton.py   # smoke test: turn-based, blocking, proves deps work
python main.py              # the real thing: streaming, full-duplex, barge-in
```

Use **headphones** — otherwise the mic hears the agent's own voice and triggers
false barge-ins. Acoustic echo cancellation (AEC) is the known next step.

---

## Layout

```
voice-agent/
├── agent/
│   ├── config.py       # every tunable knob in one place
│   ├── audio.py        # sounddevice mic capture + interruptible speaker
│   ├── vad.py          # Silero wrapper + trailing-silence endpointer
│   ├── stt.py          # faster-whisper transcription
│   ├── llm.py          # Groq streaming client (cancellable)
│   ├── tts.py          # Piper / edge-tts + streaming sentence splitter
│   ├── endpointing.py  # semantic "is the thought complete?" check
│   ├── state.py        # LISTENING / THINKING / SPEAKING / INTERRUPTED
│   └── pipeline.py     # async orchestration + barge-in control path
├── eval/
│   ├── noise.py        # inject noise at a target SNR
│   ├── wer_test.py     # WER across SNR levels -> plot
│   └── hallucination.py# silence/noise -> Whisper phantom text, gated vs raw
├── metrics/
│   └── latency.py      # timestamp taps + waterfall chart
├── main.py             # live agent
├── phase0_skeleton.py  # Phase 0 smoke test
└── requirements.txt
```

---

## Evaluation

```powershell
# 1. Put a few <name>.wav + <name>.txt pairs in eval/data/
python -m eval.wer_test        # WER at clean / 20 / 10 / 5 / 0 dB -> wer_vs_snr.png
python -m eval.hallucination   # phantom-text rate: raw Whisper vs VAD-gated
```

The live agent writes `latency_waterfall.png` after the first full turn.

---

## How the pieces fit

**Streaming that actually cuts latency.** The LLM token stream is fed into
`take_sentences`, which pulls out complete sentences the moment they finish. The
first sentence is synthesized and playing while the model is still generating the
second — so the agent starts talking well before the reply is complete.

**Barge-in as an explicit state machine.** One coroutine reads every mic frame
for the whole session. While `SPEAKING`, sustained user speech (`BARGE_IN_SPEECH_MS`)
flushes the speaker buffer (audio stops in ~one callback), cancels the in-flight
`_respond` task (which tears down the LLM stream and pending TTS), and seeds a
fresh turn with the very frames that triggered the interrupt. Every transition is
logged — that log is where interrupt bugs live.

**Semantic endpointing.** Naive VAD ends a turn on any pause. When the user goes
quiet, we transcribe and ask "complete thought?" — a cheap heuristic for the easy
cases (ends in punctuation → done; ends on a filler → keep listening) and a fast
1-token Groq call only for the genuinely ambiguous ones. If incomplete, the turn
stays open instead of cutting the user off mid-sentence.

**Measured, not invented.** `metrics/latency.py` timestamps every stage; the
headline number is endpoint → first audio out. WER and hallucination rates come
from the eval harness. Every figure in the resume line below is reproducible live.

---

## Resume entry (fill in your measured numbers)

**Real-Time Voice Agent — Streaming STT + LLM + TTS**
- Built a full-duplex voice assistant (faster-whisper → Groq → Piper) with async
  streaming, ~X ms voice-to-voice latency by piping LLM tokens into TTS
  sentence-by-sentence.
- Implemented barge-in via a live VAD gate and an explicit turn state machine,
  cancelling in-flight LLM/TTS on user speech within ~X ms.
- Added semantic endpointing to prevent mid-thought cutoffs, and a robustness
  harness measuring WER across 0–20 dB SNR and suppressing Whisper
  silence-hallucinations.

---

## Gotchas

- **Echo** — mic + speaker together can make the agent hear itself. Headphones for
  demos; AEC is the real fix.
- **Piper on Windows** can be finicky — time-box it, else `TTS_BACKEND = "edge"`.
- **Don't claim numbers you didn't measure** — the whole point is that the latency
  and WER figures are real and reproducible.
