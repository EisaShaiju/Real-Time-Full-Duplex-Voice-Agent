"""Latency instrumentation.

``Timer`` records named timestamps across one voice-to-voice turn. The headline
number is endpoint -> first audio out: the gap between the user going quiet and
the agent starting to speak. Stage deltas break that down so you can see whether
STT, LLM time-to-first-token, or TTS is the bottleneck.

``waterfall`` renders the same marks as a horizontal timeline for the README /
CV. These are the numbers you quote -- measured, not invented.
"""

import time
from collections import OrderedDict

# Order the marks appear in a turn; deltas are computed between consecutive ones.
STAGE_ORDER = ["speech_start", "endpoint", "stt_done", "llm_first_token", "tts_first_audio"]


class Timer:
    def __init__(self):
        self.marks = OrderedDict()

    def reset(self):
        self.marks = OrderedDict()

    def mark(self, name):
        # First write wins (e.g. llm_first_token should latch on the first token).
        self.marks.setdefault(name, time.perf_counter())

    def voice_to_voice_ms(self):
        a, b = self.marks.get("endpoint"), self.marks.get("tts_first_audio")
        return (b - a) * 1000 if a and b else None

    def stage_deltas(self):
        present = [(n, self.marks[n]) for n in STAGE_ORDER if n in self.marks]
        return [(f"{n0} -> {n1}", (t1 - t0) * 1000) for (n0, t0), (n1, t1) in zip(present, present[1:])]

    def print_report(self):
        for name, ms in self.stage_deltas():
            print(f"  {name:<32} {ms:7.1f} ms")
        v2v = self.voice_to_voice_ms()
        if v2v is not None:
            print(f"  {'VOICE-TO-VOICE (endpoint->audio)':<32} {v2v:7.1f} ms")


def waterfall(marks, path="latency_waterfall.png"):
    """Render an OrderedDict of name->perf_counter marks as a waterfall PNG."""
    import matplotlib.pyplot as plt

    present = [(n, marks[n]) for n in STAGE_ORDER if n in marks]
    if len(present) < 2:
        return None
    t0 = present[0][1]
    labels, starts, widths = [], [], []
    for (n0, s0), (n1, s1) in zip(present, present[1:]):
        labels.append(f"{n0} -> {n1}")
        starts.append((s0 - t0) * 1000)
        widths.append((s1 - s0) * 1000)

    fig, ax = plt.subplots(figsize=(9, 0.6 * len(labels) + 1.2))
    ys = range(len(labels))
    ax.barh(list(ys), widths, left=starts)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time since speech start (ms)")
    ax.set_title("Voice-to-voice latency waterfall")
    for i, (s, w) in enumerate(zip(starts, widths)):
        ax.text(s + w + 2, i, f"{w:.0f} ms", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
