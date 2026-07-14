"""Word error rate across noise levels.

Put a handful of clips in eval/data/ as <name>.wav with a matching <name>.txt
reference transcript (a few LibriSpeech clips or your own recordings). This
transcribes each clip clean and at SNR 20/10/5/0 dB, reports WER, and plots the
curve.

    python -m eval.wer_test
"""

import glob
import os

import jiwer
import numpy as np

from agent.stt import STT
from eval.noise import add_noise

_NORM = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _load_wav_16k(path):
    import soundfile as sf

    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16_000:  # cheap linear resample; fine for an eval harness
        n = int(len(audio) * 16_000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio), n, endpoint=False),
            np.arange(len(audio)),
            audio,
        )
    return audio.astype("float32")


def _wer(refs, hyps):
    return jiwer.wer(refs, hyps, truth_transform=_NORM, hypothesis_transform=_NORM)


def run(clip_dir="eval/data", snrs=(20, 10, 5, 0)):
    stt = STT()
    clips = []
    for wav in sorted(glob.glob(os.path.join(clip_dir, "*.wav"))):
        txt = os.path.splitext(wav)[0] + ".txt"
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                clips.append((_load_wav_16k(wav), f.read().strip()))

    if not clips:
        print(f"No <name>.wav / <name>.txt pairs found in {clip_dir}.")
        return

    refs = [ref for _, ref in clips]
    levels = ["clean"] + [f"{s} dB" for s in snrs]
    results = {}

    results["clean"] = _wer(refs, [stt.transcribe(a) for a, _ in clips])
    for s in snrs:
        hyps = [stt.transcribe(add_noise(a, s)) for a, _ in clips]
        results[f"{s} dB"] = _wer(refs, hyps)

    print("\nWER by condition")
    for level in levels:
        print(f"  {level:<8} {results[level] * 100:5.1f} %")

    _plot(levels, [results[l] * 100 for l in levels])


def _plot(levels, wers, path="wer_vs_snr.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(levels)), wers, marker="o")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_ylabel("WER (%)")
    ax.set_xlabel("Condition (decreasing SNR ->)")
    ax.set_title("Whisper WER vs noise level")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"\n[saved plot -> {path}]")


if __name__ == "__main__":
    run()
