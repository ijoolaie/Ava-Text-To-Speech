# -*- coding: utf-8 -*-
"""Energy-based segmentation of long recordings into utterances (VAD-lite).

Splits audio on silence gaps so a single studio recording becomes utterance
entries. Deterministic: fixed energy thresholds, no learned VAD dependency.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf


def segment_audio(
    path: str,
    sr: int = 24000,
    silence_rms_db: float = -45.0,
    min_speech_s: float = 0.3,
    min_silence_s: float = 0.35,
    max_segment_s: float = 15.0,
) -> list[tuple[float, float]]:
    """Return [(start_s, end_s)] speech segments of one recording."""
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    if file_sr != sr:
        import librosa
        x = librosa.resample(x.mean(axis=1), orig_sr=file_sr, target_sr=sr)
    else:
        x = x.mean(axis=1)
    frame = int(0.025 * sr)
    n = len(x) // frame
    if n == 0:
        return []
    rms_db = np.array([20 * np.log10(float(np.sqrt(np.mean(x[i * frame:(i + 1) * frame] ** 2))) + 1e-10) for i in range(n)])
    speech = rms_db > silence_rms_db
    segments: list[tuple[float, float]] = []
    start = None
    last_speech = -1
    min_sil = int(min_silence_s * sr / frame)
    min_spch = int(min_speech_s * sr / frame)
    max_seg = int(max_segment_s * sr / frame)
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            last_speech = i
        elif start is not None and i - last_speech >= min_sil:
            if last_speech - start >= min_spch:
                segments.append((start * frame / sr, (last_speech + 1) * frame / sr))
            start = None
    if start is not None and last_speech - start >= min_spch:
        segments.append((start * frame / sr, (last_speech + 1) * frame / sr))
    # split overlong segments on the lowest-energy frame inside them
    final: list[tuple[float, float]] = []
    for a, b in segments:
        a_f, b_f = int(a * sr / frame), int(b * sr / frame)
        while b_f - a_f > max_seg:
            window = rms_db[a_f + min_spch : b_f - min_spch]
            cut = a_f + min_spch + int(np.argmin(window))
            final.append((a_f * frame / sr, cut * frame / sr))
            a_f = cut
        final.append((a_f * frame / sr, b_f * frame / sr))
    return final
