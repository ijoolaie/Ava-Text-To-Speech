# -*- coding: utf-8 -*-
"""Voice prompt quality measurement (AVA Native voice system).

Checks a prompt/reference audio like a dataset validator would: duration,
level, SNR estimate, clipping — and emits a pass/fail recommendation so bad
prompts never become a shipped voice.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf

from training.datasets.validate import estimate_snr_db


def check_voice_prompt(path: str, expected_sr: int = 24000,
                       min_s: float = 3.0, max_s: float = 12.0,
                       min_snr_db: float = 20.0) -> dict:
    """Measure one prompt WAV; returns metrics + ok flag + reasons."""
    info = sf.info(path)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    duration = info.frames / sr
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-10)
    snr = estimate_snr_db(mono)
    clip = float(np.mean(np.abs(mono) >= 0.999))
    reasons = []
    if sr != expected_sr:
        reasons.append(f"sample_rate_{sr}!=_{expected_sr}")
    if not (min_s <= duration <= max_s):
        reasons.append(f"duration_{duration:.1f}s_out_of_range")
    if rms_db < -40:
        reasons.append("too_quiet")
    if snr < min_snr_db:
        reasons.append(f"low_snr_{snr:.1f}db")
    if clip > 0.02:
        reasons.append(f"clipping_{clip:.2%}")
    return {
        "file": path, "sample_rate": int(sr), "channels": int(info.channels),
        "duration_s": round(duration, 2), "rms_db": round(rms_db, 1),
        "snr_db": round(snr, 1), "clipping": round(clip, 4),
        "ok": not reasons, "reasons": reasons,
    }
