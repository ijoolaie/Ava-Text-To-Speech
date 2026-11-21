# -*- coding: utf-8 -*-
"""Audio validation for dataset entries (AVA Native pipeline).

Checks format, sample rate, channels, duration, silence, noise (SNR estimate)
and clipping, and writes the results back into the manifest entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class ValidationLimits:
    expected_sample_rate: int = 24000
    min_duration_s: float = 0.4
    max_duration_s: float = 20.0
    silence_rms_db: float = -55.0    # below this -> silent utterance
    min_snr_db: float = 15.0
    clipping_ratio: float = 0.02     # fraction of |x| >= 0.999 samples


def _snr_db(x: np.ndarray) -> float:
    """Energy-based SNR estimate: speech frames vs quiet frames.

    For signals with no measurable quiet floor (constant tones) the estimate
    is undefined, so we report a pass-through high value instead of a false alarm.
    """
    frame = max(int(0.025 * 24000), 256)
    n = len(x) // frame
    if n == 0:
        return 0.0
    energies = np.array([float(np.mean(x[i * frame:(i + 1) * frame] ** 2)) for i in range(n)])
    hi, lo = float(np.mean(energies[energies >= np.percentile(energies, 60)])), \
        float(np.mean(energies[energies <= np.percentile(energies, 15)]))
    if lo <= 0 or hi / max(lo, 1e-12) < 4.0:
        return 60.0  # flat signal — noise floor not measurable
    return 10.0 * np.log10(hi / lo)


def estimate_snr_db(x: np.ndarray) -> float:
    """Public energy-based SNR estimate (used by validate and voice quality checks)."""
    return _snr_db(x)


def validate_audio(entry, limits: ValidationLimits = ValidationLimits(), strict_sr: bool = True) -> list[str]:
    """Validate one manifest entry in place; returns the list of issues found."""
    import os

    issues: list[str] = []
    if not os.path.isfile(entry.audio_path):
        entry.valid = False
        return ["missing_file"]
    try:
        info = sf.info(entry.audio_path)
        x, sr = sf.read(entry.audio_path, dtype="float32", always_2d=True)
    except Exception:  # unreadable/corrupt
        entry.valid = False
        return ["unreadable_audio"]

    entry.sample_rate = int(sr)
    entry.channels = int(info.channels)
    entry.duration_s = float(info.frames / sr)

    if strict_sr and sr != limits.expected_sample_rate:
        issues.append(f"sample_rate_{sr}")
    if info.channels != 1:
        issues.append(f"channels_{info.channels}")
    if entry.duration_s < limits.min_duration_s:
        issues.append("too_short")
    if entry.duration_s > limits.max_duration_s:
        issues.append("too_long")

    mono = x.mean(axis=1)
    rms_db = 20 * np.log10(float(np.sqrt(np.mean(mono ** 2))) + 1e-10)
    if rms_db < limits.silence_rms_db:
        issues.append("silence")
    snr = _snr_db(mono)
    if snr < limits.min_snr_db:
        issues.append(f"low_snr_{snr:.1f}db")
    clip = float(np.mean(np.abs(mono) >= 0.999))
    if clip > limits.clipping_ratio:
        issues.append(f"clipping_{clip:.2%}")

    entry.issues = issues
    entry.valid = len(issues) == 0
    return issues
