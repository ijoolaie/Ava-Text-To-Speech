# -*- coding: utf-8 -*-
"""Independent vocoder boundary of the AVA Native product path.

The v1 vocoder is Griffin-Lim: purely algorithmic, no external code and no
pretrained weights. It validates the mel→audio path end-to-end but is
explicitly TEMPORARY and NOT the final quality target (documented in
docs/NEURAL_ARCHITECTURE.md and docs/ROADMAP.md phase 3). A dedicated AVA
neural vocoder will be trained later and plugged in behind this same
interface, without touching the model or runtime contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ava.audio.mel import griffin_lim_from_mel
from ava.models.native import AVANativeConfig


class Vocoder(ABC):
    """Contract: mel [n_mels, T] float (model log-mel output) → waveform float32
    at cfg.sample_rate. Deterministic given the same mel and options."""

    name: str = "abstract"

    @abstractmethod
    def mel_to_audio(self, mel: np.ndarray, cfg: AVANativeConfig) -> np.ndarray:
        """Convert a model mel spectrogram to a mono waveform."""


class GriffinLimVocoder(Vocoder):
    """v1 — temporary, algorithmic (no weights, no third-party code)."""

    name = "griffin-lim-v1"

    def mel_to_audio(self, mel: np.ndarray, cfg: AVANativeConfig) -> np.ndarray:
        return griffin_lim_from_mel(mel, cfg).astype(np.float32)


_VOCODERS: dict[str, type[Vocoder]] = {GriffinLimVocoder.name: GriffinLimVocoder}


def get_vocoder(name: str = GriffinLimVocoder.name) -> Vocoder:
    """Resolve a vocoder by name; future neural vocoders register here."""
    try:
        return _VOCODERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown vocoder {name!r}; available: {sorted(_VOCODERS)}"
        ) from None
