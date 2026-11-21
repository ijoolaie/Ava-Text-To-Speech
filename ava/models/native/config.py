# -*- coding: utf-8 -*-
"""AVA-N1 native model configuration (AVA Native original architecture)."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AVANativeConfig:
    # tokenizer
    max_text_tokens: int = 512
    # encoder / decoder
    d_model: int = 256
    n_heads: int = 4
    n_layers_enc: int = 4
    n_layers_dec: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_frames: int = 2048
    # duration predictor
    dur_hidden: int = 256
    dur_kernel: int = 3
    # mel spectrogram (product constants — see docs/NEURAL_ARCHITECTURE.md)
    sample_rate: int = 24000
    n_fft: int = 2048
    win_size: int = 1200
    hop_size: int = 300
    n_mels: int = 80
    fmin: float = 80.0
    fmax: float = 12000.0
    mel_min: float = -4.0
    mel_max: float = 4.0
    # speaker conditioning
    n_speakers: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AVANativeConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})
