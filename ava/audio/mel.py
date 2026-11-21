# -*- coding: utf-8 -*-
"""Mel-spectrogram conversion for AVA-N1 (24 kHz / 80 mels / 12.5 ms hop).

Torch path for training, numpy path for inference/ONNX pipelines, plus a
Griffin-Lim vocoder (v1) for verification-grade audio synthesis.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ava.models.native.config import AVANativeConfig


def mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    """Slaney-style triangular filterbank (original implementation)."""

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    bins_hz = mel_to_hz(mels)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = bins_hz[i], bins_hz[i + 1], bins_hz[i + 2]
        up = (freqs - lo) / max(center - lo, 1e-9)
        down = (hi - freqs) / max(hi - center, 1e-9)
        fb[i] = np.clip(np.minimum(up, down), 0.0, 1.0)
    return fb


def mel_spectrogram(wav: torch.Tensor, cfg: AVANativeConfig) -> torch.Tensor:
    """wav: (N,) or (1, N) float in [-1, 1] -> log-mel (n_mels, F) normalized to [-4, 4]."""
    if wav.dim() == 2:
        wav = wav.squeeze(0)
    spec = torch.stft(wav, cfg.n_fft, cfg.hop_size, cfg.win_size, torch.hann_window(cfg.win_size),
                      center=True, pad_mode="reflect", return_complex=True)
    mag = spec.abs() ** 2
    fb = torch.from_numpy(mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax)).to(wav.device)
    mel = torch.matmul(fb, mag)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = (mel - cfg.mel_min) / (cfg.mel_max - cfg.mel_min) * 2.0 - 1.0
    return mel.clamp(-1.0, 1.0)  # normalized range


def mel_to_numpy(mel: torch.Tensor, cfg: AVANativeConfig) -> np.ndarray:
    m = mel.detach().cpu().numpy()
    return (m + 1.0) / 2.0 * (cfg.mel_max - cfg.mel_min) + cfg.mel_min


def griffin_lim_from_mel(mel_norm: np.ndarray, cfg: AVANativeConfig, n_iter: int = 32) -> np.ndarray:
    """Reconstruct audio from normalized log-mel via Griffin-Lim (v1 verification vocoder).

    mel_norm: (n_mels, F) in [-1, 1] -> PCM float32 in [-1, 1].
    """
    import librosa

    log_mel = (mel_norm + 1.0) / 2.0 * (cfg.mel_max - cfg.mel_min) + cfg.mel_min
    power = np.exp(log_mel)
    fb = mel_filterbank(cfg.n_fft, cfg.n_mels, cfg.sample_rate, cfg.fmin, cfg.fmax)
    fb_pinv = np.linalg.pinv(fb)  # (n_freq, n_mels)
    mag = np.sqrt(np.maximum(fb_pinv @ power, 0.0))
    wav = librosa.griffinlim(mag, n_iter=n_iter, hop_length=cfg.hop_size,
                             win_length=cfg.win_size, n_fft=cfg.n_fft, window="hann")
    peak = np.max(np.abs(wav)) + 1e-8
    return (wav / max(peak, 1.0)).astype(np.float32)
