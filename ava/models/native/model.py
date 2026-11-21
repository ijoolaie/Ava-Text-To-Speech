# -*- coding: utf-8 -*-
"""AVA-N1: compact non-autoregressive Persian TTS (original AVA Native design).

Pipeline: text tokens -> Transformer encoder -> duration predictor ->
length regulator -> Transformer decoder -> mel spectrogram.
Training loss: mel L1 + duration MSE (log domain). Inference is fully
deterministic (no sampling), which makes ONNX export and CPU/Android
inference straightforward.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import AVANativeConfig


class TransformerBlock(nn.Module):
    """Pre-norm self-attention block (original implementation)."""

    def __init__(self, d_model: int, n_heads: int, ffn: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, d_model))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg: AVANativeConfig, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_text_tokens, cfg.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout) for _ in range(cfg.n_layers_enc))
        self.speaker = nn.Embedding(max(cfg.n_speakers, 1), cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, speaker_id: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, T) padded; lengths: (B,); returns (B, T, D)."""
        T = tokens.size(1)
        x = self.embed(tokens) + self.pos.weight[:T].unsqueeze(0)
        if speaker_id is None:
            speaker_id = torch.zeros(tokens.size(0), dtype=torch.long, device=tokens.device)
        x = x + self.speaker(speaker_id).unsqueeze(1)
        pad_mask = torch.arange(T, device=tokens.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        return self.out_norm(x)


class DurationPredictor(nn.Module):
    def __init__(self, cfg: AVANativeConfig):
        super().__init__()
        d, k = cfg.dur_hidden, cfg.dur_kernel
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.d_model, d, k, padding=k // 2), nn.ReLU(), nn.GroupNorm(1, d),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(d, d, k, padding=k // 2), nn.ReLU(), nn.GroupNorm(1, d),
        )
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> log-durations (B, T)."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.head(h).squeeze(-1)  # log-domain durations (can be negative; regulator floors at 1 frame)


def length_regulator(x: torch.Tensor, durations: torch.Tensor, max_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat each frame per its duration. durations: real-valued (rounded here).

    Returns (mel-length tensor (B, F, D), frame lengths (B,)).
    """
    B = x.size(0)
    reps = torch.round(durations).clamp(min=1).long()
    frames = []
    frame_lens = torch.zeros(B, dtype=torch.long, device=x.device)
    for b in range(B):
        seq = torch.repeat_interleave(x[b], reps[b], dim=0)[:max_frames]
        frames.append(seq)
        frame_lens[b] = seq.size(0)
    F = max(f.size(0) for f in frames)
    out = x.new_zeros(B, F, x.size(2))
    for b in range(B):
        out[b, : frames[b].size(0)] = frames[b]
    return out, frame_lens


class Decoder(nn.Module):
    def __init__(self, cfg: AVANativeConfig):
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout) for _ in range(cfg.n_layers_dec))
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.n_mels)
        # PostNet: 5 conv layers with tanh gates for mel refinement
        ch = [cfg.n_mels, 512, 512, 512, 512, cfg.n_mels]
        self.post = nn.ModuleList(
            nn.Conv1d(ch[i], ch[i + 1], 5, padding=2) for i in range(5)
        )
        self.post_bn = nn.ModuleList(nn.BatchNorm1d(ch[i + 1]) for i in range(4))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.blocks:
            x = blk(x)
        mel = self.proj(self.out_norm(x)).transpose(1, 2)  # (B, n_mels, F)
        h = mel
        for i, conv in enumerate(self.post):
            h = conv(h)
            if i < 4:
                h = torch.tanh(self.post_bn[i](h))
        return mel, mel + h


class AVANativeTTS(nn.Module):
    def __init__(self, cfg: AVANativeConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = TextEncoder(cfg, vocab_size)
        self.duration = DurationPredictor(cfg)
        self.decoder = Decoder(cfg)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor,
        log_durations: torch.Tensor,
        speaker_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training step. mel: (B, n_mels, F) padded; log_durations: (B, T) target."""
        enc = self.encoder(tokens, lengths, speaker_id)
        pred_log_dur = self.duration(enc)
        # During training the decoder consumes the TARGET durations (FastSpeech-style),
        # expanded from log domain to linear frames; predictions feed the MSE only.
        reg, _ = length_regulator(enc, torch.exp(log_durations), self.cfg.max_frames)
        F_target = mel.size(2)
        if reg.size(1) > F_target:
            reg = reg[:, :F_target]
        elif reg.size(1) < F_target:
            reg = torch.cat([reg, reg.new_zeros(reg.size(0), F_target - reg.size(1), reg.size(2))], dim=1)
        mel_pre, mel_post = self.decoder(reg)
        mask = torch.arange(mel.size(2), device=mel.device).unsqueeze(0) < mel_lengths.unsqueeze(1)
        mask_f = mask.float().unsqueeze(1)
        l1 = (torch.abs(mel_post - mel) * mask_f).sum() / (mask_f.sum() * mel.size(1) + 1e-8)
        l1_pre = (torch.abs(mel_pre - mel) * mask_f).sum() / (mask_f.sum() * mel.size(1) + 1e-8)
        dur_mask = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        dur_loss = (((pred_log_dur - log_durations) ** 2) * dur_mask.float()).sum() / (dur_mask.float().sum() + 1e-8)
        return {"loss": l1 + l1_pre + dur_loss, "mel_l1": l1 + l1_pre, "dur_loss": dur_loss}

    @torch.no_grad()
    def infer(self, tokens: torch.Tensor, lengths: torch.Tensor, speed: float = 1.0, speaker_id: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (1, T) -> mel (n_mels, F). Durations predicted in log domain,
        expanded to linear frames (speed scales the log-duration)."""
        enc = self.encoder(tokens, lengths, speaker_id)
        log_dur = self.duration(enc) - math.log(max(speed, 0.05))
        reg, _ = length_regulator(enc, torch.exp(log_dur), self.cfg.max_frames)
        _, mel_post = self.decoder(reg)
        return mel_post.squeeze(0).clamp(-1.0, 1.0)
