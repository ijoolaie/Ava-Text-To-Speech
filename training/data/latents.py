"""Encoding audio to frozen-Mimi latents (the training targets)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ava.data.audio import audio_read  # noqa: E402
from ava.data.audio_utils import convert_audio  # noqa: E402


class LatentEncoder:
    """Wraps the frozen Mimi encoder; produces raw 12.5 Hz latents [T, 32].

    Only the codec is built and loaded (from the config's bundle) — the flow LM
    is not needed for data preparation, so any architecture config works as long
    as its `mimi` section and `weights_path` match the released checkpoint.
    """

    def __init__(self, config_path: str | Path, device: str = "cpu"):
        from ava.models.mimi import build_mimi
        from ava.utils.config import load_config
        from ava.utils.utils import resolve_path
        from ava.utils.weights_io import load_file

        arch = load_config(config_path)
        self.mimi = build_mimi(arch.mimi)
        base = load_file(resolve_path(arch.weights_path, REPO_ROOT))
        self.mimi.load_state_dict(
            {k.removeprefix("mimi."): v for k, v in base.items() if k.startswith("mimi.")},
            strict=True,
        )
        self.mimi.eval().to(device)
        self.device = device
        self.sample_rate = arch.mimi.sample_rate
        self.frame_rate = arch.mimi.frame_rate

    @torch.no_grad
    def encode_file(self, path: str | Path) -> np.ndarray:
        wav, sr = audio_read(path)
        wav = convert_audio(wav, sr, self.sample_rate, 1)
        return self.encode_waveform(wav, self.sample_rate)

    @torch.no_grad
    def encode_waveform(self, wav: torch.Tensor, sample_rate: int) -> np.ndarray:
        wav = convert_audio(wav, sample_rate, self.sample_rate, 1)
        latents = self.mimi.encode_to_latent(wav.unsqueeze(0).to(self.device))
        return latents[0].to(torch.float32).cpu().numpy()
