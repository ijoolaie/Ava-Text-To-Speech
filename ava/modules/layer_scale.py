# Vendored from Kyutai pocket-tts v4.0.1 (https://github.com/kyutai-labs/pocket-tts), MIT License.
# Copyright (c) 2025 Kyutai. Adapted for the Ava project; see NOTICE.md for attribution.
import torch
import torch.nn as nn


class LayerScale(nn.Module):
    def __init__(self, channels: int, init: float):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x
