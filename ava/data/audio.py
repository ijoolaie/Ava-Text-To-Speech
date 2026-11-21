# Adapted from Kyutai pocket-tts v4.0.1 (MIT). WAV read via the stdlib wave module;
# other formats need the optional soundfile extra.
"""Audio IO (read WAV/other formats) for prompts and dataset preparation."""

import logging
import wave
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def audio_read(filepath: str | Path) -> tuple[torch.Tensor, int]:
    """Read an audio file as (mono float32 tensor [1, T], sample_rate)."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".wav":
        # Use built-in wave module for WAV files
        with wave.open(str(filepath), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            if sample_width != 2:
                return _audio_read_with_soundfile(filepath)
            raw_data = wav_file.readframes(-1)
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
            return torch.from_numpy(samples.copy()).unsqueeze(0), sample_rate

    return _audio_read_with_soundfile(filepath)


def _audio_read_with_soundfile(filepath: Path) -> tuple[torch.Tensor, int]:
    # For non-WAV and non-16-bit WAV formats, use soundfile (optional dependency)
    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError(
            "soundfile is required to read non-WAV or non-16-bit WAV audio files. "
            "Install with: `pip install soundfile`"
        ) from e

    data, sample_rate = sf.read(str(filepath), dtype="float32")
    if data.ndim == 1:
        wav = torch.from_numpy(data).unsqueeze(0)
    else:
        wav = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
    return wav, sample_rate
