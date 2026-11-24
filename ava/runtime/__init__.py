# -*- coding: utf-8 -*-
"""ava.runtime — pluggable synthesis runtime of AVA Native."""
from .backends import LegacyPocketTTSBackend, NativeN1Backend
from .alifathi import AliFathiTTSBackend
from .engine import (
    BackendRegistry, SynthesisCancelled, SynthesisEngine, SynthesisModel,
    SynthesisOptions, SynthesisResult, Voice, wav_bytes,
)

__all__ = [
    "SynthesisEngine", "SynthesisModel", "SynthesisOptions", "SynthesisResult",
    "SynthesisCancelled", "Voice", "BackendRegistry", "wav_bytes",
    "NativeN1Backend", "LegacyPocketTTSBackend", "AliFathiTTSBackend",
]
