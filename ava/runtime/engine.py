# -*- coding: utf-8 -*-
"""AVA Native runtime — a pluggable, architecture-independent synthesis engine.

The engine speaks to models through the `SynthesisModel` protocol; concrete
backends register by format name. This is the stable API layer used by the
CLI, the HTTP API, the web UI and (via ONNX) the Android/Windows runtimes.
"""
from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np
import soundfile as sf

from ava.language import LanguageEngine, LanguageConfig, estimate_duration, split_sentences


class SynthesisCancelled(Exception):
    """Raised when engine.stop() was requested at a sentence boundary."""




@dataclass
class Voice:
    """A voice profile: identity + routing to the backend that owns it."""

    voice_id: str
    display_name: str
    language: str = "fa"
    gender: str = ""
    backend: str = ""          # model format that owns this voice ("ava-n1", "legacy-pocket-tts")
    ref: str = ""              # backend-specific reference (file path, speaker id, ...)
    description: str = ""
    license: str = ""
    commercial_ok: bool = False


@dataclass
class SynthesisOptions:
    speed: float = 1.0
    sample_rate: int | None = None   # None = model native rate
    speaker_id: int = 0
    pause_scale: float = 1.0         # punctuation pause scaling (native model v2)
    stop_event: "threading.Event | None" = None  # per-request cancellation


@dataclass
class SynthesisResult:
    audio: np.ndarray                # float32 mono
    sample_rate: int
    voice: str
    text: str
    duration_s: float
    elapsed_s: float
    rtf: float                       # real-time factor: synthesis time / audio time


class SynthesisModel(Protocol):
    """Contract every backend implements (AC5.3: architecture-independent)."""

    def synthesize(self, text: str, voice: Voice, options: SynthesisOptions) -> SynthesisResult: ...
    def list_voices(self) -> list[Voice]: ...
    def info(self) -> dict: ...


class BackendRegistry:
    """Model-format registry — nothing is hard-wired to one architecture."""

    def __init__(self):
        self._factories: dict[str, callable] = {}

    def register(self, fmt: str, factory: callable) -> None:
        self._factories[fmt] = factory

    def create(self, fmt: str, **kwargs) -> SynthesisModel:
        if fmt not in self._factories:
            raise KeyError(f"unknown model format: {fmt!r} (registered: {sorted(self._factories)})")
        return self._factories[fmt](**kwargs)


class SynthesisEngine:
    """High-level engine: load/unload models, voices, batch and streaming synthesis."""

    def __init__(self, language_config: LanguageConfig | None = None):
        self.engine_lang = LanguageEngine(language_config) if language_config else LanguageEngine()
        self.registry = BackendRegistry()
        self._models: dict[str, SynthesisModel] = {}
        self._voices: dict[str, Voice] = {}
        self._voice_owner: dict[str, str] = {}  # voice_id -> model_id
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---- model lifecycle ----------------------------------------------------
    def register_backend(self, fmt: str, factory: callable) -> None:
        self.registry.register(fmt, factory)

    def load_model(self, model_id: str, fmt: str, **kwargs) -> SynthesisModel:
        """Load and register a model; also imports its voices."""
        with self._lock:
            if model_id in self._models:
                return self._models[model_id]
            model = self.registry.create(fmt, **kwargs)
            self._models[model_id] = model
            for v in model.list_voices():
                self._voices[v.voice_id] = v
                self._voice_owner[v.voice_id] = model_id
            return model

    def unload_model(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self._models:
                raise KeyError(model_id)
            del self._models[model_id]
            self._voices = {k: v for k, v in self._voices.items() if self._voice_owner.get(k) != model_id}
            self._voice_owner = {k: m for k, m in self._voice_owner.items() if m != model_id}
        import gc
        gc.collect()

    def get_model_info(self) -> list[dict]:
        return [{"model_id": mid, **m.info()} for mid, m in self._models.items()]

    # ---- voices -------------------------------------------------------------
    def list_voices(self) -> list[Voice]:
        return sorted(self._voices.values(), key=lambda v: v.voice_id)

    def get_voice(self, voice_id: str) -> Voice:
        if voice_id not in self._voices:
            raise KeyError(f"unknown voice: {voice_id!r} (available: {sorted(self._voices)})")
        return self._voices[voice_id]

    # ---- synthesis ----------------------------------------------------------
    def _synthesize_sentence(self, model: SynthesisModel, sentence: str, voice: Voice, options: SynthesisOptions, event=None) -> SynthesisResult:
        if (event or options.stop_event or self._stop).is_set():
            raise SynthesisCancelled("stopped by caller")
        return model.synthesize(sentence, voice, options)

    def synthesize(self, text: str, voice_id: str, options: SynthesisOptions | None = None) -> SynthesisResult:
        options = options or SynthesisOptions()
        voice = self.get_voice(voice_id)
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        sentences = split_sentences(text)
        model = self._model_for(voice)
        event = options.stop_event or self._stop
        results = [self._synthesize_sentence(model, s, voice, options, event) for s in sentences]
        return _concat(results, voice, text)

    def synthesize_stream(self, text: str, voice_id: str, options: SynthesisOptions | None = None) -> Iterator[SynthesisResult]:
        """Yield one SynthesisResult per sentence — the caller can start playing early."""
        options = options or SynthesisOptions()
        voice = self.get_voice(voice_id)
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")
        model = self._model_for(voice)
        event = options.stop_event or self._stop
        for sentence in split_sentences(text):
            if event.is_set():
                raise SynthesisCancelled("stopped by caller")
            yield self._synthesize_sentence(model, sentence, voice, options)

    def synthesize_batch(self, texts: list[str], voice_id: str, options: SynthesisOptions | None = None) -> list[SynthesisResult]:
        return [self.synthesize(t, voice_id, options) for t in texts]

    def stop(self) -> None:
        """Request cancellation; the next sentence boundary raises a cancellation error.

        The flag stays armed until `reset_stop()` — a pending stop also cancels
        the next call (the user asked to stop). Per-request events can be passed
        via `SynthesisOptions.stop_event` instead (used by the HTTP API).
        """
        self._stop.set()

    def reset_stop(self) -> None:
        self._stop.clear()

    def _model_for(self, voice: Voice) -> SynthesisModel:
        model_id = self._voice_owner.get(voice.voice_id)
        if model_id is None or model_id not in self._models:
            raise RuntimeError(f"voice {voice.voice_id!r} has no loaded owner model")
        return self._models[model_id]


def _concat(results: list[SynthesisResult], voice: Voice, text: str) -> SynthesisResult:
    sr = results[0].sample_rate
    gap = np.zeros(int(0.12 * sr), dtype=np.float32)  # inter-sentence pause
    parts: list[np.ndarray] = []
    for i, r in enumerate(results):
        parts.append(r.audio)
        if i < len(results) - 1:
            parts.append(gap)
    audio = np.concatenate(parts)
    audio_s = len(audio) / sr
    elapsed = sum(r.elapsed_s for r in results)
    return SynthesisResult(audio=audio.astype(np.float32), sample_rate=sr, voice=voice.voice_id,
                           text=text, duration_s=audio_s, elapsed_s=elapsed, rtf=elapsed / max(audio_s, 1e-6))


def wav_bytes(result: SynthesisResult) -> bytes:
    """Encode a SynthesisResult as 16-bit PCM WAV bytes (HTTP-friendly)."""
    buf = io.BytesIO()
    sf.write(buf, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()
