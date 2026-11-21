# -*- coding: utf-8 -*-
"""Model backends for the AVA Native runtime.

Two implementations of the `SynthesisModel` contract:

- `NativeN1Backend` — the original AVA-N1 model (product path, MIT weights).
- `LegacyPocketTTSBackend` — optional adapter for the legacy Pocket-TTS
  checkpoint (CC BY-NC 4.0 weights). NOT part of commercial bundles; used for
  comparison/evaluation only. Its voices carry commercial_ok=False.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ava.language import LanguageEngine
from ava.runtime.engine import SynthesisOptions, SynthesisResult, Voice

LEGACY_VOICES = ("nava", "shirin", "arash")


class NativeN1Backend:
    """The product backend: AVA-N1 bundle (weights + tokenizer + config)."""

    def __init__(self, bundle_dir: str):
        from training.train_native import load_native_bundle
        self.model, self.tok, self.cfg = load_native_bundle(bundle_dir)
        self.lang = LanguageEngine()
        self.bundle_dir = bundle_dir

    def _voices(self) -> list[Voice]:
        out = []
        for sid in range(max(self.cfg.n_speakers, 1)):
            name = "ava" if sid == 0 else f"ava-{sid + 1}"
            out.append(Voice(
                voice_id=f"ava-n1-{name}", display_name=f"آوا (بوم Native) {sid + 1}",
                language="fa", backend="ava-n1", ref=str(sid),
                description="صدای اختصاصی AVA-N1 (بعد از آموزش واقعی، با دادهٔ مجاز)",
                license="MIT (کد/وزن‌های مخزن)", commercial_ok=True,
            ))
        return out

    def list_voices(self) -> list[Voice]:
        return self._voices()

    def info(self) -> dict:
        return {"format": "ava-n1", "sample_rate": self.cfg.sample_rate,
                "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2),
                "bundle": self.bundle_dir}

    def synthesize(self, text: str, voice: Voice, options: SynthesisOptions) -> SynthesisResult:
        t0 = time.time()
        normalized = self.lang.normalize(text)[: self.cfg.max_text_tokens - 2]
        ids = torch.tensor([self.tok.encode(normalized)])
        mel = self.model.infer(ids, torch.tensor([ids.size(1)]), speed=options.speed,
                               speaker_id=torch.tensor([int(voice.ref or 0)]))
        from ava.audio.mel import griffin_lim_from_mel
        wav = griffin_lim_from_mel(mel.numpy(), self.cfg)
        sr = options.sample_rate or self.cfg.sample_rate
        elapsed = time.time() - t0
        return SynthesisResult(audio=wav.astype(np.float32), sample_rate=sr, voice=voice.voice_id,
                               text=text, duration_s=len(wav) / sr, elapsed_s=elapsed,
                               rtf=elapsed / max(len(wav) / sr, 1e-6))


class LegacyPocketTTSBackend:
    """Optional, non-commercial adapter for the legacy Pocket-TTS checkpoint."""

    def __init__(self, config_path: str, voices: tuple[str, ...] = LEGACY_VOICES, device: str = "cpu"):
        from ava.models.tts_model import TTSModel
        self.model = TTSModel.load_model(config_path)
        self.model.eval()
        self.voice_names = voices
        self.config_path = config_path
        if device == "cpu":
            self.model.cpu()

    def _voice_ref(self, name: str) -> str:
        """Resolve a bare voice name to its .safetensors path (legacy format)."""
        import os
        from pathlib import Path
        direct = Path(f"{name}.safetensors")
        if direct.exists():
            return str(direct)
        try:
            from ava.utils.config import resolve_path
            vd = resolve_path(self.model.config.voices_dir, self.model._base)
            cand = Path(vd) / f"{name}.safetensors"
            if cand.exists():
                return str(cand)
        except Exception:
            pass
        cand = Path(os.path.dirname(self.config_path)).parent / "weights" / "voices" / f"{name}.safetensors"
        return str(cand) if cand.exists() else name

    def list_voices(self) -> list[Voice]:
        return [Voice(
            voice_id=f"legacy-{name}", display_name=f"Legacy: {name}", language="fa",
            backend="legacy-pocket-tts", ref=name,
            description="صدای چک‌پوینت Legacy — فقط مقایسه/ارزیابی؛ غیرتجاری",
            license="CC BY-NC 4.0", commercial_ok=False,
        ) for name in self.voice_names]

    def info(self) -> dict:
        return {"format": "legacy-pocket-tts", "sample_rate": self.model.sample_rate,
                "license": "CC BY-NC 4.0 (weights)", "commercial": False}

    def synthesize(self, text: str, voice: Voice, options: SynthesisOptions) -> SynthesisResult:
        t0 = time.time()
        state = self.model.get_state_for_audio_prompt(self._voice_ref(str(voice.ref)))
        audio = self.model.generate_audio(state, text)
        wav = audio.detach().cpu().numpy().astype(np.float32)
        sr = self.model.sample_rate
        elapsed = time.time() - t0
        return SynthesisResult(audio=wav, sample_rate=sr, voice=voice.voice_id,
                               text=text, duration_s=len(wav) / sr, elapsed_s=elapsed,
                               rtf=elapsed / max(len(wav) / sr, 1e-6))
