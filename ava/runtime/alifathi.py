# -*- coding: utf-8 -*-
"""Ali Fathi Persian TTS backend — pretrained ONNX, Piper ONNX convention.

A third implementation of the `SynthesisModel` contract: the published
pretrained `alifthi/persian-TTS` VITS model (https://huggingface.co/alifthi/persian-TTS).
Weights are CC-BY-4.0 (commercial use permitted with attribution — see NOTICE.md;
LibriTTS-R and FLEURS attribution requirements per the official model card).

Fixed integration decisions:
- inference ONLY — no training/fine-tuning, no new weights, no datasets;
- the model directory is a configurable LOCAL path: constructor argument,
  the ``AVA_ALIFATHI_MODEL_DIR`` environment variable, or by default the
  repository-local ``alifathi/`` directory (companion files must be placed
  there once, offline); NO network access and NO download at runtime;
- ONNX Runtime CPU inference (``onnxruntime`` is an OPTIONAL dependency,
  declared as the ``alifathi`` extra: ``pip install ".[alifathi]"``);
- the text→phoneme frontend is eSpeak NG (`fa` IPA phonemes + the model's own
  Piper phoneme table). eSpeak NG is an OPTIONAL EXTERNAL GPL-3.0-or-later
  component that AVA never bundles; when missing the backend fails closed with
  a clear installation hint.
- the existing `NativeN1` backend, engine API, CLI and frontend stay untouched.

Frontend provenance (license-safe): the glue below is original clean-room code
of this repository. It implements the I/O convention published in the model
card and the model's own ``model.onnx.json`` (phoneme_id_map / speaker_id_map /
inference scales) — verified byte-for-byte against the official calibration
inputs (``tf/calib/*.npy``, staged in ``alifathi/``). No code was copied from
the model's training repository (which ships without a license file).

Punctuation (verified against the model's phoneme_id_map): the table contains
only ASCII punctuation (``! " # $ ' ( ) , - . : ; ?``). ASCII marks are kept in
the phoneme stream (they carry prosody/pauses); Persian marks (``؟ ، ؛ « » …``)
have no model symbols — they act as clause boundaries for the phonemizer and
are dropped from the ID stream (documented + tested behaviour).

The two table members ``#`` and ``$`` need special handling:
- ``#`` is a normal table symbol and is kept/mapped like any other ASCII mark;
- ``$`` is the Piper EOS control symbol. In free text espeak-ng expands it to
  the word "dollar", but a punctuation run (e.g. ``قیمت$``) keeps a literal
  ``$`` in the phoneme string; :meth:`encode` therefore never injects a
  mid-stream EOS — the terminator is appended exactly once, at the end.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from ava.runtime.engine import SynthesisOptions, SynthesisResult, Voice

MODEL_HF_URL = "https://huggingface.co/alifthi/persian-TTS"
MODEL_DIR_ENV = "AVA_ALIFATHI_MODEL_DIR"
BACKEND_FORMAT = "alifathi-onnx"
VOICE_PREFIX = "alifathi"
REPO_ROOT = Path(__file__).resolve().parents[2]

# Piper phoneme-table control symbols (verified against the model config)
_PAD, _BOS, _EOS = "_", "^", "$"
# punctuation chars present in the model's phoneme_id_map → kept in the stream.
# '#' maps like any other mark; '$' is the EOS control symbol — kept in the
# phoneme STRING but never injected mid-stream (see EspeakNGPhonemizer.encode).
_KEEP_PUNCT = set("!\'(),-.:;?\"#$")
# additional clause boundaries: Persian/script marks with NO model symbols
_SPLIT_PUNCT = _KEEP_PUNCT | set("؟،؛«»…")

# validated option ranges (invalid input must fail closed, not make broken audio)
_MIN_SPEED, _MAX_SPEED = 0.05, 10.0
_MIN_SAMPLE_RATE, _MAX_SAMPLE_RATE = 8000, 192000


def validate_speed(speed: float) -> float:
    """Speed must be a finite number in (0.05, 10] — reject garbage early.

    Booleans are rejected explicitly (``True`` would silently coerce to 1.0).
    """
    if isinstance(speed, bool):
        raise ValueError(f"speed must be a number, not a bool — got {speed!r}")
    try:
        s = float(speed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"speed must be a number, got {speed!r}") from exc
    if not math.isfinite(s):
        raise ValueError(f"speed must be finite, got {speed!r}")
    if not (_MIN_SPEED < s <= _MAX_SPEED):
        raise ValueError(
            f"speed must be in ({_MIN_SPEED}, {_MAX_SPEED}] — got {s}. "
            "Invalid speed would silently produce broken audio."
        )
    return s


def validate_sample_rate(sample_rate) -> int | None:
    """None → native rate; otherwise a whole number of Hz in [8000, 192000].

    Strict typing (fail closed instead of silent coercion): booleans, strings
    and fractional values (``16000.5``) are rejected; ``NaN``/``inf`` fail the
    whole-number check. An integral float (``16000.0``) is accepted and
    converted to ``int``.
    """
    if sample_rate is None:
        return None
    if isinstance(sample_rate, bool) or isinstance(sample_rate, str):
        raise ValueError(f"sample_rate must be an integer or None, got {sample_rate!r}")
    try:
        sr_float = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sample_rate must be an integer or None, got {sample_rate!r}") from exc
    if not sr_float.is_integer():
        raise ValueError(
            f"sample_rate must be a whole number of Hz (int) or None — got {sample_rate!r}"
        )
    sr = int(sr_float)
    if not (_MIN_SAMPLE_RATE <= sr <= _MAX_SAMPLE_RATE):
        raise ValueError(
            f"sample_rate must be in [{_MIN_SAMPLE_RATE}, {_MAX_SAMPLE_RATE}] Hz or None — "
            f"got {sr}. Invalid rates would silently produce broken audio."
        )
    return sr


def default_model_dir() -> Path:
    """Local model directory candidates, in priority order (all offline).

    1. ``AVA_ALIFATHI_MODEL_DIR`` env var (explicit override)
    2. ``<repo>/alifathi/``            — the designated local model directory
    3. ``<repo>/models/persian-tts/``  — legacy download location from the
                                          model-review step
    """
    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        return Path(env).expanduser()
    for cand in (REPO_ROOT / "alifathi", REPO_ROOT / "models" / "persian-tts"):
        if cand.is_dir():
            return cand
    return REPO_ROOT / "alifathi"


class PhonemizerUnavailable(RuntimeError):
    """eSpeak NG (optional external component) is missing or unusable."""


class ModelUnavailable(RuntimeError):
    """The local ONNX model / config is missing or unloadable."""


class EspeakNGPhonemizer:
    """Clean-room text→phoneme frontend: eSpeak NG CLI → IPA → Piper IDs.

    The espeak-ng binary runs as an external process (`-v fa --ipa -q`); the
    raw-text punctuation characters that exist in the model's phoneme table
    are appended after their clause (Piper convention). Persian punctuation
    (no model symbols) splits clauses and is dropped from the stream.
    Unmapped characters (ZWNJ, …) are skipped by :meth:`encode`.
    """

    def __init__(self, voice: str = "fa"):
        self.voice = voice

    def _ipa(self, text: str) -> str:
        try:
            proc = subprocess.run(
                ["espeak-ng", "-v", self.voice, "--ipa", "-q", text],
                capture_output=True, text=True, timeout=30, check=True,
            )
        except FileNotFoundError as exc:
            raise PhonemizerUnavailable(
                "eSpeak NG is required for the Ali Fathi phoneme frontend but was "
                "not found on PATH. It is an optional external component (GPL-3.0-or-later) "
                "and is NOT bundled with AVA — install it separately "
                "(Windows: winget install espeak-ng / https://github.com/espeak-ng/espeak-ng/releases, "
                "Linux: apt install espeak-ng, macOS: brew install espeak-ng) and make sure "
                "'espeak-ng' is on PATH."
            ) from exc
        except subprocess.SubprocessError as exc:
            raise PhonemizerUnavailable(
                f"eSpeak NG failed to phonemize the text ({exc.__class__.__name__}: {exc})."
            ) from exc
        return proc.stdout.strip()

    def phonemize(self, text: str) -> str:
        """IPA phonemes with per-clause punctuation (id_map members only).

        Clauses are split on ASCII *and* Persian punctuation. Punctuation runs
        are filtered down to the model-table subset and glue directly onto the
        preceding IPA (verified against the official calibration input:
        `ˈɑst.` — no space before the mark); word chunks are joined with
        single spaces. Persian-only marks contribute no tokens.
        """
        text = (text or "").strip()
        if not text:
            return ""
        parts = re.split("([" + re.escape("".join(_SPLIT_PUNCT)) + "]+)", text)
        buf = ""
        for chunk in parts:
            if not chunk:
                continue
            if all(c in _SPLIT_PUNCT for c in chunk):
                kept = "".join(c for c in chunk if c in _KEEP_PUNCT)
                if kept:
                    buf += kept
            else:
                ipa = self._ipa(chunk)
                if ipa:
                    buf = f"{buf} {ipa}".strip() if buf else ipa
        return buf.strip()

    def encode(self, phonemes: str, id_map: dict) -> list[int]:
        """Piper phoneme IDs: [BOS, PAD] + (id, PAD) per mapped char + [EOS].

        Verified byte-for-byte against the model's official calibration input.
        ``$`` (EOS) and ``^`` (BOS) appearing as literal characters inside the
        phoneme string are skipped — they are control symbols, and the
        terminator is appended exactly once at the end (a literal ``$`` from a
        punctuation run must not end the utterance mid-stream).
        """
        for sym in (_PAD, _BOS, _EOS):
            if sym not in id_map:
                raise ModelUnavailable(
                    f"phoneme table is missing the Piper control symbol {sym!r} — "
                    "the model config does not follow the Piper convention."
                )
        ids = [id_map[_BOS][0], id_map[_PAD][0]]
        for ch in phonemes:
            if ch in (_BOS, _EOS):  # control symbols: never mid-stream
                continue
            if ch in id_map:
                ids.append(id_map[ch][0])
                ids.append(id_map[_PAD][0])
        ids.append(id_map[_EOS][0])
        return ids


class AliFathiTTSBackend:
    """Pretrained Ali Fathi Persian VITS — ONNX Runtime, 7 speakers, 22050 Hz."""

    def __init__(self, model_dir: str | None = None, espeak_voice: str = "fa"):
        self.model_dir = Path(model_dir or default_model_dir()).expanduser()
        onnx_path, config_path = self._resolve_paths()
        cfg = self._load_config(config_path)
        self.cfg = cfg
        self.sample_rate = int(cfg.get("audio", {}).get("sample_rate") or 22050)
        self.id_map = cfg.get("phoneme_id_map") or {}
        self.speaker_map = cfg.get("speaker_id_map") or {}
        inf = cfg.get("inference") or {}
        self.noise_scale = float(inf.get("noise_scale", 0.667))
        self.length_scale = float(inf.get("length_scale", 1.0))
        self.noise_w = float(inf.get("noise_w", 0.8))
        self.phonemizer = EspeakNGPhonemizer(espeak_voice)
        self.session = self._load_session(onnx_path)

    # ---- loading -------------------------------------------------------------
    def _resolve_paths(self) -> tuple[Path, Path]:
        if not self.model_dir or not Path(self.model_dir).is_dir():
            raise ModelUnavailable(
                f"Ali Fathi ONNX model directory not found: {str(self.model_dir) or '(unset)'}. "
                f"Provide a local directory containing model.onnx + model.onnx.json "
                f"(official source: {MODEL_HF_URL}) via model_dir=…, the "
                f"{MODEL_DIR_ENV} environment variable, or place the companion files "
                f"in <repo>/alifathi/. No network access happens at runtime."
            )
        onnx_path = self.model_dir / "model.onnx"
        config_path = self.model_dir / "model.onnx.json"
        if not onnx_path.is_file() or not config_path.is_file():
            raise ModelUnavailable(
                f"incomplete Ali Fathi model directory {self.model_dir}: expected "
                f"'model.onnx' and 'model.onnx.json' (companion files from "
                f"{MODEL_HF_URL} → onnx/). Got: {sorted(p.name for p in self.model_dir.iterdir())}."
            )
        return onnx_path, config_path

    @staticmethod
    def _load_config(config_path: Path) -> dict:
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise ModelUnavailable(f"cannot read voice config {config_path}: {exc}") from exc

    @staticmethod
    def _load_session(onnx_path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelUnavailable(
                "onnxruntime is required for the Ali Fathi backend "
                "(pip install onnxruntime — or install AVA with the 'alifathi' extra: "
                "pip install \".[alifathi]\") — CPU inference, no GPU needed."
            ) from exc
        try:
            return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        except Exception as exc:  # corrupted/foreign file → clear failure
            raise ModelUnavailable(f"cannot load ONNX model {onnx_path}: {exc}") from exc

    # ---- SynthesisModel contract ----------------------------------------------
    def _voices(self) -> list[Voice]:
        out = []
        for name, sid in sorted(self.speaker_map.items(), key=lambda kv: int(kv[1])):
            out.append(Voice(
                voice_id=f"{VOICE_PREFIX}-{name}", display_name=f"علی فتحی (ONNX): {name}",
                language="fa", backend=BACKEND_FORMAT, ref=str(sid),
                description=(
                    f"صدای pretrained مدل VITS فارسی (alifthi/persian-TTS)، گویندهٔ «{name}» "
                    "(fleurs / GPTInformal / Mana-TTS / Persian-Speech) — وزن CC-BY-4.0 با ذکر منبع"
                ),
                license="CC-BY-4.0 (weights; attribution required — see NOTICE.md)",
                commercial_ok=True,
            ))
        return out

    def list_voices(self) -> list[Voice]:
        return self._voices()

    def info(self) -> dict:
        return {"format": BACKEND_FORMAT, "sample_rate": self.sample_rate,
                "speakers": len(self.speaker_map),
                "license": "CC-BY-4.0 (weights; attribution required)",
                "commercial": True, "model": str(self.model_dir),
                "frontend": f"espeak-ng ({self.phonemizer.voice}) + Piper phoneme table",
                "source": MODEL_HF_URL}

    def synthesize(self, text: str, voice: Voice, options: SynthesisOptions) -> SynthesisResult:
        # validated options first — invalid input must fail closed, not produce broken audio
        # (only an explicit None maps to the 1.0 default; 0/negative/NaN/inf must raise)
        speed = validate_speed(1.0 if options.speed is None else options.speed)
        target_sr = validate_sample_rate(options.sample_rate)

        t0 = time.time()
        phonemes = self.phonemizer.phonemize(text)
        if not phonemes:
            raise ValueError(f"text produced no phonemes: {text!r}")
        ids = np.array([self.phonemizer.encode(phonemes, self.id_map)], dtype=np.int64)
        length_scale = float(np.clip(self.length_scale / speed, 0.1, 4.0))
        scales = np.array([self.noise_scale, length_scale, self.noise_w], dtype=np.float32)
        sid = int(voice.ref or 0)
        audio = self.session.run(None, {
            "input": ids,
            "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
            "scales": scales,
            "sid": np.array([sid], dtype=np.int64),
        })[0]
        wav = np.asarray(audio[0, 0], dtype=np.float32)
        sr = self.sample_rate
        if target_sr is not None and target_sr != sr:
            # linear-interpolation resample (no external resampler dependency)
            n = max(1, int(round(len(wav) * target_sr / sr)))
            wav = np.interp(np.linspace(0.0, len(wav) - 1.0, n),
                            np.arange(len(wav)), wav).astype(np.float32)
            sr = target_sr
        elapsed = time.time() - t0
        return SynthesisResult(audio=wav, sample_rate=sr, voice=voice.voice_id,
                               text=text, duration_s=len(wav) / sr, elapsed_s=elapsed,
                               rtf=elapsed / max(len(wav) / sr, 1e-6))
