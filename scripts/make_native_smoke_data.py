# -*- coding: utf-8 -*-
"""Generate a synthetic smoke dataset for the native trainer (license-clean).

Each Persian sentence is rendered as a deterministic vowel-like tone pattern
(bursts per word, pitch from a content hash) so the training pipeline can be
verified end-to-end WITHOUT any external audio. This data is a test fixture,
not real speech, and is explicitly labeled as such in the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.datasets.manifest import ManifestEntry, write_manifest  # noqa: E402
from training.datasets.split import assign_splits  # noqa: E402
from training.datasets.validate import validate_audio  # noqa: E402

SENTENCES = [
    "سلام دنیا",
    "کتاب خواندن خوب است",
    "صدا طبیعی است",
    "امروز هوا آفتابی است",
    "ما به بازار می‌رویم",
    "این جمله کمی طولانی‌تر است و چند کلمه دارد",
    "قیمت کتاب دویست تومان است",
    "شب‌ها ستاره‌ها را می‌بینم",
    "زبان فارسی زیباست",
    "موزیک آرام پخش می‌شود",
    "کودکان در پارک بازی می‌کنند",
    "چای تازه دم شده لذت‌بخش است",
    "برنامه‌نویسی مهارت خلاقیت می‌خواهد",
    "پنجره باز است و نسیم می‌آید",
    "خواندن شعر حس خوبی می‌دهد",
    "دوستی ارزشمندترین سرمایه است",
    "او هر روز صبح می‌دود",
    "آشپزخانه بوی نان می‌دهد",
    "کوه در زمستان پوشیده از برف است",
    "یادگیری تدریجی و پیوسته است",
]


def render_sentence(text: str, seed: str, sr: int = 24000) -> np.ndarray:
    """Deterministic tone-burst rendering: one burst per word, pitch per sentence."""
    h = int(hashlib.sha256((seed + text).encode()).hexdigest()[:8], 16)
    f0 = 150 + (h % 140)
    words = text.split()
    parts = []
    for wi, w in enumerate(words):
        dur = 0.25 + 0.03 * (len(w) % 5)
        n = int(dur * sr)
        t = np.arange(n) / sr
        wf = f0 + 18 * ((wi + h) % 3)
        # attack/release envelope with flat sustain — avoids artificial SNR flags
        env = np.minimum(1.0, t * 40) * np.minimum(1.0, np.maximum(dur - t, 0) * 8)
        vib = 1 + 0.01 * np.sin(2 * np.pi * 5.5 * t)
        parts.append(0.5 * env * np.sin(2 * np.pi * wf * vib * t))
        parts.append(np.zeros(int(0.06 * sr)))
    wav = np.concatenate(parts)
    peak = np.abs(wav).max() + 1e-8
    return (0.85 * wav / peak).astype(np.float32)


def generate(out_dir: str, n: int = 20, sr: int = 24000, speakers: int = 3) -> str:
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    entries = []
    for i in range(n):
        text = SENTENCES[i % len(SENTENCES)]
        name = f"smoke_{i:04d}"
        path = os.path.join(audio_dir, name + ".wav")
        sf.write(path, render_sentence(text, f"v1-{i}"), sr)
        e = ManifestEntry(
            id=name, audio_path=path, text=text, speaker_id=f"smoke-spk-{i % speakers + 1}",
            sample_rate=sr, language="fa",
            source="synthetic-fixture (tone bursts, not real speech)",
            license="MIT (this repository)", consent_ref="N/A-synthetic",
        )
        validate_audio(e, strict_sr=True)
        entries.append(e)
    assign_splits(entries, seed="smoke-native-v1", speaker_disjoint=False)
    # fixture guarantee: val/test must be non-empty for trainer validation paths
    splits = {e.split for e in entries}
    if "val" not in splits and len(entries) >= 2:
        entries[-1].split = "val"
    if "test" not in splits and len(entries) >= 3:
        entries[-2].split = "test"
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    write_manifest(manifest_path, entries)
    return manifest_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "smoke_native")
    mp = generate(out, n=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    from training.datasets.manifest import provenance_report, read_manifest
    print(json.dumps(provenance_report(read_manifest(mp)), ensure_ascii=False))
    print("manifest:", mp)
