# -*- coding: utf-8 -*-
"""Deterministic dataset splitting (AVA Native pipeline).

Splitting is a pure function of the utterance id and a seed, so re-running
the pipeline always yields identical splits; speaker-disjoint mode guarantees
no speaker appears in two splits (required for honest speaker generalization).
"""
from __future__ import annotations

import hashlib


def _bucket(uid: str, seed: str) -> float:
    h = hashlib.sha256(f"{seed}:{uid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_splits(entries, ratios: tuple[float, float, float] = (0.9, 0.05, 0.05),
                  seed: str = "ava-native-v1", speaker_disjoint: bool = True) -> dict[str, int]:
    """Assign entry.split in place; returns counts per split."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    train_r, val_r = ratios[0], ratios[0] + ratios[1]

    if not speaker_disjoint:
        for e in entries:
            b = _bucket(e.id, seed)
            e.split = "train" if b < train_r else ("val" if b < val_r else "test")
        return _counts(entries)

    speakers = sorted({e.speaker_id for e in entries})
    spk_split: dict[str, str] = {}
    for s in speakers:
        b = _bucket(s, seed + ":speaker")
        spk_split[s] = "train" if b < train_r else ("val" if b < val_r else "test")
    for e in entries:
        e.split = spk_split[e.speaker_id]
    return _counts(entries)


def _counts(entries) -> dict[str, int]:
    out = {"train": 0, "val": 0, "test": 0}
    for e in entries:
        out[e.split] = out.get(e.split, 0) + 1
    return out
