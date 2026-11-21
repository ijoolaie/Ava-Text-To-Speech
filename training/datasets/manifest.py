# -*- coding: utf-8 -*-
"""Dataset manifest for the AVA Native training pipeline.

One JSON line per utterance. Provenance and consent fields are mandatory so
no untraceable audio can enter training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class ManifestEntry:
    id: str
    audio_path: str            # relative to the dataset root
    text: str                  # raw reference text (before normalization)
    speaker_id: str
    sample_rate: int = 0       # filled by validate/prepare
    duration_s: float = 0.0
    channels: int = 0
    language: str = "fa"
    source: str = ""           # dataset/recording origin
    license: str = ""          # license of THIS utterance's audio
    consent_ref: str = ""      # consent/reference id for recorded speakers
    split: str = ""            # train | val | test (filled by split)
    valid: bool = True
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ManifestEntry":
        return cls(**json.loads(line))


def read_manifest(path: str) -> list[ManifestEntry]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(ManifestEntry.from_json(line))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"manifest line {i + 1} is malformed: {e}") from e
    return entries


def write_manifest(path: str, entries: list[ManifestEntry]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.to_json() + "\n")


def dataset_fingerprint(entries: list[ManifestEntry]) -> str:
    """Stable content hash of the manifest (dataset version identity)."""
    canonical = "\n".join(e.to_json() for e in sorted(entries, key=lambda x: x.id))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def provenance_report(entries: list[ManifestEntry]) -> dict:
    """Count utterances per source/license/consent — the audit view."""
    def group(key):
        out: dict[str, int] = {}
        for e in entries:
            out[getattr(e, key) or "(unset)"] = out.get(getattr(e, key) or "(unset)", 0) + 1
        return out
    return {
        "total": len(entries),
        "sources": group("source"),
        "licenses": group("license"),
        "consent_missing": sum(1 for e in entries if not e.consent_ref),
        "invalid": sum(1 for e in entries if not e.valid),
        "splits": group("split"),
    }
