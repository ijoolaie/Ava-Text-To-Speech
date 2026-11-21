# -*- coding: utf-8 -*-
"""Voice profiles and store (AVA Native voice system, phase 6).

A voice profile is a JSON document describing one voice: identity, backend
routing, license/commercial flags and measured quality. Profiles live outside
model code so new voices (recordings or trained) can be added without code
changes. Voice cloning stays OFF until a consent reference exists — the store
refuses profiles with `requires_consent=True` and empty `consent_ref`.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class VoiceProfile:
    voice_id: str
    display_name: str
    language: str = "fa"
    gender: str = ""                 # female/male/neutral
    backend: str = ""                # "ava-n1" | "legacy-pocket-tts" | ...
    ref: str = ""                    # backend reference (speaker id / prompt path)
    description: str = ""
    license: str = ""
    commercial_ok: bool = False
    requires_consent: bool = False   # true for cloned/recorded voices
    consent_ref: str = ""
    quality: dict = field(default_factory=dict)  # measured: duration_s, snr_db, clipping, ...
    tags: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "VoiceProfile":
        d = json.loads(text)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


class VoiceStore:
    """File-backed collection of voice profiles (voices/profiles/*.json)."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save(self, profile: VoiceProfile) -> str:
        if profile.requires_consent and not profile.consent_ref:
            raise ValueError(f"voice {profile.voice_id!r} requires a consent reference (cloning without consent is disabled)")
        path = os.path.join(self.directory, profile.voice_id + ".json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(profile.to_json())
        return path

    def load(self, voice_id: str) -> VoiceProfile:
        path = os.path.join(self.directory, voice_id + ".json")
        with open(path, encoding="utf-8") as f:
            return VoiceProfile.from_json(f.read())

    def all(self) -> list[VoiceProfile]:
        out = []
        for fn in sorted(os.listdir(self.directory)):
            if fn.endswith(".json"):
                out.append(VoiceProfile.from_json(open(os.path.join(self.directory, fn), encoding="utf-8").read()))
        return out

    def delete(self, voice_id: str) -> None:
        path = os.path.join(self.directory, voice_id + ".json")
        if os.path.exists(path):
            os.remove(path)
