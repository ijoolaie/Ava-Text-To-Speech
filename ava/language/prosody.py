# -*- coding: utf-8 -*-
"""Punctuation→pause mapping for prosody control (AVA Native language engine).

The pause table is used by the native model's duration/post-processing stage
and by the API when callers ask for explicit pacing. Values are seconds.
"""
from __future__ import annotations

# Persian kept punctuation -> pause after the mark, in seconds
PUNCT_PAUSE_SECONDS: dict[str, float] = {
    "،": 0.25,
    "؛": 0.40,
    ":": 0.40,
    ".": 0.50,
    "!": 0.50,
    "؟": 0.55,
    "…": 0.60,
}

DEFAULT_PAUSE = 0.15


def pause_after(punct: str) -> float:
    return PUNCT_PAUSE_SECONDS.get(punct, DEFAULT_PAUSE)


def estimate_duration(text: str, words_per_second: float = 2.6) -> float:
    """Rough speech duration estimate (words + pause budget), in seconds."""
    import re

    words = len(re.findall(r"[\u0600-\u06ff\w]+", text))
    pauses = sum(PUNCT_PAUSE_SECONDS.get(c, 0.0) for c in text)
    return words / max(words_per_second, 0.1) + pauses
