# -*- coding: utf-8 -*-
"""Script detection for mixed Persian/Latin text (AVA Native language engine).

Classifies tokens by script so the pipeline can decide how to treat Latin
words embedded in Persian sentences: drop them (legacy models), spell them
letter by letter, or keep them (native models with Latin support).
"""
from __future__ import annotations

import re

FA = re.compile(r"[\u0600-\u06ff]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")


def token_script(token: str) -> str:
    """fa | latin | digit | punct | mixed."""
    if not token:
        return "other"
    if FA.search(token) and not LATIN.search(token):
        return "fa"
    if LATIN.search(token) and not FA.search(token):
        return "latin" if DIGIT.search(token) is None else "mixed"
    if DIGIT.search(token) and not FA.search(token) and not LATIN.search(token):
        return "digit"
    if FA.search(token) and LATIN.search(token):
        return "mixed"
    if re.fullmatch(r"[،؛:.!؟\s]+", token):
        return "punct"
    return "other"


def token_scripts(text: str) -> list[tuple[str, str]]:
    return [(t, token_script(t)) for t in text.split()]


def detect_primary(text: str) -> str:
    """fa | latin | mixed — primary language of the text."""
    counts: dict[str, int] = {}
    for _, script in token_scripts(text):
        if script in ("fa", "latin"):
            counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "fa"
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed"


# Persian reading of Latin letters, used by the "spell" policy
LATIN_LETTERS = {
    "a": "اِی", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اف",
    "g": "جی", "h": "اِچ", "i": "آی", "j": "جی", "k": "کِی", "l": "ال",
    "m": "ام", "n": "اِن", "o": "اُو", "p": "پی", "q": "کیو", "r": "آر",
    "s": "اِس", "t": "تی", "u": "یو", "v": "وی", "w": "دابلیو", "x": "اکس",
    "y": "وای", "z": "زد",
}


def spell_latin_word(word: str) -> str:
    """Letter-by-letter Persian reading of a Latin word/acronym."""
    return " ".join(LATIN_LETTERS[c] for c in word.lower() if c in LATIN_LETTERS)
