# -*- coding: utf-8 -*-
"""Character-level processing for Persian text (AVA Native language engine).

This module owns the character maps and the two low-level stages every text
goes through: cleaning (unicode forms, invisible characters, Arabic letter
mapping, digit mapping) and finalization (allowed-alphabet filtering, ZWNJ
repair, punctuation spacing, whitespace collapse).

The behavior is intentionally identical to the legacy `ava.text_norm`
implementation so both inference paths stay byte-compatible.
"""
from __future__ import annotations

import re
import unicodedata

ZWNJ = "\u200c"

# Persian letters allowed in the output alphabet
PERSIAN_LETTERS = "آاءأإئؤپبتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# Punctuation kept in the output alphabet
KEPT_PUNCT = "،؛:.!؟"

# Arabic / mixed-alphabet characters mapped to their Persian equivalents
CHAR_MAP = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "آ": "آ",
    "ؤ": "و", "ئ": "ی", "ء": " ",
    "\u0640": " ",  # tatweel
}

# Persian and Arabic-Indic digits -> ASCII digits
DIGIT_MAP = {ch: str(i) for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹")}
DIGIT_MAP.update({ch: str(i) for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩")})

# Other punctuation mapped to a kept equivalent or a space
PUNCT_MAP = {",": "،", ";": "؛", "?": "؟", "\u066d": " ", "\u06d4": "."}

ALLOWED = set(PERSIAN_LETTERS + KEPT_PUNCT + " " + ZWNJ)
DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0610-\u061a]")
INVISIBLES = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")

LETTERS_CLASS = re.escape(PERSIAN_LETTERS)
KEPT_PUNCT_CLASS = re.escape(KEPT_PUNCT)


def is_persian_letter(c: str) -> bool:
    return c in PERSIAN_LETTERS


def map_chars(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in text)


def to_ascii_digits(text: str) -> str:
    return map_chars(text, DIGIT_MAP)


def clean(text: str) -> str:
    """Stage 1 of every pipeline: unicode form, invisibles, diacritics, char/digit maps."""
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLES.sub("", text)
    text = DIACRITICS.sub("", text)
    text = map_chars(text, CHAR_MAP)
    text = map_chars(text, DIGIT_MAP)
    text = text.replace("٬", "").replace("٫", ".")
    return text


def finalize(text: str) -> str:
    """Stage 3 of every pipeline: alphabet filter, ZWNJ repair, punctuation, whitespace.

    Input must already contain only Persian letters, kept punctuation, ZWNJ,
    spaces and any Latin/digit characters that survived earlier stages.
    """
    text = "".join(c if c in ALLOWED else " " for c in text)
    text = re.sub(f"{ZWNJ}+", ZWNJ, text)
    text = re.sub(rf"\s*{ZWNJ}\s*", lambda m: " " if " " in m.group(0) else ZWNJ, text)
    text = re.sub(rf"(?<![{LETTERS_CLASS}]){ZWNJ}|{ZWNJ}(?![{LETTERS_CLASS}])", "", text)
    text = re.sub(rf"\s+([{KEPT_PUNCT_CLASS}])", r"\1", text)
    text = re.sub(rf"([{KEPT_PUNCT_CLASS}])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    i = 0
    while i < len(text) and (text[i] in KEPT_PUNCT or text[i] == " "):
        i += 1
    return text[i:]
