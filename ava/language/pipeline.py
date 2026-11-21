# -*- coding: utf-8 -*-
"""The AVA Native language pipeline: orchestration of every text stage.

Stages (each one optional and individually testable):

    clean → lexicon → phones → dates/times → currency → units → ordinals
          → latin policy → percent → numbers → punctuation → finalize

`normalize()` reproduces the legacy behavior byte-for-byte (used by the
legacy Pocket-TTS path); `LanguageEngine.normalize()` applies the full
configurable pipeline used by AVA Native.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .chars import PUNCT_MAP, clean, finalize, to_ascii_digits
from .datetime_speech import expand_dates_and_times
from .language_detect import spell_latin_word
from .numbers import NUMBER_RE, numbers_to_words, number_to_words, ordinal_words
from .pronunciation import PronunciationLexicon
from .prosody import pause_after  # re-export convenience

ZWNJ = "\u200c"

# --- units -----------------------------------------------------------------
# Longest-first so km/h wins over km, khz over hz, and so on.
UNIT_WORDS = {
    "km/h": "کیلومتر بر ساعت", "kcal": "کالری", "kwh": "کیلووات‌ساعت",
    "kg": "کیلوگرم", "km": "کیلومتر", "cm": "سانتی‌متر", "mm": "میلی‌متر",
    "khz": "کیلوهرتز", "mhz": "مگاهرتز", "ghz": "گیگاهرتز",
    "gb": "گیگابایت", "mb": "مگابایت", "kb": "کیلوبایت",
    "kw": "کیلووات", "hz": "هرتز", "db": "دسی‌بل",
    "min": "دقیقه", "ml": "میلی‌لیتر", "kg": "کیلوگرم",
    "g": "گرم", "m": "متر", "s": "ثانیه", "h": "ساعت",
    "t": "تن", "l": "لیتر", "w": "وات",
}
_UNIT_NAMES = sorted(UNIT_WORDS, key=len, reverse=True)
UNIT_RE = re.compile(r"(?<=[0-9])[ ]?(" + "|".join(re.escape(u) for u in _UNIT_NAMES) + r")(?![A-Za-z0-9])", re.IGNORECASE)

CURRENCY_AFTER_RE = re.compile(r"([0-9][0-9,]*(?:\.[0-9]+)?)[ ]?[$€£]")
CURRENCY_BEFORE_RE = re.compile(r"[$€£][ ]?([0-9][0-9,]*(?:\.[0-9]+)?)")
CURRENCY_WORDS = {"$": "دلار", "€": "یورو", "£": "پوند"}

ORDINAL_RE = re.compile(r"(?<![\d\u0600-\u06ff])([0-9]{1,4})[ ]?\u200c?[ ]?اُ?م(?![\u0600-\u06ff])")

LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")
PERCENT_RE = re.compile(r"[٪%]")


@dataclass(frozen=True)
class LanguageConfig:
    """Pipeline switches. `LEGACY_CONFIG` reproduces the old normalizer."""

    latin_policy: str = "drop"      # drop | spell | keep
    expand_dates: bool = True       # dates and clock times
    expand_units: bool = True       # unit abbreviations and currency symbols
    expand_ordinals: bool = True    # «۲ام» → دوم
    spell_phones: bool = True       # phone numbers digit by digit
    apply_lexicon: bool = True      # pronunciation lexicon entries
    lexicon_path: str | None = None
    spell_numbers: bool = True


LEGACY_CONFIG = LanguageConfig(
    latin_policy="drop", expand_dates=False, expand_units=False,
    expand_ordinals=False, spell_phones=False, apply_lexicon=False,
    spell_numbers=True,
)


def _latin_policy(text: str, policy: str) -> str:
    if policy == "keep":
        return text
    if policy == "drop":
        return LATIN_TOKEN_RE.sub(" ", text)
    if policy == "spell":
        return " ".join(
            spell_latin_word(t) if re.fullmatch(r"[A-Za-z][A-Za-z\-]*", t) else t
            for t in text.split(" ")
        )
    raise ValueError(f"unknown latin policy: {policy}")


def _expand_ordinals(text: str) -> str:
    return ORDINAL_RE.sub(lambda m: f" {ordinal_words(int(m.group(1)))} ", text)


def _expand_units(text: str) -> str:
    text = UNIT_RE.sub(lambda m: " " + UNIT_WORDS[m.group(1).lower()] + " ", text)
    text = CURRENCY_AFTER_RE.sub(lambda m: f"{m.group(1)} {CURRENCY_WORDS[m.group(0)[-1]]} ", text)
    text = CURRENCY_BEFORE_RE.sub(lambda m: f"{m.group(1)} {CURRENCY_WORDS[m.group(0)[0]]} ", text)
    return text


class LanguageEngine:
    """Configurable text-processing pipeline for Persian speech."""

    def __init__(self, config: LanguageConfig | None = None, lexicon: PronunciationLexicon | None = None):
        self.config = config or LanguageConfig()
        self.lexicon = None
        if self.config.apply_lexicon:
            self.lexicon = lexicon or PronunciationLexicon.load(self.config.lexicon_path)

    def normalize(self, text: str) -> str:
        cfg = self.config
        text = clean(text)
        if cfg.apply_lexicon and self.lexicon is not None:
            text = self.lexicon.apply(text)
        if cfg.spell_phones:
            from .numbers import expand_phone_numbers
            text = expand_phone_numbers(text)
        if cfg.expand_dates:
            text = expand_dates_and_times(text)
        if cfg.expand_units:
            text = _expand_units(text)
        if cfg.expand_ordinals:
            text = _expand_ordinals(text)
        text = _latin_policy(text, cfg.latin_policy)
        text = PERCENT_RE.sub(" درصد ", text)
        if cfg.spell_numbers:
            text = numbers_to_words(text)
        text = "".join(PUNCT_MAP.get(c, c) for c in text)
        return finalize(text)


def normalize(text: str, spell_numbers: bool = True) -> str:
    """Legacy-compatible normalizer (byte-identical to the historical implementation)."""
    text = clean(text)
    if spell_numbers:
        text = PERCENT_RE.sub(" درصد ", text)
        text = numbers_to_words(text)
    text = "".join(PUNCT_MAP.get(c, c) for c in text)
    return finalize(text)
