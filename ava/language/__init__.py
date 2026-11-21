# -*- coding: utf-8 -*-
"""ava.language — Persian language engine of AVA Native (original code).

Public API:
    LanguageEngine / LanguageConfig / LEGACY_CONFIG   pipeline
    normalize(text)                                    legacy-compatible normalizer
    split_sentences / split_paragraphs / with_end      segmentation
    number_to_words / ordinal_words / spell_digits     numbers
    expand_dates_and_times                             dates & clock times
    detect_primary / token_scripts / spell_latin_word  language detection
    PronunciationLexicon                               pronunciation
    pause_after / PUNCT_PAUSE_SECONDS / estimate_duration  prosody
    TokenCounter                                       BPE token counting
"""
from .chars import ZWNJ
from .datetime_speech import date_to_words, expand_dates_and_times, gregorian_to_jalali
from .language_detect import detect_primary, spell_latin_word, token_script, token_scripts
from .numbers import number_to_words, ordinal_words, numbers_to_words, spell_digits
from .pipeline import LEGACY_CONFIG, LanguageConfig, LanguageEngine, normalize
from .pronunciation import PronunciationLexicon
from .prosody import PUNCT_PAUSE_SECONDS, estimate_duration, pause_after
from .segmentation import split_paragraphs, split_sentences, with_end
from .token_counter import TokenCounter

__all__ = [
    "LanguageEngine", "LanguageConfig", "LEGACY_CONFIG", "normalize",
    "split_sentences", "split_paragraphs", "with_end",
    "number_to_words", "ordinal_words", "numbers_to_words", "spell_digits",
    "expand_dates_and_times", "date_to_words", "gregorian_to_jalali",
    "detect_primary", "token_script", "token_scripts", "spell_latin_word",
    "PronunciationLexicon", "PUNCT_PAUSE_SECONDS", "pause_after", "estimate_duration",
    "TokenCounter", "ZWNJ",
]
