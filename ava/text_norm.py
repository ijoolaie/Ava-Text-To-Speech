# -*- coding: utf-8 -*-
"""سازگاری با نسخه‌های قبلی — پیاده‌سازی به بستهٔ ماژولار `ava.language` منتقل شده است.

این فایل فقط re-export می‌کند؛ رفتار `normalize` با نسخهٔ تاریخی بایت‌به‌بایت
یکسان است (تست‌های طلایی در tests/test_language_engine.py این را تثبیت کرده‌اند).
"""
from ava.language.chars import KEPT_PUNCT, PERSIAN_LETTERS, ZWNJ
from ava.language.numbers import number_to_words
from ava.language.pipeline import normalize
from ava.language.segmentation import split_sentences, with_end
from ava.language.token_counter import TokenCounter

__all__ = ["normalize", "split_sentences", "with_end", "number_to_words", "TokenCounter", "ZWNJ", "PERSIAN_LETTERS", "KEPT_PUNCT"]
