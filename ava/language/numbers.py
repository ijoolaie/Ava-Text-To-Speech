# -*- coding: utf-8 -*-
"""Number handling for Persian speech: cardinal words, ordinals, serial digits.

Original implementation of the AVA Native language engine; the cardinal
reader is the same algorithm as the legacy normalizer (kept byte-compatible).
"""
from __future__ import annotations

import re

from .chars import to_ascii_digits

ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
SCALES = [(1_000_000_000, "میلیارد"), (1_000_000, "میلیون"), (1_000, "هزار")]
FRACTION_UNITS = {1: "دهم", 2: "صدم", 3: "هزارم", 4: "ده‌هزارم"}
JOIN = " و "
MAX_NUMBER_DIGITS = 15
MIN_SERIAL_DIGITS = 7

THOUSANDS_RE = re.compile(r"(?<=[0-9]),(?=[0-9]{3}(?![\w\u0600-\u06ff]))")
NUMBER_RE = re.compile(r"([0-9]+)(?:\.([0-9]+))?")
# Candidate phone runs: starts with 0, then digits/-/space, 10–12 digits total
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w\u0600-\u06ff])0[\d\- ]{9,14}\d(?![\d])")


def _under_thousand(n: int) -> str:
    parts: list[str] = []
    if n >= 100:
        parts.append(HUNDREDS[n // 100])
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    if 10 <= n < 20:
        parts.append(TEENS[n - 10])
        n = 0
    if n > 0:
        parts.append(ONES[n])
    return JOIN.join(p for p in parts if p)


def number_to_words(n: int) -> str:
    """Cardinal reading of an integer in Persian (e.g. 1403 -> هزار و چهارصد و سه)."""
    if n < 0:
        return "منفی " + number_to_words(-n)
    if n == 0:
        return "صفر"
    parts: list[str] = []
    for value, name in SCALES:
        if n >= value:
            count = n // value
            n %= value
            head = "" if (count == 1 and value == 1000) else _under_thousand(count) + " "
            parts.append((head + name).strip())
    if n > 0:
        parts.append(_under_thousand(n))
    return JOIN.join(parts)


ORDINAL_IRREGULAR = {1: "اول", 2: "دوم", 3: "سوم"}
ORDINAL_SUFFIX = "م"


def ordinal_words(n: int) -> str:
    """Ordinal reading (e.g. 21 -> بیست و یکم، 30 -> سی‌ام، 1 -> اول)."""
    if n in ORDINAL_IRREGULAR:
        return ORDINAL_IRREGULAR[n]
    cardinal = number_to_words(n)
    if JOIN in cardinal:
        head, last = cardinal.rsplit(JOIN, 1)
        if last == "سه":
            last = "سو"
        return head + JOIN + last + ORDINAL_SUFFIX
    if cardinal.endswith("ی"):  # round tens: سی، چهل، ... → سی‌ام
        return cardinal + "\u200cام"
    if cardinal == "سه":
        cardinal = "سو"
    return cardinal + ORDINAL_SUFFIX


def spell_digits(digits: str) -> str:
    """Digit-by-digit reading for phone numbers, codes and serials."""
    return " ".join("صفر" if d == "0" else ONES[int(d)] for d in digits)


def _number_token_to_words(whole: str, frac: str | None) -> str:
    if len(whole) > MAX_NUMBER_DIGITS or (whole.startswith("0") and len(whole) > 1):
        return " " + spell_digits(whole) + (" " + spell_digits(frac) if frac else "") + " "
    whole = whole.lstrip("0") or "0"
    words = number_to_words(int(whole))
    if frac:
        frac = frac.rstrip("0")
        if not frac:
            return f" {words} "
        unit = FRACTION_UNITS.get(len(frac))
        words += " ممیز " + number_to_words(int(frac)) + (" " + unit if unit else " " + spell_digits(frac))
    return f" {words} "


def numbers_to_words(text: str) -> str:
    """Read every plain decimal number in the text as Persian words."""
    text = THOUSANDS_RE.sub("", text)
    return NUMBER_RE.sub(lambda m: _number_token_to_words(m.group(1), m.group(2)), text)


def expand_phone_numbers(text: str) -> str:
    """Read Iranian phone numbers digit by digit (landline 0xx-xxxxxxx, mobile 09xxxxxxxxx)."""

    def _repl(m: re.Match) -> str:
        digits = re.sub(r"[\- ]", "", to_ascii_digits(m.group(0)))
        if digits.isdigit() and digits.startswith("0") and 10 <= len(digits) <= 12:
            return " " + spell_digits(digits) + " "
        return m.group(0)

    return PHONE_CANDIDATE_RE.sub(_repl, text)
