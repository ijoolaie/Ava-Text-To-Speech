# -*- coding: utf-8 -*-
"""نرمال‌سازی متن فارسی برای TTS — پورت پایتونی رفتار «persian.js» دموی آوا.

گام‌ها همان‌ها هستند: نویسه‌های عربی به فارسی، حذف اعراب و نویسه‌های نامرئی،
تبدیل ارقام به کلمات فارسی، درصد، نقطه‌گذاری، اصلاح نیم‌فاصله و حذف نویسه‌های مجاز‌نشده.
خروجی فقط شامل حروف فارسی، نیم‌فاصله، فاصله و «،؛:.!؟» است — همان چیزی که مدل انتظار دارد.

اجرا به‌صورت خط فرمان برای آزمایش:
    python3 normalize_fa.py "قیمت ۲۵۰ هزار تومان است و 12% تخفیف دارد."
"""
from __future__ import annotations

import re
import sys
import unicodedata

ZWNJ = "\u200c"

# حروف فارسی مجاز (پس از نگاشت عربی به فارسی)
PERSIAN_LETTERS = "آاءأإئؤپبتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

# نقطه‌گذاری‌هایی که نگه داشته می‌شوند
KEPT_PUNCT = "،؛:.!؟"

# نگاشت نویسه‌های عربی/ناخالص به فارسی
CHAR_MAP = {
    "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "آ": "آ",
    "ؤ": "و", "ئ": "ی", "ء": " ",
    "\u0640": " ",  # تطویل
}

# ارقام فارسی و عربی به لاتین
DIGIT_MAP = {ch: str(i) for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹")}
DIGIT_MAP.update({ch: str(i) for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩")})

# نقطه‌گذاری‌های دیگر به معادل مجاز یا فاصله
PUNCT_MAP = {",": "،", ";": "؛", "?": "؟", "\u066d": " ", "\u06d4": "."}

ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
SCALES = [(1_000_000_000, "میلیارد"), (1_000_000, "میلیون"), (1_000, "هزار")]
FRACTION_UNITS = {1: "دهم", 2: "صدم", 3: "هزارم", 4: "ده‌هزارم"}
JOIN = " و "
MAX_NUMBER_DIGITS = 15

ALLOWED = set(PERSIAN_LETTERS + KEPT_PUNCT + " " + ZWNJ)
DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0610-\u061a]")
INVISIBLES = re.compile(r"[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")

THOUSANDS_RE = re.compile(r"(?<=[0-9]),(?=[0-9]{3}(?![\w\u0600-\u06ff]))")
NUMBER_RE = re.compile(r"([0-9]+)(?:\.([0-9]+))?")


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


def _digits_one_by_one(digits: str) -> str:
    return " ".join("صفر" if d == "0" else ONES[int(d)] for d in digits)


def _number_token_to_words(whole: str, frac: str | None) -> str:
    if len(whole) > MAX_NUMBER_DIGITS or (whole.startswith("0") and len(whole) > 1):
        return " " + _digits_one_by_one(whole) + (" " + _digits_one_by_one(frac) if frac else "") + " "
    whole = whole.lstrip("0") or "0"
    words = number_to_words(int(whole))
    if frac:
        frac = frac.rstrip("0")
        if not frac:
            return f" {words} "
        unit = FRACTION_UNITS.get(len(frac))
        words += " ممیز " + number_to_words(int(frac)) + (" " + unit if unit else " " + _digits_one_by_one(frac))
    return f" {words} "


def numbers_to_words(text: str) -> str:
    text = THOUSANDS_RE.sub("", text)
    return NUMBER_RE.sub(lambda m: _number_token_to_words(m.group(1), m.group(2)), text)


def _map_chars(text: str, mapping: dict[str, str]) -> str:
    return "".join(mapping.get(c, c) for c in text)


def normalize(text: str, spell_numbers: bool = True) -> str:
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLES.sub("", text)
    text = DIACRITICS.sub("", text)
    text = _map_chars(text, CHAR_MAP)
    text = _map_chars(text, DIGIT_MAP)
    text = text.replace("٬", "").replace("٫", ".")
    text = re.sub("[٪%]", " درصد ", text)
    if spell_numbers:
        text = numbers_to_words(text)
    text = _map_chars(text, PUNCT_MAP)
    text = "".join(c if c in ALLOWED else " " for c in text)
    text = re.sub(f"{ZWNJ}+", ZWNJ, text)
    text = re.sub(rf"\s*{ZWNJ}\s*", lambda m: " " if " " in m.group(0) else ZWNJ, text)
    text = re.sub(rf"(?<![{PERSIAN_LETTERS}]){ZWNJ}|{ZWNJ}(?![{PERSIAN_LETTERS}])", "", text)
    text = re.sub(rf"\s+([{re.escape(KEPT_PUNCT)}])", r"\1", text)
    text = re.sub(rf"([{re.escape(KEPT_PUNCT)}])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    i = 0
    while i < len(text) and (text[i] in KEPT_PUNCT or text[i] == " "):
        i += 1
    return text[i:]


def with_end(text: str) -> str:
    text = text.strip()
    if text == "" or text[-1] in ".؟!":
        return text
    return text + "."


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """جمله‌بندی ساده برای CLI: بریدن روی «.؟!» و در نهایت روی «،؛:» برای جمله‌های خیلی بلند."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!؟])\s+", re.sub(r"\s+", " ", text).strip()):
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(c, 0, max_chars) for c in "،؛:")
            if cut <= 0:
                cut = max_chars
            out.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            out.append(sentence)
    return out


# ---------- شمارش توکن BPE مدل (اختیاری؛ برای اندازه‌گیری قطعه‌ها) ----------
class TokenCounter:
    """همان منطق makeTokenCounter در persian.js: BPE با جدول merges مدل و شمارش توکن‌ها.

    tokenizer.json فایل‌های unknown پیوسته را یکی می‌شمارد؛ نیم‌فاصله‌ها پیش از شمارش
    به فاصله تبدیل می‌شوند چون نرمالایزرِ توکنایزر همین کار را می‌کند.
    """

    def __init__(self, tokenizer_json: dict):
        model = tokenizer_json["model"]
        self.vocab: set[str] = set(model["vocab"])
        self.rank: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = m if isinstance(m, list) else tuple(m.split(" "))
            self.rank[(a, b)] = i
        self.cache: dict[str, int] = {}

    def _word_tokens(self, word: str) -> int:
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while True:
            best, best_rank = -1, 1 << 30
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]), None)
                if r is not None and r < best_rank:
                    best_rank, best = r, i
            if best < 0:
                break
            parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2 :]
        n, prev_unknown = 0, False
        for p in parts:
            unknown = p not in self.vocab
            if not (unknown and prev_unknown):
                n += 1
            prev_unknown = unknown
        self.cache[word] = n
        return n

    def __call__(self, text: str) -> int:
        return sum(self._word_tokens("▁" + w) for w in text.replace(ZWNJ, " ").split(" ") if w)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(normalize(" ".join(sys.argv[1:])))
    else:
        demo = "قیمت این کتاب ۲۵۰ هزار تومان است و ۱۲ درصد تخفیف دارد. نسخهٔ 1.5 را می‌خواهم!"
        print(normalize(demo))
