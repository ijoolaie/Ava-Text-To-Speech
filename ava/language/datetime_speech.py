# -*- coding: utf-8 -*-
"""Date and time reading for Persian speech (AVA Native language engine).

Supports Jalali (solar Hijri) and Gregorian dates in numeric formats plus
numeric clock times. The Gregorian→Jalali conversion implements the standard
published arithmetic (33-year cycle rule); the reading rules are original.

Runs AFTER digit normalization, so all patterns see ASCII digits.
"""
from __future__ import annotations

import re

from .numbers import number_to_words, ordinal_words

JALALI_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
                 "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
GREGORIAN_MONTHS = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن",
                    "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]

# Numeric date 1403/08/01 / 1403-08-01 (with . as a rare variant, guarded)
DATE_SLASH_RE = re.compile(r"(?<![\d\u0600-\u06ff])(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})(?![\d])")
# «۱۲ آبان ۱۴۰۳» — day + month name + year (optional «سال»)
DATE_NAME_RE = re.compile(r"(?<![\d\u0600-\u06ff])(\d{1,2}) (" + "|".join(JALALI_MONTHS + GREGORIAN_MONTHS) + r")( سال)? (\d{2,4})(?![\d])")
# Time 14:30 or 14:30:45 — «ساعت» keyword is consumed so it is never doubled
TIME_RE = re.compile(r"(?<![\d\u0600-\u06ff])(ساعت\s*)?(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d])")


def gregorian_to_jalali(gy: int, gm: int, gd: int) -> tuple[int, int, int]:
    """Standard Gregorian→Jalali arithmetic (33-year cycle)."""
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2, gm2, gd2 = gy - 1600, gm - 1, gd - 1
    g_day_no = 365 * gy2 + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
    g_day_no += g_d_m[gm2]  # cumulative days before the current month (non-leap table)
    if gm2 > 1 and ((gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0):
        g_day_no += 1
    g_day_no += gd2
    j_day_no = g_day_no - 79
    j_np = j_day_no // 12053
    j_day_no %= 12053
    jy = 979 + 33 * j_np + 4 * (j_day_no // 1461)
    j_day_no %= 1461
    if j_day_no >= 366:
        jy += (j_day_no - 1) // 365
        j_day_no = (j_day_no - 1) % 365
    if j_day_no < 186:
        jm, jd = 1 + j_day_no // 31, 1 + j_day_no % 31
    else:
        jm, jd = 7 + (j_day_no - 186) // 30, 1 + (j_day_no - 186) % 30
    return jy, jm, jd


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    """Standard Jalali→Gregorian arithmetic (inverse of the above)."""
    jy2, jm2, jd2 = jy - 979, jm - 1, jd - 1
    j_day_no = 365 * jy2 + (jy2 // 33) * 8 + (jy2 % 33 + 3) // 4
    for i in range(jm2):
        j_day_no += 31 if i < 6 else 30
    j_day_no += jd2
    g_day_no = j_day_no + 79
    gy = 1600 + 400 * (g_day_no // 146097)
    g_day_no %= 146097
    leap = True
    if g_day_no >= 36525:
        g_day_no -= 1
        gy += 100 * (g_day_no // 36524)
        g_day_no %= 36524
        if g_day_no >= 365:
            g_day_no += 1
        else:
            leap = False
    gy += 4 * (g_day_no // 1461)
    g_day_no %= 1461
    if g_day_no >= 366:
        leap = False
        g_day_no -= 1
        gy += g_day_no // 365
        g_day_no %= 365
    gd = g_day_no + 1
    g_md = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    for days in g_md:
        if gd <= days:
            break
        gd -= days
        gm += 1
    return gy, gm + 1, gd


def _is_leap_jalali(jy: int) -> bool:
    """Leap year by the 33-year cycle rule used above."""
    return jy % 33 in (1, 5, 9, 13, 17, 22, 26, 30)


def _validate_date(y: int, m: int, d: int, jalali: bool) -> bool:
    if jalali:
        days = 31 if m <= 6 else (30 if m <= 11 else (30 if _is_leap_jalali(y) else 29))
        return 1 <= m <= 12 and 1 <= d <= days and 1000 <= y <= 1600
    days = [31, 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
    return 1 <= m <= 12 and 1 <= d <= days and 1900 <= y <= 2100


def date_to_words(y: int, m: int, d: int, jalali: bool) -> str:
    month_name = (JALALI_MONTHS if jalali else GREGORIAN_MONTHS)[m - 1]
    year_words = number_to_words(y)
    return f"{ordinal_words(d)} {month_name} سال {year_words}"


def _read_date(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    jalali = y <= 1600
    if not _validate_date(y, mo, d, jalali):
        return m.group(0)
    return f" {date_to_words(y, mo, d, jalali)} "


def _read_date_named(m: re.Match) -> str:
    d, name, sal, y = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
    if name in GREGORIAN_MONTHS:
        jalali, mo = False, GREGORIAN_MONTHS.index(name) + 1
        if not (1900 <= y <= 2100):
            return m.group(0)
    else:
        jalali, mo = True, JALALI_MONTHS.index(name) + 1
        if not (1000 <= y <= 1600):
            return m.group(0)
    if not _validate_date(y, mo, d, jalali):
        return m.group(0)
    prefix = f"{ordinal_words(d)} {name}"
    return f" {prefix} {'سال ' if sal else ''}{number_to_words(y)} "


def _read_time(m: re.Match) -> str:
    keyword, h_g, mi_g, s_g = m.group(1), m.group(2), m.group(3), m.group(4)
    h, mi, s = int(h_g), int(mi_g), s_g
    if not (0 <= h <= 23 and 0 <= mi <= 59 and (s is None or 0 <= int(s) <= 59)):
        return m.group(0)
    hour_word = {"0": "نیمه‌شب", "12": "ظهر"}.get(str(h)) or f"ساعت {number_to_words(h)}"
    if h == 0 and mi == 0 and s is None:
        return " نیمه‌شب "
    if h == 12 and mi == 0 and s is None:
        return " ظهر "
    if mi == 0 and s is None:
        return f" {hour_word} "
    out = f" {hour_word} و {number_to_words(mi)} دقیقه"
    if s is not None:
        out += f" و {number_to_words(int(s))} ثانیه"
    return out + " "


def expand_dates_and_times(text: str) -> str:
    text = DATE_NAME_RE.sub(_read_date_named, text)
    text = DATE_SLASH_RE.sub(_read_date, text)
    text = TIME_RE.sub(_read_time, text)
    return text
