# -*- coding: utf-8 -*-
"""فاز ۱ — تست‌های موتور زبان فارسی AVA Native (معیارهای AC1.1 تا AC1.7)."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.language import (  # noqa: E402
    LEGACY_CONFIG, LanguageConfig, LanguageEngine, PronunciationLexicon,
    detect_primary, estimate_duration, expand_dates_and_times, gregorian_to_jalali,
    number_to_words, ordinal_words, pause_after, spell_digits, spell_latin_word,
    split_sentences, with_end,
)
from ava.language.pipeline import normalize  # legacy-compatible  # noqa: E402


class TestCharNormalization(unittest.TestCase):
    """AC1.1 — نرمال‌سازی کاراکترها."""

    def test_arabic_to_persian(self):
        self.assertEqual(normalize("كتاب يوسف"), "کتاب یوسف")

    def test_invisible_and_diacritics_removed(self):
        self.assertEqual(normalize("س\u200fلام\u064e ز\u200bیبا\u0650"), "سلام زیبا")

    def test_zwnj_preserved(self):
        self.assertEqual(normalize("می‌روم"), "می\u200cروم")

    def test_persian_digits_read_as_words(self):
        self.assertEqual(normalize("۲۵۰"), "دویست و پنجاه")

    def test_thousands_separator(self):
        self.assertEqual(normalize("1,250,000 ریال"), "یک میلیون و دویست و پنجاه هزار ریال")

    def test_golden_legacy_behavior(self):
        """رفتار مسیر Legacy باید با خروجی‌های ثبت‌شدهٔ قبلی بایت‌به‌بایت یکسان بماند."""
        golden = [
            ("قیمت این کتاب ۲۵۰ هزار تومان است و ۱۲ درصد تخفیف دارد.",
             "قیمت این کتاب دویست و پنجاه هزار تومان است و دوازده درصد تخفیف دارد."),
            ("نسخهٔ 1.5 را می‌خواهم!", "نسخه یک ممیز پنج دهم را می\u200cخواهم!"),
            ("كتاب يوسف را در سال 1403 خواندم.", "کتاب یوسف را در سال هزار و چهارصد و سه خواندم."),
            ("سلام؛\tروزتان خوش!", "سلام؛ روزتان خوش!"),
            ("درصد 50       تخفیف", "درصد پنجاه تخفیف"),
            ("نیم‌فاصله‌ها   می‌آیند", "نیم\u200cفاصله\u200cها می\u200cآیند"),
            ("3.14 عدد پی است", "سه ممیز چهارده صدم عدد پی است"),
            ("«نقل‌قول» و «سؤال»؟", "نقل\u200cقول و سوال؟"),
        ]
        for src, expected in golden:
            self.assertEqual(normalize(src), expected, src)


class TestNumbers(unittest.TestCase):
    """AC1.2 — اعداد."""

    def test_cardinals(self):
        self.assertEqual(number_to_words(0), "صفر")
        self.assertEqual(number_to_words(1403), "هزار و چهارصد و سه")
        self.assertEqual(number_to_words(5000000), "پنج میلیون")
        self.assertEqual(number_to_words(-12), "منفی دوازده")

    def test_ordinals(self):
        self.assertEqual(ordinal_words(1), "اول")
        self.assertEqual(ordinal_words(2), "دوم")
        self.assertEqual(ordinal_words(3), "سوم")
        self.assertEqual(ordinal_words(21), "بیست و یکم")
        self.assertEqual(ordinal_words(33), "سی و سوم")
        self.assertEqual(ordinal_words(30), "سی\u200cام")
        self.assertEqual(ordinal_words(40), "چهلم")

    def test_phone_digit_by_digit(self):
        self.assertEqual(
            spell_digits("0211"), "صفر دو یک یک",
        )

    def test_percent_and_decimal_in_engine(self):
        eng = LanguageEngine()
        out = eng.normalize("12% تخفیف و 0.5 لیتر")
        self.assertIn("دوازده درصد", out)
        self.assertIn("صفر ممیز پنج دهم", out)

    def test_ordinal_marker_engine(self):
        eng = LanguageEngine()
        out = eng.normalize("در ۲ام آبان برگزار شد")
        self.assertIn("دوم", out)
        self.assertNotIn("۲ام", out)

    def test_leading_zero_number_read_digitwise(self):
        out = normalize("کد 00123 ثبت شد")
        self.assertEqual(out, "کد صفر صفر یک دو سه ثبت شد")


class TestDateTime(unittest.TestCase):
    """AC1.3 — تاریخ و ساعت."""

    def test_jalali_slash_date(self):
        eng = LanguageEngine()
        out = eng.normalize("تاریخ 1403/08/01 ثبت شد")
        self.assertIn("اول آبان سال هزار و چهارصد و سه", out)

    def test_gregorian_dash_date(self):
        eng = LanguageEngine()
        out = eng.normalize("رویداد 2026-11-21 ثبت شد")
        self.assertIn("بیست و یکم نوامبر سال دو هزار و بیست و شش", out)

    def test_named_date(self):
        eng = LanguageEngine()
        out = eng.normalize("۱۲ آبان ۱۴۰۳ تعطیل است")
        self.assertIn("دوازدهم آبان هزار و چهارصد و سه", out)
        out2 = eng.normalize("۱۲ آبان سال ۱۴۰۳")
        self.assertIn("دوازدهم آبان سال هزار و چهارصد و سه", out2)

    def test_time(self):
        eng = LanguageEngine()
        self.assertIn("ساعت چهارده و سی دقیقه", eng.normalize("ساعت 14:30 جلسه داریم"))
        self.assertIn("ساعت نه و پنج دقیقه", eng.normalize("09:05"))

    def test_time_specials(self):
        eng = LanguageEngine()
        self.assertIn("نیمه‌شب", eng.normalize("00:00"))
        self.assertIn("ظهر", eng.normalize("12:00"))

    def test_invalid_time_untouched(self):
        out = expand_dates_and_times("25:70")
        self.assertEqual(out, "25:70")

    def test_gregorian_to_jalali_known_dates(self):
        # 2026-11-21 → آبان/آذر ۱۴۰۵ (از تقویم رسمی)؛ رفت و برگشت باید سازگار باشد
        jy, jm, jd = gregorian_to_jalali(2026, 11, 21)
        self.assertEqual((jy, jm, jd), (1405, 8, 30))
        from ava.language.datetime_speech import jalali_to_gregorian
        self.assertEqual(jalali_to_gregorian(jy, jm, jd), (2026, 11, 21))
        # 2025-03-21 == 1404/01/01
        self.assertEqual(gregorian_to_jalali(2025, 3, 21), (1404, 1, 1))


class TestSegmentation(unittest.TestCase):
    """AC1.4 — بخش‌بندی و تقسیم متن طولانی."""

    def test_sentence_split(self):
        self.assertEqual(split_sentences("جمله اول. جمله دوم؟ جمله سوم!"), ["جمله اول.", "جمله دوم؟", "جمله سوم!"])

    def test_decimal_protected(self):
        self.assertEqual(split_sentences("قیمت 3.14 است. بعد آمد"), ["قیمت 3.14 است.", "بعد آمد"])

    def test_long_sentence_falls_back_to_weak_punct(self):
        s = "این جمله بسیار طولانی است، " * 20
        parts = split_sentences(s, max_chars=100)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= 130 for p in parts))

    def test_with_end(self):
        self.assertEqual(with_end("سلام"), "سلام.")
        self.assertEqual(with_end("چطور؟"), "چطور؟")


class TestMixedLanguage(unittest.TestCase):
    """AC1.5 و AC1.6 — متن ترکیبی و تلفظ."""

    def test_detect_primary(self):
        self.assertEqual(detect_primary("سلام دنیا"), "fa")
        self.assertEqual(detect_primary("hello world"), "latin")
        self.assertEqual(detect_primary("از Python استفاده می‌کنیم"), "mixed")

    def test_latin_drop_is_legacy_default(self):
        eng = LanguageEngine()
        self.assertEqual(eng.normalize("کتاب Python خوب است"), "کتاب خوب است")

    def test_latin_spell(self):
        eng = LanguageEngine(LanguageConfig(latin_policy="spell"))
        out = eng.normalize("من Wi-Fi دارم")
        self.assertIn("وای فای", out)  # از واژگان تلفظ

    def test_lexicon_exact_entry(self):
        eng = LanguageEngine()
        self.assertIn("ال ال ام", eng.normalize("LLM چیست؟"))

    def test_custom_lexicon(self):
        lex = PronunciationLexicon()
        lex.add("تهرانپارس", "تهران پارس")
        eng = LanguageEngine(lexicon=lex)
        self.assertIn("تهران پارس", eng.normalize("تهرانپارس بزرگ است"))

    def test_spell_latin_word(self):
        self.assertEqual(spell_latin_word("API"), "اِی پی آی")


class TestProsody(unittest.TestCase):
    def test_pause_table(self):
        self.assertEqual(pause_after("،"), 0.25)
        self.assertEqual(pause_after("."), 0.5)
        self.assertEqual(pause_after("؟"), 0.55)

    def test_duration_estimate_positive(self):
        self.assertGreater(estimate_duration("سلام دنیا. حالت چطور است؟"), 1.0)


class TestPersianScenarios(unittest.TestCase):
    """۱۲ سناریوی فارسی مشخص‌شده در معیارهای پذیرش (بخش متن)."""

    def setUp(self):
        self.eng = LanguageEngine()

    def test_scenarios(self):
        cases = [
            # متن ساده
            ("سلام دنیا", lambda o: o == "سلام دنیا"),
            # نیم‌فاصله
            ("کتاب‌های می‌روند", lambda o: "کتاب\u200cهای" in o and "می\u200cروند" in o),
            # اعداد
            ("250 نسخه", lambda o: "دویست و پنجاه" in o),
            # تاریخ شمسی
            ("1403/01/15", lambda o: "پانزدهم فروردین" in o),
            # ساعت
            ("ساعت 9:15", lambda o: "ساعت نه و پانزده دقیقه" in o),
            # درصد و واحدها
            ("45% از 5 kg", lambda o: "چهل و پنج درصد" in o and "پنج کیلوگرم" in o),
            # ترکیبی
            ("گزارش PDF آماده است", lambda o: "پی دی اف" in o),
            # نام‌ها
            ("دکتر مریمی گفت", lambda o: "دکتر مریمی گفت" in o),
            # پرسشی
            ("حالت چطور است؟", lambda o: o.endswith("است؟")),
            # احساسی
            ("چه خبر فوق‌العاده!", lambda o: o.endswith("فوق\u200cالعاده!")),
            # طولانی
            ("جمله، " * 60, lambda o: len(o) > 0),
            # علائم متعدد
            ("یک، دو؛ سه: چهار. پنج!", lambda o: all(p in o for p in "،؛:.")),
        ]
        for text, check in cases:
            out = self.eng.normalize(text)
            self.assertTrue(check(out), f"{text!r} -> {out!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
