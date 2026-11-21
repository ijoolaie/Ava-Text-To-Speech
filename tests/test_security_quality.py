# -*- coding: utf-8 -*-
"""فاز ۹ — تست‌های امنیت و کیفیت (معیارهای AC9.1) + سناریوهای فارسی."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.language import LanguageEngine  # noqa: E402
from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer  # noqa: E402
from training.train_native import export_native_bundle  # noqa: E402


def make_bundle(out_dir: str) -> str:
    cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,
                          ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256,
                          max_frames=128, n_speakers=1)
    tok = CharTokenizer()
    model = AVANativeTTS(cfg, len(tok))
    return export_native_bundle(model, tok, cfg, out_dir)


class TestSecurity(unittest.TestCase):
    """AC9.1 — امنیت ورودی‌ها."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))
        from ava.runtime import NativeN1Backend, SynthesisEngine
        cls.engine = SynthesisEngine()
        cls.engine.register_backend("ava-n1", NativeN1Backend)
        cls.engine.load_model("n1", "ava-n1", bundle_dir=cls.bundle)

    def test_empty_and_whitespace_text_rejected(self):
        for t in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                self.engine.synthesize(t, "ava-n1-ava")

    def test_huge_text_is_bounded_by_chunking_not_memory_blowup(self):
        """متن طولانی نباید بی‌رویه حافظه مصرف کند — جمله‌بندی محدودکننده است."""
        result = self.engine.synthesize("سلام. " * 40, "ava-n1-ava")
        self.assertGreater(result.duration_s, 0)

    def test_control_characters_are_stripped(self):
        eng = LanguageEngine()
        out = eng.normalize("سلام\u0000\u0007دنیا\u202e  ok")
        self.assertNotIn("\u0000", out)
        self.assertNotIn("\u0007", out)
        self.assertNotIn("\u202e", out)

    def test_path_traversal_in_voice_id_is_just_a_lookup_miss(self):
        with self.assertRaises(KeyError):
            self.engine.synthesize("سلام", "../../etc/passwd")

    def test_bundle_path_is_not_user_controlled_in_api(self):
        """در API تنها منبع بارگذاری، متغیر محیطی است؛ کاربر نمی‌تواند مسیر بدهد."""
        import inspect
        from ava.api import server
        src = inspect.getsource(server)
        self.assertNotIn("bundle_dir=req", src)

    def test_no_secrets_in_repo(self):
        """بررسی ساده: هیچ توکن/کلیدی در فایل‌های کد کامیت‌شده نیست."""
        hit = os.popen(
            "git grep -l -iE 'api[_-]?key|secret|password|token=' -- '*.py' '*.yml' '*.yaml' || true"
        ).read().strip().splitlines()
        allowed_prefixes = ("tests/",)
        bad = [h for h in hit if not h.startswith(allowed_prefixes) and "vendor" not in h]
        self.assertEqual(bad, [], f"possible secrets in: {bad}")

    def test_speed_option_bounded(self):
        r = self.engine.synthesize("سلام", "ava-n1-ava", type("O", (), {"speed": 0.01, "sample_rate": None, "speaker_id": 0, "stop_event": None, "pause_scale": 1.0})())
        self.assertGreater(r.duration_s, 0)  # model floors the speed internally


class TestPersianQualityScenarios(unittest.TestCase):
    """سناریوهای کیفی فارسی (متن‌محور) — صدای نهایی نیازمند مدل آموزش‌دیده است."""

    @classmethod
    def setUpClass(cls):
        cls.eng = LanguageEngine()

    def test_all_12_scenarios_normalize_cleanly(self):
        scenarios = [
            "سلام دنیا",                      # ساده
            "کتاب‌های بیشتری می‌خرم",          # نیم‌فاصله
            "۲۵۰ نسخه فروخته شد",             # اعداد
            "امروز ۱۴۰۳/۰۸/۰۱ است",           # تاریخ شمسی
            "ساعت 14:30 قرار گذاشتیم",        # ساعت
            "۴۵٪ تخفیف روی 2 kg سیب",         # درصد و واحدها
            "از Python برای داده استفاده می‌کنیم",  # ترکیبی
            "دکتر مریمی به شیراز رفت",        # نام اشخاص/مکان
            "وضعیت چطور است؟",                # پرسشی
            "چه شگفت‌انگیز!",                  # احساسی
            "جمله، " * 40,                     # طولانی
            "یک، دو؛ سه: چهار؟ پنج!",          # علائم متعدد
        ]
        allowed = set("آاءأإئؤپبتثجچحخدذرزژسشصضطظعغفقکگلمنوهی،؛:.!?؟ \u200c")
        for s in scenarios:
            out = self.eng.normalize(s)
            self.assertTrue(out and len(out) > 0, s)
            for ch in out:
                self.assertIn(ch, allowed, f"unexpected char {ch!r} in {out!r}")

    def test_regression_snapshot(self):
        """رجسیون: خروجی چند جمله به‌صورت snapshot قفل می‌شود."""
        cases = {
            "قیمت ۲۵۰ تومان": "قیمت دویست و پنجاه تومان",
            "45% تخفیف": "چهل و پنج درصد تخفیف",
            "ساعت 9:00 صبح": "ساعت نه صبح",
        }
        for src, expected in cases.items():
            self.assertEqual(self.eng.normalize(src), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
