# -*- coding: utf-8 -*-
"""فاز ۶ — تست‌های سیستم صداها (معیارهای AC6.1 و AC6.2)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.runtime import Voice  # noqa: E402
from ava.voices import VoiceProfile, VoiceStore, check_voice_prompt  # noqa: E402


class TestVoiceProfile(unittest.TestCase):
    def test_json_roundtrip(self):
        p = VoiceProfile(voice_id="sara", display_name="سارا", gender="female",
                         backend="ava-n1", ref="0", license="own-recording",
                         commercial_ok=True, requires_consent=True, consent_ref="RC-2026-01",
                         quality={"snr_db": 31.2}, tags=["primary"])
        text = p.to_json()
        p2 = VoiceProfile.from_json(text)
        self.assertEqual(p2.voice_id, "sara")
        self.assertEqual(p2.consent_ref, "RC-2026-01")

    def test_store_save_load_all_delete(self):
        store = VoiceStore(tempfile.mkdtemp())
        p = VoiceProfile(voice_id="ava", display_name="آوا", backend="ava-n1", ref="0", commercial_ok=True)
        store.save(p)
        self.assertEqual(store.load("ava").voice_id, "ava")
        self.assertEqual(len(store.all()), 1)
        store.delete("ava")
        self.assertEqual(store.all(), [])

    def test_cloning_without_consent_refused(self):
        """Voice cloning بدون رضایت ممنوع است."""
        store = VoiceStore(tempfile.mkdtemp())
        p = VoiceProfile(voice_id="clone", display_name="کلون", requires_consent=True, consent_ref="")
        with self.assertRaises(ValueError):
            store.save(p)

    def test_to_runtime_voice_fields(self):
        p = VoiceProfile(voice_id="ava", display_name="آوا", backend="ava-n1", ref="0",
                         commercial_ok=True, license="MIT")
        v = Voice(voice_id=p.voice_id, display_name=p.display_name, language=p.language,
                  backend=p.backend, ref=p.ref, license=p.license, commercial_ok=p.commercial_ok)
        self.assertTrue(v.commercial_ok)


class TestVoiceQuality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_good_prompt_passes(self):
        sr = 24000
        bursts, gap = [], np.zeros(int(0.4 * sr), dtype=np.float32)
        for f in (180, 220, 190, 240):
            n = int(1.2 * sr)
            t = np.arange(n) / sr
            env = np.minimum(1, t * 40) * np.minimum(1, np.maximum(1.2 - t, 0) * 8)
            bursts.append(0.5 * env * np.sin(2 * np.pi * f * t))
            bursts.append(gap)
        wav = np.concatenate(bursts)
        path = os.path.join(self.tmp, "good.wav")
        sf.write(path, wav, sr)
        rep = check_voice_prompt(path)
        self.assertTrue(rep["ok"], rep["reasons"])
        self.assertEqual(rep["sample_rate"], sr)

    def test_too_short_prompt_rejected(self):
        path = os.path.join(self.tmp, "short.wav")
        sf.write(path, 0.5 * np.sin(2 * np.pi * 200 * np.arange(int(1.0 * 24000)) / 24000), 24000)
        rep = check_voice_prompt(path)
        self.assertFalse(rep["ok"])
        self.assertTrue(any(r.startswith("duration") for r in rep["reasons"]))

    def test_quiet_prompt_rejected(self):
        sr = 24000
        path = os.path.join(self.tmp, "quiet.wav")
        t = np.arange(int(4 * sr)) / sr
        env = np.minimum(1, t % 1.0 * 8)
        sf.write(path, 0.003 * env * np.sin(2 * np.pi * 200 * t), sr)
        rep = check_voice_prompt(path)
        self.assertFalse(rep["ok"])
        self.assertIn("too_quiet", rep["reasons"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
