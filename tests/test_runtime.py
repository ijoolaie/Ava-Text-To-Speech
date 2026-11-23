# -*- coding: utf-8 -*-
"""فاز ۵ — تست‌های Runtime (معیارهای AC5.1 تا AC5.3)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer  # noqa: E402
from ava.runtime import (  # noqa: E402
    BackendRegistry, NativeN1Backend, SynthesisCancelled, SynthesisEngine,
    SynthesisOptions, wav_bytes,
)
from training.train_native import export_native_bundle  # noqa: E402

WEIGHTS_READY = os.path.isfile(os.path.join(ROOT, "config", "fa.yaml")) and \
    os.path.isdir(os.path.join(ROOT, "weights", "voices"))


def make_bundle(out_dir: str) -> str:
    """Untrained tiny bundle — runtime mechanics don't need quality."""
    cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,
                          ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256,
                          max_frames=256, n_speakers=1)
    tok = CharTokenizer()
    model = AVANativeTTS(cfg, len(tok))
    return export_native_bundle(model, tok, cfg, out_dir)


class TestSynthesisEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))
        cls.engine = SynthesisEngine()
        cls.engine.register_backend("ava-n1", NativeN1Backend)
        cls.engine.load_model("n1", "ava-n1", bundle_dir=cls.bundle)

    def test_load_and_list_voices(self):
        voices = self.engine.list_voices()
        self.assertTrue(any(v.voice_id == "ava-n1-ava" for v in voices))
        # صدای Native آزمایشی است: وزن آموزش‌ندیده، مجوز نهایی تعیین‌نشده → تجاری نیست
        self.assertFalse(any(v.commercial_ok for v in voices))
        self.assertIn("آزمایشی", voices[0].license)

    def test_synthesize_real_wav(self):
        """AC5.2 — خروجی WAV واقعی با SR و کانال مشخص."""
        result = self.engine.synthesize("سلام دنیا. حالتان چطور است؟", "ava-n1-ava")
        self.assertGreater(result.duration_s, 0)
        self.assertEqual(result.sample_rate, 24000)
        wav_path = os.path.join(self.tmp, "out.wav")
        sf.write(wav_path, result.audio, result.sample_rate)
        info = sf.info(wav_path)
        self.assertEqual(info.samplerate, 24000)
        self.assertEqual(info.channels, 1)
        self.assertGreater(info.frames, 0)

    def test_stream_yields_per_sentence(self):
        chunks = list(self.engine.synthesize_stream("جمله اول. جمله دوم؟", "ava-n1-ava"))
        self.assertEqual(len(chunks), 2)

    def test_batch(self):
        results = self.engine.synthesize_batch(["الف", "ب"], "ava-n1-ava")
        self.assertEqual(len(results), 2)

    def test_stop_cancels_and_reset_restores(self):
        self.engine.stop()
        with self.assertRaises(SynthesisCancelled):
            self.engine.synthesize("جمله اول. جمله دوم.", "ava-n1-ava")
        self.engine.reset_stop()
        r = self.engine.synthesize("جمله اول.", "ava-n1-ava")
        self.assertGreater(r.duration_s, 0)

    def test_per_request_stop_event(self):
        import threading
        ev = threading.Event()
        ev.set()
        with self.assertRaises(SynthesisCancelled):
            self.engine.synthesize("جمله اول. جمله دوم.", "ava-n1-ava", SynthesisOptions(stop_event=ev))
        # engine-level flag untouched — other requests unaffected
        r = self.engine.synthesize("جمله اول.", "ava-n1-ava")
        self.assertGreater(r.duration_s, 0)

    def test_unknown_voice_and_empty_text(self):
        with self.assertRaises(KeyError):
            self.engine.synthesize("سلام", "ناموجود")
        with self.assertRaises(ValueError):
            self.engine.synthesize("  ", "ava-n1-ava")

    def test_unload_model(self):
        engine = SynthesisEngine()
        engine.register_backend("ava-n1", NativeN1Backend)
        engine.load_model("x", "ava-n1", bundle_dir=self.bundle)
        self.assertTrue(engine.list_voices())
        engine.unload_model("x")
        self.assertEqual(engine.list_voices(), [])
        self.assertEqual(engine.get_model_info(), [])
        with self.assertRaises(KeyError):
            engine.unload_model("x")

    def test_registry_unknown_format(self):
        with self.assertRaises(KeyError):
            self.engine.registry.create("no-such-format")

    def test_wav_bytes_header(self):
        result = self.engine.synthesize("تست", "ava-n1-ava")
        data = wav_bytes(result)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertGreater(len(data), 1000)

    def test_multi_sentence_concat(self):
        one = self.engine.synthesize("سلام.", "ava-n1-ava")
        two = self.engine.synthesize("سلام. سلام.", "ava-n1-ava")
        self.assertGreater(two.duration_s, one.duration_s)


@unittest.skipUnless(WEIGHTS_READY, "legacy weights/config not present (commercial-clean checkout)")
class TestLegacyAdapter(unittest.TestCase):
    """آداپتور اختیاری Legacy — صدای واقعی فارسی برای مقایسه؛ غیرتجاری."""

    @classmethod
    def setUpClass(cls):
        if not WEIGHTS_READY:
            raise unittest.SkipTest("no weights")
        from ava.runtime.backends import LegacyPocketTTSBackend
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))
        cls.engine = SynthesisEngine()
        cls.engine.register_backend("legacy-pocket-tts", LegacyPocketTTSBackend)
        cls.engine.load_model("legacy", "legacy-pocket-tts", config_path=os.path.join(ROOT, "config", "fa.yaml"))

    def test_legacy_voices_marked_non_commercial(self):
        voices = self.engine.list_voices()
        self.assertEqual(len(voices), 3)
        self.assertFalse(all(v.commercial_ok for v in voices))
        self.assertTrue(all(v.license.startswith("CC BY-NC") for v in voices))

    def test_legacy_real_persian_audio(self):
        result = self.engine.synthesize("سلام دنیا", "legacy-shirin")
        self.assertGreater(result.duration_s, 0.5)
        wav_path = os.path.join(self.tmp, "legacy.wav")
        sf.write(wav_path, result.audio, result.sample_rate)
        self.assertGreater(os.path.getsize(wav_path), 10000)

    def test_engine_holds_both_backends(self):
        """هم‌زمانی دو معماری متفاوت در یک runtime — اثبات استقلال از معماری."""
        from ava.runtime.backends import LegacyPocketTTSBackend
        engine = SynthesisEngine()
        engine.register_backend("ava-n1", NativeN1Backend)
        engine.register_backend("legacy-pocket-tts", LegacyPocketTTSBackend)
        engine.load_model("n1", "ava-n1", bundle_dir=self.bundle)
        engine.load_model("legacy", "legacy-pocket-tts", config_path=os.path.join(ROOT, "config", "fa.yaml"))
        backs = {v.backend for v in engine.list_voices()}
        self.assertEqual(backs, {"ava-n1", "legacy-pocket-tts"})
        r1 = engine.synthesize("سلام", "ava-n1-ava")
        r2 = engine.synthesize("سلام", "legacy-shirin")
        self.assertGreater(r1.duration_s, 0)
        self.assertGreater(r2.duration_s, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
