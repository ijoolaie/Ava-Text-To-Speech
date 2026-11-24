# -*- coding: utf-8 -*-
"""تست‌های متمرکز بک‌اند Ali Fathi Persian TTS (pretrained ONNX — CC-BY-4.0).

فقط تست‌های هدفمند اینتگریشن: ثبت/بارگذاری بک‌اند، خروجی فرانت phoneme،
سازگاری I/O مدل ONNX، هر ۷ گوینده، خروجی WAV با 22050Hz، اعتبارسنجی گزینه‌ها
(speed / sample_rate)، رفتار علائم نگارشی فارسی و انگلیسی، و یکپارچگی با
موتور موجود. تست‌های unrelated اجرا نمی‌شوند.

مدل باید قبلاً به‌صورت محلی در `<repo>/alifathi/` قرار گرفته باشد
(یا متغیر محیطی AVA_ALIFATHI_MODEL_DIR) — در زمان تست هیچ دسترسی شبکه‌ای وجود ندارد.
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.runtime import (  # noqa: E402
    AliFathiTTSBackend, BackendRegistry, SynthesisEngine, SynthesisOptions,
)
from ava.runtime.alifathi import (  # noqa: E402
    BACKEND_FORMAT, MODEL_DIR_ENV, MODEL_HF_URL, default_model_dir,
    validate_sample_rate, validate_speed,
    EspeakNGPhonemizer, ModelUnavailable, PhonemizerUnavailable,
)


def _model_dir() -> str | None:
    cand = default_model_dir()
    return str(cand) if cand.is_dir() else None


MODEL_DIR = _model_dir()
ONNX_READY = bool(MODEL_DIR and os.path.isfile(os.path.join(MODEL_DIR, "model.onnx")))
CALIB_READY = bool(MODEL_DIR and os.path.isfile(os.path.join(MODEL_DIR, "calib_input.npy")))
ESPEAK_READY = shutil.which("espeak-ng") is not None


class TestBackendRegistrationAndLoading(unittest.TestCase):
    """Backend registration and model loading (fail-closed error messages)."""

    @unittest.skipUnless(ONNX_READY, "local ONNX model not staged in alifathi/")
    def test_registry_create_and_engine_load(self):
        reg = BackendRegistry()
        reg.register(BACKEND_FORMAT, AliFathiTTSBackend)
        model = reg.create(BACKEND_FORMAT, model_dir=MODEL_DIR)
        info = model.info()
        self.assertEqual(info["format"], BACKEND_FORMAT)
        self.assertEqual(info["sample_rate"], 22050)
        self.assertTrue(info["commercial"])
        # registration through the existing engine imports the 7 voices
        engine = SynthesisEngine()
        engine.register_backend(BACKEND_FORMAT, AliFathiTTSBackend)
        engine.load_model("alifathi", BACKEND_FORMAT, model_dir=MODEL_DIR)
        self.assertEqual(len(engine.list_voices()), 7)
        self.assertEqual({v.backend for v in engine.list_voices()}, {BACKEND_FORMAT})

    def test_default_model_dir_is_repo_alifathi(self):
        """بدون env، مسیر پیش‌فرض `alifathi/` ریشهٔ مخزن است (بارگذاری فقط محلی)."""
        self.assertEqual(str(default_model_dir()), os.path.join(ROOT, "alifathi"))
        with_env = dict(os.environ)
        os.environ[MODEL_DIR_ENV] = os.path.join(tempfile.gettempdir(), "whatever")
        try:
            self.assertEqual(str(default_model_dir()), os.path.join(tempfile.gettempdir(), "whatever"))
        finally:
            os.environ.clear(); os.environ.update(with_env)

    def test_missing_model_dir_fails_closed(self):
        with self.assertRaises(ModelUnavailable) as ctx:
            AliFathiTTSBackend(model_dir=os.path.join(tempfile.mkdtemp(), "missing"))
        msg = str(ctx.exception)
        self.assertIn("not found", msg)
        self.assertIn(MODEL_HF_URL, msg)          # where to get the model
        self.assertIn(MODEL_DIR_ENV, msg)         # how to configure the local dir
        self.assertIn("No network access", msg)   # runtime stays offline

    def test_incomplete_model_dir_fails_closed(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "README"), "w") as f:
            f.write("not a model")
        with self.assertRaises(ModelUnavailable) as ctx:
            AliFathiTTSBackend(model_dir=tmp)
        self.assertIn("incomplete", str(ctx.exception))

    def test_missing_onnxruntime_reported(self):
        # lazy import path: simulate a missing onnxruntime module
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *a, **kw)

        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "model.onnx"), "wb") as f:
            f.write(b"placeholder")
        with open(os.path.join(tmp, "model.onnx.json"), "w", encoding="utf-8") as f:
            f.write("{}")  # valid JSON so loading reaches the onnxruntime import
        builtins.__import__, saved = fake_import, builtins.__import__
        try:
            with self.assertRaises(ModelUnavailable) as ctx:
                AliFathiTTSBackend(model_dir=tmp)
            self.assertIn("pip install onnxruntime", str(ctx.exception))
        finally:
            builtins.__import__ = saved


class TestOptionValidation(unittest.TestCase):
    """ورودی نامعتبر باید fail-closed شود، نه اینکه خروجی خراب بسازد."""

    def test_speed_validation(self):
        self.assertEqual(validate_speed(1.0), 1.0)
        self.assertEqual(validate_speed(0.5), 0.5)
        # bool must be rejected explicitly (True would silently coerce to 1.0)
        for bad in (0, -0.5, float("nan"), float("inf"), 99.0, "abc", None, True, False):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_speed(bad)

    def test_sample_rate_validation(self):
        self.assertIsNone(validate_sample_rate(None))
        self.assertEqual(validate_sample_rate(16000), 16000)
        self.assertEqual(validate_sample_rate(16000.0), 16000)  # integral float ok
        # bools, strings, fractional values and NaN/inf must all be rejected
        for bad in (0, -22050, 7999, 192001, "abc", "16000", 16000.5, 1e10,
                    float("nan"), float("inf"), True, False):
            with self.assertRaises(ValueError, msg=repr(bad)):
                validate_sample_rate(bad)


class TestPhonemeFrontend(unittest.TestCase):
    """Phoneme frontend output (clean-room espeak-ng glue + Piper table)."""

    def test_encode_piper_convention_structure(self):
        pim = {"_": [0], "^": [1], "$": [2], "a": [3], "b": [4]}
        enc = EspeakNGPhonemizer()
        # [BOS, PAD] + (id, PAD) per char + [EOS]
        self.assertEqual(enc.encode("ab", pim), [1, 0, 3, 0, 4, 0, 2])
        # unmapped chars are skipped
        self.assertEqual(enc.encode("azb", pim), [1, 0, 3, 0, 4, 0, 2])
        # missing control symbol → fail closed with a clear reason
        with self.assertRaises(ModelUnavailable):
            enc.encode("ab", {"a": [3]})

    def test_hash_and_dollar_punctuation_behavior(self):
        """`#` مثل علامت عادی نگه داشته و map می‌شود؛ `$` هرگز EOS میانی تزریق نمی‌کند."""
        pim = {"_": [0], "^": [1], "$": [2], "#": [3], "a": [4], "b": [5]}
        enc = EspeakNGPhonemizer()
        # '#' is a normal table symbol → mapped like any other mark
        self.assertEqual(enc.encode("a#b", pim), [1, 0, 4, 0, 3, 0, 5, 0, 2])
        # literal '$' mid-stream is skipped; EOS appears exactly once, at the end
        self.assertEqual(enc.encode("a$b", pim), [1, 0, 4, 0, 5, 0, 2])
        self.assertEqual(enc.encode("a$$b", pim), [1, 0, 4, 0, 5, 0, 2])
        # literal '^' (BOS) mid-stream is skipped for the same reason
        self.assertEqual(enc.encode("a^b", pim), [1, 0, 4, 0, 5, 0, 2])

    @unittest.skipUnless(ESPEAK_READY, "espeak-ng not installed locally (optional external component)")
    def test_hash_and_dollar_runs_survive_phonemization(self):
        """run علائم متصل به کلمه، `#` و `$` را به‌صورت literal نگه می‌دارد."""
        enc = EspeakNGPhonemizer()
        self.assertTrue(enc.phonemize("تست#.").endswith("#."))
        self.assertTrue(enc.phonemize("قیمت$").endswith("$"))

    @unittest.skipUnless(ESPEAK_READY, "espeak-ng not installed locally (optional external component)")
    def test_phonemize_keeps_punctuation_and_strips_padding(self):
        enc = EspeakNGPhonemizer()
        ipa = enc.phonemize("سلام.")
        self.assertTrue(ipa.endswith("."))
        self.assertNotIn("_", ipa)  # padding never leaks into the phoneme string
        self.assertTrue(ipa.strip())

    @unittest.skipUnless(ESPEAK_READY, "espeak-ng not installed locally (optional external component)")
    def test_ascii_punctuation_preserved(self):
        """علامت‌های ASCII (اعضای جدول مدل) در جریان phoneme حفظ می‌شوند."""
        enc = EspeakNGPhonemizer()
        ipa = enc.phonemize("Hello, world! How are you?")
        for mark in (",", "!", "?"):
            self.assertIn(mark, ipa, f"ASCII mark {mark!r} must survive phonemization")

    @unittest.skipUnless(ESPEAK_READY, "espeak-ng not installed locally (optional external component)")
    def test_persian_punctuation_splits_clauses_and_drops_marks(self):
        """علامت‌های فارسی (؟ ، ؛) مرز بند می‌سازند اما نماد مدل ندارند → drop."""
        enc = EspeakNGPhonemizer()
        ipa = enc.phonemize("سلام، حالتان چطور است؟")
        self.assertNotIn("،", ipa)  # Persian comma has no model symbol
        self.assertNotIn("؟", ipa)  # Persian question mark has no model symbol
        self.assertTrue(ipa.strip())  # but the words themselves are phonemized
        # mixed run: only the ASCII member of a punctuation run survives
        mixed = enc.phonemize("سلام؟.")
        self.assertIn(".", mixed)
        self.assertNotIn("؟", mixed)

    def test_espeak_missing_message(self):
        """The phonemizer is an OPTIONAL EXTERNAL component — clear failure."""
        enc = EspeakNGPhonemizer()
        saved = os.environ.get("PATH")
        os.environ["PATH"] = ""
        try:
            with self.assertRaises(PhonemizerUnavailable) as ctx:
                enc.phonemize("سلام")
            msg = str(ctx.exception)
            self.assertIn("espeak-ng", msg)
            self.assertIn("NOT bundled", msg)   # GPL component is never shipped
            self.assertIn("GPL-3.0-or-later", msg)
        finally:
            if saved is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved

    @unittest.skipUnless(CALIB_READY, "official calibration files not staged in alifathi/")
    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_frontend_matches_official_calibration(self):
        """Clean-room frontend reproduces the official tf/calib input EXACTLY."""
        import json
        inp = np.load(os.path.join(MODEL_DIR, "calib_input.npy"))
        ilen = int(np.load(os.path.join(MODEL_DIR, "calib_input_lengths.npy"))[0])
        with open(os.path.join(MODEL_DIR, "model.onnx.json"), encoding="utf-8") as f:
            pim = json.load(f)["phoneme_id_map"]
        enc = EspeakNGPhonemizer()
        # the calibration phrase (verified against tf/calib/input.npy)
        got = enc.encode(enc.phonemize("این یک جمله آزمایشی برای تبدیل مدل است."), pim)
        self.assertEqual(got, inp[0, :ilen].tolist())

    @unittest.skipUnless(ONNX_READY, "local ONNX model not staged in alifathi/")
    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_ids_inside_symbol_table(self):
        import json
        with open(os.path.join(MODEL_DIR, "model.onnx.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        pim, num_symbols = cfg["phoneme_id_map"], cfg["num_symbols"]
        enc = EspeakNGPhonemizer()
        ids = enc.encode(enc.phonemize("سلام دنیا"), pim)
        self.assertEqual(ids[0], pim["^"][0])
        self.assertEqual(ids[-1], pim["$"][0])
        self.assertTrue(all(0 <= i < num_symbols for i in ids))
        self.assertGreater(len(ids), 8)

    @unittest.skipUnless(ONNX_READY, "local ONNX model not staged in alifathi/")
    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_hash_and_dollar_on_real_model_table(self):
        """رفتار `#` و `$` روی جدول واقعی مدل: # دارای ID، $ میانی بدون اثر."""
        import json
        with open(os.path.join(MODEL_DIR, "model.onnx.json"), encoding="utf-8") as f:
            pim = json.load(f)["phoneme_id_map"]
        enc = EspeakNGPhonemizer()
        # mid-stream '$' contributes nothing vs. the same text without it
        with_dollar = enc.encode("س$لام", pim)
        self.assertEqual(with_dollar, enc.encode("سلام", pim))
        # trailing '#' keeps its table id inside a valid full sequence
        with_hash = enc.encode(enc.phonemize("سلام#"), pim)
        self.assertIn(pim["#"][0], with_hash)
        self.assertEqual(with_hash[0], pim["^"][0])
        self.assertEqual(with_hash[-1], pim["$"][0])  # exactly one EOS, at the end
        self.assertNotIn(pim["$"][0], with_hash[:-1])  # no mid-stream EOS


@unittest.skipUnless(ONNX_READY, "local ONNX model not staged in alifathi/")
class TestOnnxInferenceAndEngineIntegration(unittest.TestCase):
    """ONNX I/O compatibility, all 7 speakers, 22050 Hz WAV, engine integration."""

    @classmethod
    def setUpClass(cls):
        cls.engine = SynthesisEngine()
        cls.engine.register_backend(BACKEND_FORMAT, AliFathiTTSBackend)
        cls.model = cls.engine.load_model("alifathi", BACKEND_FORMAT)  # default dir = alifathi/

    def test_onnx_io_names_and_shapes(self):
        names = sorted(i.name for i in self.model.session.get_inputs())
        self.assertEqual(names, ["input", "input_lengths", "scales", "sid"])
        types = {i.name: i.type for i in self.model.session.get_inputs()}
        self.assertEqual(types["input"], "tensor(int64)")
        self.assertEqual(types["scales"], "tensor(float)")
        (out,) = self.model.session.get_outputs()
        self.assertEqual(out.name, "output")
        self.assertEqual(out.type, "tensor(float)")
        self.assertEqual(len(out.shape), 3)  # [batch, 1, time]

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_seven_speakers_with_exact_mapping(self):
        voices = self.model.list_voices()
        self.assertEqual(len(voices), 7)
        expected = sorted(self.model.cfg["speaker_id_map"].items(), key=lambda kv: int(kv[1]))
        self.assertEqual([(v.voice_id, int(v.ref)) for v in voices],
                         [(f"alifathi-{name}", int(sid)) for name, sid in expected])
        for v in voices:
            self.assertTrue(v.commercial_ok, "CC-BY-4.0 weights are commercial")
            self.assertIn("CC-BY-4.0", v.license)
        # every speaker produces real audio through the engine
        for v in voices:
            r = self.engine.synthesize("سلام", v.voice_id)
            self.assertGreater(r.duration_s, 0.05, v.voice_id)
            self.assertTrue(np.isfinite(r.audio).all())

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_wav_output_at_22050(self):
        r = self.engine.synthesize("سلام دنیا.", "alifathi-fleurs")
        self.assertEqual(r.sample_rate, 22050)
        wav_path = os.path.join(tempfile.mkdtemp(), "alifathi.wav")
        sf.write(wav_path, r.audio, r.sample_rate)
        info = sf.info(wav_path)
        self.assertEqual(info.samplerate, 22050)
        self.assertEqual(info.channels, 1)
        self.assertGreater(info.frames, 0)
        self.assertTrue(wav_bytes_header(r).startswith(b"RIFF"))

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_sample_rate_override_resamples(self):
        native = self.engine.synthesize("سلام", "alifathi-fleurs")
        forced = self.engine.synthesize("سلام", "alifathi-fleurs",
                                        SynthesisOptions(sample_rate=16000))
        self.assertEqual(forced.sample_rate, 16000)
        self.assertAlmostEqual(forced.duration_s, native.duration_s, delta=native.duration_s * 0.25)

    def test_invalid_options_fail_closed(self):
        """گزینهٔ نامعتبر نباید بی‌صدا خروجی خراب بسازد (بولی/اعشاری/رشته هم)."""
        for opts in (SynthesisOptions(speed=0), SynthesisOptions(speed=-0.5),
                     SynthesisOptions(speed=float("nan")), SynthesisOptions(speed=99.0),
                     SynthesisOptions(speed=True), SynthesisOptions(speed=False),
                     SynthesisOptions(sample_rate=0), SynthesisOptions(sample_rate=7999),
                     SynthesisOptions(sample_rate=192001), SynthesisOptions(sample_rate=16000.5),
                     SynthesisOptions(sample_rate=True), SynthesisOptions(sample_rate="16000")):
            with self.assertRaises(ValueError, msg=str(opts)):
                self.engine.synthesize("سلام", "alifathi-fleurs", opts)

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_speed_option_scales_length(self):
        normal = self.engine.synthesize("سلام دنیا.", "alifathi-mana")
        slow = self.engine.synthesize("سلام دنیا.", "alifathi-mana", SynthesisOptions(speed=0.5))
        self.assertGreater(slow.duration_s, normal.duration_s * 1.3)

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_persian_punctuation_sentence_synthesizes(self):
        """جملهٔ کامل با علائم فارسی و انگلیسی از موتور صدا تولید می‌کند."""
        r = self.engine.synthesize("سلام، حالتان چطور است؟ سریع جواب بده.", "alifathi-gptinformal")
        self.assertGreater(r.duration_s, 0.5)
        self.assertTrue(np.isfinite(r.audio).all())

    @unittest.skipUnless(ESPEAK_READY, "eSpeak NG not installed locally (optional external component)")
    def test_engine_stream_per_sentence(self):
        chunks = list(self.engine.synthesize_stream("سلام. حالتان چطور است؟", "alifathi-psd_M04"))
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(c.duration_s > 0.05 for c in chunks))

    def test_runtime_info_discloses_frontend_and_license(self):
        info = self.engine.get_model_info()[0]
        self.assertEqual(info["model_id"], "alifathi")
        self.assertIn("espeak-ng", info["frontend"])
        self.assertIn("CC-BY-4.0", info["license"])
        self.assertEqual(info["speakers"], 7)


def wav_bytes_header(result):
    """16-bit PCM WAV header check without importing the private helper twice."""
    import io
    buf = io.BytesIO()
    sf.write(buf, result.audio, result.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()[:4]


if __name__ == "__main__":
    unittest.main(verbosity=2)
