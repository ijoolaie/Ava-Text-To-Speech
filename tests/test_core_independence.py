# -*- coding: utf-8 -*-
"""فاز ۱ — تست‌های استقلال هستهٔ Native از Legacy (Pocket-TTS) و قراردادهای مرزی.

پوشش الزامات §۶ فاز ۱:
  1) import هستهٔ AVA Native بدون Legacy
  2) قرارداد Text Frontend فارسی
  3) قرارداد ورودی/خروجی مدل
  4) مسیر inference با وزن آزمایشی داخلی مجاز (fixture خودِ پروژه)
  5) Runtime بدون وابستگی Legacy
  6) API و CLI موجود
  7) عدم import اجباری Pocket-TTS در مسیر Native

تولید صدای واقعی محصول ادعا نمی‌شود؛ وزن‌ها fixture آموزش‌ندیدهٔ داخلی‌اند.
"""
import ast
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ماژول‌های اقتباسی/Legacy که نباید در مسیر Native بار شوند
FORBIDDEN = [
    "ava.modules",
    "ava.models.tts_model",
    "ava.models.flow_lm",
    "ava.models.mimi",
    "ava.models.text_chunking",
    "ava.models.model_state",
    "ava.utils.weights_loading",
    "ava.data",
]

BLOCKER_PRELUDE = (
    "import sys\n"
    "for _name in %r:\n"
    "    sys.modules[_name] = None\n" % (FORBIDDEN,)
)

NATIVE_PATH_FILES = [
    os.path.join(ROOT, "ava", "__init__.py"),
    os.path.join(ROOT, "ava", "cli.py"),
    os.path.join(ROOT, "ava", "text_norm.py"),
] + [
    os.path.join(ROOT, "ava", sub, f)
    for sub, files in [
        ("language", ["__init__.py", "chars.py", "numbers.py", "datetime_speech.py",
                      "language_detect.py", "pronunciation.py", "segmentation.py",
                      "prosody.py", "pipeline.py", "token_counter.py"]),
        ("models/native", ["__init__.py", "model.py", "config.py", "tokenizer.py"]),
        ("audio", ["__init__.py", "mel.py", "vocoder.py"]),
        ("runtime", ["engine.py"]),
        ("voices", ["__init__.py", "registry.py", "quality.py"]),
        ("api", ["server.py"]),
        ("utils", ["weights_io.py", "paths.py"]),
    ]
    for f in files
]


def existing_native_files():
    return [p for p in NATIVE_PATH_FILES if os.path.isfile(p)]


class TestNoMandatoryLegacyImport(unittest.TestCase):
    """§۶-۷ — هیچ import اجباری از کد اقتباسی pocket-tts در مسیر Native."""

    def test_module_level_imports_of_native_path_are_clean(self):
        """AST scan: importهای سطح ماژول در مسیر Native نباید به Legacy اشاره کنند."""
        bad_tokens = ("ava.modules", "tts_model", "flow_lm", "mimi",
                      "text_chunking", "model_state", "weights_loading", "ava.data",
                      "pocket", "phonon")
        problems = []
        for path in existing_native_files():
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.iter_child_nodes(tree):  # فقط سطح ماژول
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or "") + "." + a.name for a in node.names]
                    names.append(node.module or "")
                else:
                    continue
                for n in names:
                    if any(tok in n for tok in bad_tokens):
                        problems.append(f"{os.path.relpath(path, ROOT)}: {n}")
        self.assertEqual(problems, [], f"legacy imports at module level: {problems}")

    def test_backends_legacy_imports_are_lazy_only(self):
        """آداپتور Legacy مجاز است ولی هیچ import سطح ماژولِ ممنوعی ندارد؛
        importهای اقتباسی فقط lazy داخل متدهای کلاس Legacy هستند."""
        path = os.path.join(ROOT, "ava", "runtime", "backends.py")
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        bad_tokens = ("ava.modules", "tts_model", "flow_lm", "mimi", "phonon", "pocket")
        module_level = [n for n in ast.iter_child_nodes(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom))]
        found = []
        for node in module_level:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                else [f"{node.module}.{a.name}" if node.module else a.name
                      for a in node.names]
            found.extend(names)
        offenders = [n for n in found if any(t in n for t in bad_tokens)]
        self.assertEqual(offenders, [],
                         "legacy imports must stay lazy inside the adapter, found at module level: %s" % offenders)

    def test_ava_package_init_does_not_load_legacy(self):
        """import ava نباید استک Legacy را بار کند (اصلاح فاز ۱)."""
        code = (
            BLOCKER_PRELUDE
            + "import ava\n"
            + "try:\n"
            + "    ava.TTSModel  # noqa — باید ImportError بدهد چون Legacy بلوکه است\n"
            + "except ImportError:\n"
            + "    pass\n"
            + "else:\n"
            + "    raise AssertionError('legacy stack loaded despite blocker')\n"
            + "print('ok')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])

    def test_native_core_imports_without_legacy(self):
        """§۶-۱ — کل هستهٔ Native بدون هیچ ماژول Legacy import می‌شود."""
        code = (
            BLOCKER_PRELUDE
            + "import ava\n"
            + "import ava.language\n"
            + "import ava.models.native\n"
            + "import ava.runtime.engine\n"
            + "import ava.runtime.backends\n"
            + "import ava.audio.vocoder\n"
            + "import ava.voices\n"
            + "import ava.api.server\n"
            + "import ava.cli\n"
            + "print('ok')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])


class TestBoundaryContracts(unittest.TestCase):
    """§۶-۲ و §۶-۳ — قرارداد Text Frontend و ورودی/خروجی مدل."""

    @classmethod
    def setUpClass(cls):
        from ava.language import LanguageEngine
        from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer
        cls.engine = LanguageEngine()
        cls.tok = CharTokenizer()
        cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,
                              ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256,
                              max_frames=512, n_speakers=1)
        cls.cfg = cfg
        cls.model = AVANativeTTS(cfg, len(cls.tok))
        cls.model.eval()

    def test_text_frontend_contract(self):
        """normalize(str)->str و encode(str)->list[int]، قطعی و درون واژگان."""
        from ava.models.native import CharTokenizer
        text = self.engine.normalize("در سال ۱۴۰۳، ۳ کتاب خریدم.")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertEqual(self.engine.normalize(text), text)  # قطعی
        tok2 = CharTokenizer()
        self.assertEqual(tok2.encode(text), self.tok.encode(text))  # قطعی
        ids = self.tok.encode(text)
        vocab = len(self.tok)
        self.assertTrue(all(0 <= i < vocab for i in ids))

    def test_acoustic_model_io_contract(self):
        """ids [1,T] → mel [1,80,T']؛ استنتاج NAR قطعی."""
        import torch
        ids = torch.tensor([self.tok.encode("سلام دنیا")])
        lens = torch.tensor([ids.size(1)])
        with torch.no_grad():
            mel1 = self.model.infer(ids, lens, speed=1.0, speaker_id=torch.tensor([0]))
            mel2 = self.model.infer(ids, lens, speed=1.0, speaker_id=torch.tensor([0]))
        self.assertEqual(tuple(mel1.shape), (80, mel1.size(1)))  # (n_mels, F) — batch 1 فشرده
        self.assertGreater(mel1.size(1), 0)
        self.assertLessEqual(mel1.size(1), self.cfg.max_frames)
        self.assertTrue(torch.equal(mel1, mel2))  # قطعی

    def test_vocoder_boundary_contract(self):
        """مرز Vocoder مستقل: mel → float32 با SR پیکربندی."""
        import numpy as np
        from ava.audio.vocoder import GriffinLimVocoder, get_vocoder
        voc = get_vocoder("griffin-lim-v1")
        self.assertIsInstance(voc, GriffinLimVocoder)
        mel = np.random.RandomState(0).randn(80, 40).astype(np.float32) * 0.1
        wav = voc.mel_to_audio(mel, self.cfg)
        self.assertEqual(wav.dtype, np.float32)
        self.assertGreater(wav.size, 0)
        with self.assertRaises(ValueError):
            get_vocoder("does-not-exist")


class TestInferencePathWithInternalWeights(unittest.TestCase):
    """§۶-۴ و §۶-۵ — مسیر inference با وزن آزمایشی داخلی؛ Runtime بدون Legacy."""

    def _make_bundle(self, out_dir: str) -> str:
        from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer
        from training.train_native import export_native_bundle
        cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,
                              ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256,
                              max_frames=256, n_speakers=1)
        tok = CharTokenizer()
        model = AVANativeTTS(cfg, len(tok))
        return export_native_bundle(model, tok, cfg, out_dir)

    def test_full_inference_with_internal_test_weights(self):
        """وزن آزمایشیِ تولیدشده توسط خود پروژه (مجاز و داخلی) → صوت WAV معتبر."""
        import soundfile as sf
        from ava.runtime import NativeN1Backend, SynthesisOptions, wav_bytes
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._make_bundle(os.path.join(tmp, "bundle"))
            be = NativeN1Backend(bundle)
            voice = be.list_voices()[0]
            res = be.synthesize("سلام. این یک آزمون است.", voice, SynthesisOptions())
            self.assertEqual(res.sample_rate, 24000)
            self.assertGreater(res.duration_s, 0)
            raw = wav_bytes(res)
            self.assertEqual(raw[:4], b"RIFF")

    def test_bundle_format_matches_documentation(self):
        """باندل واقعی دقیقاً سه فایل است (ava_n1.pt + tokenizer.json + meta.json)
        — همان فرمت مستندشده در NATIVE_CONTRACTS/RUNTIME_INSTALL؛ فرمت فرضی
        weights.safetensors + config.json نباید در مستندات باندل Native برگردد."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            bundle = self._make_bundle(os.path.join(tmp, "bundle"))
            self.assertEqual(set(os.listdir(bundle)),
                             {"ava_n1.pt", "tokenizer.json", "meta.json"})
            meta = json.load(open(os.path.join(bundle, "meta.json"), encoding="utf-8"))
            self.assertEqual(meta["format"], "ava-native-n1")
            self.assertIn("sample_rate", meta)
            for doc in ("docs/NATIVE_CONTRACTS.md", "docs/RUNTIME_INSTALL.md"):
                text = open(os.path.join(ROOT, doc), encoding="utf-8").read()
                self.assertIn("ava_n1.pt", text, doc)
                self.assertNotIn("weights.safetensors + tokenizer.json", text, doc)

    def test_runtime_synthesis_without_legacy_modules(self):
        """§۶-۵ — سنتز کامل در پروسه‌ای که همهٔ ماژول‌های Legacy در آن بلوکه‌اند."""
        script = "\n".join([
            BLOCKER_PRELUDE,
            "import os, tempfile, sys",
            f"sys.path.insert(0, {ROOT!r})",
            "import soundfile as sf",
            "from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer",
            "from training.train_native import export_native_bundle",
            "from ava.runtime import NativeN1Backend, SynthesisEngine, SynthesisOptions",
            "cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,"
            " ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256, max_frames=256, n_speakers=1)",
            "tok = CharTokenizer()",
            "model = AVANativeTTS(cfg, len(tok))",
            "tmp = tempfile.mkdtemp()",
            "bundle = export_native_bundle(model, tok, cfg, os.path.join(tmp, 'bundle'))",
            "eng = SynthesisEngine()",
            "eng.register_backend('ava-n1', NativeN1Backend)",
            "eng.load_model('n1', 'ava-n1', bundle_dir=bundle)",
            "r = eng.synthesize('سلام دنیا', 'ava-n1-ava')",
            "assert r.sample_rate == 24000 and r.duration_s > 0",
            "print('ok')",
        ])
        r = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_cli_import_and_help_without_legacy(self):
        """§۶-۶ — CLI موجود در محیط بدون Legacy اجرا می‌شود."""
        code = (BLOCKER_PRELUDE + "from ava.cli import main\n"
                + "try:\n    main(['--help'])\nexcept SystemExit as e:\n"
                + "    assert e.code == 0, e.code\nprint('ok')\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("ava-cli", r.stdout)


class TestLegacyIsolation(unittest.TestCase):
    """§۴ — برچسب‌گذاری و جداسازی دارایی‌های Legacy."""

    def test_native_web_ui_is_self_contained(self):
        """وب Native به هیچ وندور خارجی (phonon/pocket) ارجاع ندارد."""
        page = open(os.path.join(ROOT, "web-native", "index.html"), encoding="utf-8").read()
        self.assertNotIn("phonon", page.lower())
        self.assertNotIn("pocket", page.lower())
        self.assertNotIn("vendor/", page.lower())

    def test_legacy_web_vendor_is_marked_and_licensed(self):
        """وندور Legacy (phonon-tts) فایل مجوز خودش را دارد و خارج از مسیر Native است."""
        vendor = os.path.join(ROOT, "web", "vendor", "phonon-tts")
        self.assertTrue(os.path.isdir(vendor))
        self.assertTrue(os.path.isfile(os.path.join(vendor, "README.md")))
        has_license = (os.path.isfile(os.path.join(vendor, "LICENSE-APACHE"))
                       or os.path.isfile(os.path.join(vendor, "LICENSE-MIT"))
                       or os.path.isfile(os.path.join(vendor, "THIRD_PARTY_LICENSES.md")))
        self.assertTrue(has_license)

    def test_legacy_symbol_access_is_lazy(self):
        """دسترسی به نماد Legacy از راه import ava، فقط با بار تنبل کار می‌کند."""
        code = (
            BLOCKER_PRELUDE
            + "import ava\n"
            + "try:\n"
            + "    ava.TTSModel\n"
            + "except ImportError:\n"
            + "    print('blocked-as-expected')\n"
            + "else:\n"
            + "    raise AssertionError('legacy loaded despite blocker')\n"
        )
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                           capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("blocked-as-expected", r.stdout)


if __name__ == "__main__":
    unittest.main()
