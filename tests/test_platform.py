# -*- coding: utf-8 -*-
"""فاز ۷ — تست‌های API، CLI و مسیر ONNX (معیارهای AC7.1، AC7.2، AC7.4)."""
import base64
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

from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer  # noqa: E402
from ava.runtime import NativeN1Backend, SynthesisEngine  # noqa: E402
from training.train_native import export_native_bundle  # noqa: E402


def make_bundle(out_dir: str) -> str:
    cfg = AVANativeConfig(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2,
                          ffn_dim=64, dur_hidden=32, dropout=0.0, max_text_tokens=256,
                          max_frames=128, n_speakers=1)
    tok = CharTokenizer()
    model = AVANativeTTS(cfg, len(tok))
    return export_native_bundle(model, tok, cfg, out_dir)


class TestAPI(unittest.TestCase):
    """AC7.1 — API با تست."""

    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))
        os.environ["AVA_NATIVE_BUNDLES"] = f"n1={cls.bundle}"
        from ava.api.server import app, engine, load_bundles_from_env
        engine.register_backend("ava-n1", NativeN1Backend)
        load_bundles_from_env()
        cls.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/v1/health")
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.json()["models"], 1)

    def test_voices(self):
        r = self.client.get("/v1/voices")
        ids = [v["voice_id"] for v in r.json()]
        self.assertIn("ava-n1-ava", ids)

    def test_synthesis_wav(self):
        r = self.client.post("/v1/synthesis", json={"text": "سلام دنیا.", "voice": "ava-n1-ava"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.content.startswith(b"RIFF"))
        self.assertIn("x-duration-seconds", {k.lower() for k in r.headers})

    def test_rejects_unknown_voice(self):
        r = self.client.post("/v1/synthesis", json={"text": "سلام", "voice": "ناموجود"})
        self.assertEqual(r.status_code, 404)

    def test_rejects_empty_and_oversized_text(self):
        self.assertEqual(self.client.post("/v1/synthesis", json={"text": "", "voice": "ava-n1-ava"}).status_code, 422)
        self.assertEqual(self.client.post("/v1/synthesis", json={"text": "x" * 6000, "voice": "ava-n1-ava"}).status_code, 422)

    def test_rejects_out_of_range_speed(self):
        r = self.client.post("/v1/synthesis", json={"text": "سلام", "voice": "ava-n1-ava", "speed": 9.0})
        self.assertEqual(r.status_code, 422)

    def test_streaming_ndjson(self):
        r = self.client.post("/v1/synthesis/stream", json={"text": "الف. ب.", "voice": "ava-n1-ava"})
        self.assertEqual(r.status_code, 200)
        lines = [json.loads(l) for l in r.text.strip().splitlines()]
        self.assertEqual(len(lines), 2)
        wav = base64.b64decode(lines[0]["audio_wav_b64"])
        self.assertTrue(wav.startswith(b"RIFF"))


class TestCLI(unittest.TestCase):
    """AC7.2 — CLI با اجرای واقعی."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))

    def _cli(self, *args):
        from ava.cli import main
        return main(list(args))

    def test_speak_and_voices(self):
        out = os.path.join(self.tmp, "cli.wav")
        rc = self._cli("speak", "--text", "سلام دنیا.", "--voice", "ava-n1-ava",
                       "--out", out, "--bundle", self.bundle)
        self.assertEqual(rc, 0)
        info = sf.info(out)
        self.assertEqual(info.samplerate, 24000)
        self.assertGreater(info.frames, 0)

        import contextlib, io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self._cli("voices", "--bundle", self.bundle)
        self.assertEqual(rc, 0)
        self.assertIn("ava-n1-ava", buf.getvalue())

    def test_unknown_voice_fails_cleanly(self):
        import contextlib, io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stderr(buf):
            rc = self._cli("speak", "--text", "سلام", "--voice", "nope", "--out",
                           os.path.join(self.tmp, "x.wav"), "--bundle", self.bundle)
        self.assertEqual(rc, 2)

    def test_voice_quality_command(self):
        wav = os.path.join(self.tmp, "q.wav")
        t = np.arange(int(4 * 24000)) / 24000
        burst = np.minimum(1, (t % 1.0) * 40) * 0.5 * np.sin(2 * np.pi * 210 * t)
        sf.write(wav, burst, 24000)
        import contextlib, io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = self._cli("voice-quality", "--wav", wav)
        self.assertEqual(rc, 0)
        self.assertIn('"ok": true', buf.getvalue())


class TestOnnxExport(unittest.TestCase):
    """AC7.4 — export ONNX واقعی + اجرا با onnxruntime."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.bundle = make_bundle(os.path.join(cls.tmp, "bundle"))

    def test_export_and_parity(self):
        from training.export.onnx_export import export_onnx, verify_onnx
        path = export_onnx(self.bundle, opset=17)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 10_000)
        rep = verify_onnx(self.bundle, path, text="سلام")
        self.assertTrue(rep["ok"], rep)
        self.assertGreater(rep["onnx_frames"], 0)

    def test_onnx_runs_in_ort_only_session(self):
        import onnxruntime as ort
        from training.export.onnx_export import export_onnx
        path = export_onnx(self.bundle)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        n_in = sess.get_inputs()[0].shape[1]
        ids = [1, 5, 7, 2] + [0] * (n_in - 4)  # pad to the static budget
        out = sess.run(None, {"tokens": np.array([ids], dtype="int64"),
                              "lengths": np.array([4], dtype="int64"),
                              "speed": np.array([1.0], dtype="float32")})
        self.assertEqual(out[0].shape[1], 80)  # n_mels


if __name__ == "__main__":
    unittest.main(verbosity=2)
