# -*- coding: utf-8 -*-
"""فاز ۴ — تست‌های پایپ‌لاین آموزش AVA-N1 (معیارهای AC4.1 تا AC4.4).

همهٔ آموزش‌ها واقعی روی CPU اجرا می‌شوند؛ دادهٔ ساختگی صریحاً fixture است.
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.make_native_smoke_data import generate  # noqa: E402
from training.train_native import (  # noqa: E402
    AVANativeConfig, TrainConfig, export_native_bundle, load_checkpoint, load_native_bundle,
    save_checkpoint, synth_to_wav, train,
)
import torch  # noqa: E402

TINY_MODEL = dict(d_model=32, n_heads=2, n_layers_enc=2, n_layers_dec=2, ffn_dim=64,
                  dur_hidden=32, dropout=0.0, max_text_tokens=256, max_frames=256, n_speakers=1)


def tiny_cfg(**kw) -> TrainConfig:
    base = dict(steps=30, batch_size=4, lr=3e-3, log_every=5, val_every=30, checkpoint_every=30)
    base["model"] = {**TINY_MODEL, **kw.get("model", {})}
    return TrainConfig(**{**base, **{k: v for k, v in kw.items() if k != "model"}})


class TestNativeTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """یک آموزش پایهٔ واقعی یک‌بار اجرا می‌شود تا همهٔ تست‌ها مستقل از ترتیب، artifact داشته باشند."""
        cls.tmp = tempfile.mkdtemp()
        cls.manifest = generate(os.path.join(cls.tmp, "ds"), n=12)
        cls.out = os.path.join(cls.tmp, "run")
        cls.baseline = train(cls.manifest, cls.out, tiny_cfg())
        cls.baseline_steps = tiny_cfg().steps

    def test_full_smoke_training_with_real_loss_decrease(self):
        """AC4.1 — آموزش واقعی در مسیر تازه: loss باید کاهش معنادار داشته باشد."""
        out2 = os.path.join(self.tmp, "run2")
        summary = train(self.manifest, out2, tiny_cfg())
        self.assertIsNotNone(summary["first"])
        self.assertLess(summary["last"], summary["first"] * 0.8,
                        f"loss should drop >20%: {summary['first']:.3f} -> {summary['last']:.3f}")
        with open(os.path.join(out2, "log.jsonl"), encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        self.assertTrue(any("val_loss" in r for r in recs))  # AC4.2 validation visible
        self.assertTrue(os.path.isfile(os.path.join(out2, "checkpoint.pt")))  # AC4.1 checkpoint

    def test_resume_from_checkpoint(self):
        """AC4.1 — resume واقعی از checkpoint."""
        from ava.models.native import AVANativeTTS, CharTokenizer
        ckpt = os.path.join(self.out, "checkpoint.pt")
        self.assertTrue(os.path.isfile(ckpt))
        mcfg = AVANativeConfig.from_dict(tiny_cfg().model)
        model = AVANativeTTS(mcfg, len(CharTokenizer()))
        opt = torch.optim.AdamW(model.parameters())
        step, tcfg = load_checkpoint(ckpt, model, opt)
        self.assertEqual(step, self.baseline_steps)
        cfg2 = tiny_cfg()
        cfg2.steps = step + 10
        summary = train(self.manifest, self.out, cfg2, resume=True)
        self.assertEqual(summary["steps"], step + 10)

    def test_bundle_export_load_and_synth(self):
        """AC4.3 و AC4.4 — export واقعی + تولید WAV واقعی از باندل."""
        bundle = os.path.join(self.out, "bundle")
        self.assertTrue(os.path.isdir(bundle))
        model, tok, mcfg = load_native_bundle(bundle)
        meta = json.load(open(os.path.join(bundle, "meta.json"), encoding="utf-8"))
        self.assertEqual(meta["format"], "ava-native-n1")
        self.assertGreater(meta["params_m"], 0)
        wav_path = os.path.join(self.tmp, "synth.wav")
        dur = synth_to_wav(model, tok, mcfg, "سلام دنیا", wav_path)
        import soundfile as sf
        info = sf.info(wav_path)
        self.assertEqual(info.samplerate, mcfg.sample_rate)
        self.assertGreater(info.frames, 0)
        self.assertGreater(dur, 0.05)

    def test_empty_dataset_raises(self):
        import json as _json
        from training.datasets.manifest import ManifestEntry, write_manifest
        with tempfile.TemporaryDirectory() as d:
            # all-invalid entries -> empty dataset -> RuntimeError
            entries = [ManifestEntry(id="x", audio_path="/nonexistent.wav", text="سلام",
                                     speaker_id="s", source="t", license="t", consent_ref="c", valid=False)]
            mp = os.path.join(d, "bad.jsonl")
            write_manifest(mp, entries)
            with self.assertRaises(RuntimeError):
                train(mp, os.path.join(d, "out"), tiny_cfg(steps=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
