# -*- coding: utf-8 -*-
"""فاز ۲ — تست‌های معماری اختصاصی AVA-N1 (معیارهای AC2.2 و AC2.3)."""
import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ava.audio.mel import griffin_lim_from_mel, mel_spectrogram  # noqa: E402
from ava.language import LanguageEngine  # noqa: E402
from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer  # noqa: E402


def make_cfg(**kw) -> AVANativeConfig:
    base = dict(d_model=64, n_heads=2, n_layers_enc=2, n_layers_dec=2, ffn_dim=128,
                dur_hidden=64, dropout=0.0, max_text_tokens=256, max_frames=512, n_speakers=2)
    return AVANativeConfig(**{**base, **kw})


def synthetic_batch(cfg: AVANativeConfig, tok: CharTokenizer, texts: list[str], seed: int = 7):
    """Deterministic toy training batch: vowel-like tones as mel targets."""
    rng = np.random.default_rng(seed)
    engine = LanguageEngine()
    tokens, lengths, mels, mel_lens, log_durs = [], [], [], [], []
    for i, text in enumerate(texts):
        ids = tok.encode(engine.normalize(text))
        T = len(ids)
        freq = 180 + 60 * i
        n_frames = 40 + 8 * i
        t = np.arange(n_frames * cfg.hop_size) / cfg.sample_rate
        wav = 0.5 * np.sin(2 * np.pi * freq * t) * np.exp(-0.4 * t)
        mel = mel_spectrogram(torch.from_numpy(wav).float(), cfg).unsqueeze(0)
        mels.append(mel)
        mel_lens.append(mel.size(2))
        # uniform duration target: spread the frames over the tokens
        per = max(1, mel.size(2) // T)
        durs = torch.full((T,), float(per), dtype=torch.float32)
        log_durs.append(torch.log(durs))
        tokens.append(torch.tensor(ids))
        lengths.append(T)
    B = len(texts)
    Tm = max(x.size(0) for x in tokens)
    tokens_p = torch.full((B, Tm), tok.pad_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        tokens_p[i, : x.size(0)] = x
    Fm = max(m.size(2) for m in mels)
    mels_p = torch.zeros(B, cfg.n_mels, Fm)
    for i, m in enumerate(mels):
        mels_p[i, :, : m.size(2)] = m[0]
    return {
        "tokens": tokens_p,
        "lengths": torch.tensor(lengths),
        "mel": mels_p,
        "mel_lengths": torch.tensor(mel_lens),
        "log_durations": torch.stack([d[: Tm] if d.size(0) >= Tm else torch.cat([d, d.new_zeros(Tm - d.size(0))]) for d in log_durs]),
    }


class TestTokenizer(unittest.TestCase):
    def test_roundtrip(self):
        tok = CharTokenizer()
        engine = LanguageEngine()
        # توکنایزر روی خروجی موتور زبان کار می‌کند (ارقام قبلاً به کلمه تبدیل شده‌اند)
        text = engine.normalize("سلام دنیا؛ چطور ۲۵۰ تومان؟")
        self.assertEqual(tok.decode(tok.encode(text)), text)

    def test_unknown_char_becomes_unk(self):
        tok = CharTokenizer()
        ids = tok.encode("abc")
        self.assertIn(tok.unk_id, ids)
        self.assertEqual(tok.decode(ids), "<unk><unk><unk>")

    def test_save_load(self):
        tok = CharTokenizer()
        tok.save("/tmp/ava_tok.json")
        tok2 = CharTokenizer.load("/tmp/ava_tok.json")
        self.assertEqual(tok2.encode("می‌روم"), tok.encode("می‌روم"))


class TestMel(unittest.TestCase):
    def test_mel_shape_and_range(self):
        cfg = make_cfg()
        t = np.arange(cfg.sample_rate) / cfg.sample_rate
        wav = np.sin(2 * np.pi * 220 * t) * 0.5
        mel = mel_spectrogram(torch.from_numpy(wav).float(), cfg)
        self.assertEqual(mel.shape[0], cfg.n_mels)
        self.assertTrue(torch.isfinite(mel).all())
        self.assertLessEqual(mel.max().item(), cfg.mel_max)
        self.assertGreaterEqual(mel.min().item(), cfg.mel_min)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.cfg = make_cfg()
        self.tok = CharTokenizer()
        self.model = AVANativeTTS(self.cfg, len(self.tok))
        self.batch = synthetic_batch(self.cfg, self.tok, ["سلام دنیا", "کتاب خواندن خوب است", "صدا طبیعی است"])

    def test_forward_backward_loss_finite(self):
        out = self.model(**self.batch)
        self.assertTrue(torch.isfinite(out["loss"]).all())
        out["loss"].backward()

    def test_loss_actually_decreases(self):
        """آموزش واقعی روی دادهٔ ساختگی: loss باید به‌طور معنادار کاهش یابد."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-3)
        first = last = None
        for step in range(40):
            opt.zero_grad()
            out = self.model(**self.batch)
            out["loss"].backward()
            opt.step()
            if step == 0:
                first = out["loss"].item()
            last = out["loss"].item()
        self.assertLess(last, first * 0.5, f"loss did not halve: {first:.3f} -> {last:.3f}")

    def test_infer_shapes_and_wav(self):
        self.model.eval()
        ids = torch.tensor([self.tok.encode("سلام")])
        mel = self.model.infer(ids, torch.tensor([ids.size(1)]))
        self.assertEqual(mel.size(0), self.cfg.n_mels)
        self.assertGreater(mel.size(1), 0)
        wav = griffin_lim_from_mel(mel.numpy(), self.cfg)
        self.assertEqual(wav.dtype, np.float32)
        self.assertGreater(np.abs(wav).max(), 0.0)
        import soundfile as sf
        sf.write("/tmp/ava_n1_test.wav", wav, self.cfg.sample_rate)
        info = sf.info("/tmp/ava_n1_test.wav")
        self.assertEqual(info.samplerate, self.cfg.sample_rate)
        self.assertGreater(info.frames, 0)


class TestArchitectureReadiness(unittest.TestCase):
    """AC2.3 — آمادگی چندگوینده و استنتاج قطعی."""

    def test_multi_speaker_embedding_swaps_output(self):
        cfg = make_cfg()
        tok = CharTokenizer()
        model = AVANativeTTS(cfg, len(tok))
        model.eval()
        ids = torch.tensor([tok.encode("سلام")])
        with torch.no_grad():
            m0 = model.infer(ids, torch.tensor([4]), speaker_id=torch.tensor([0]))
            m1 = model.infer(ids, torch.tensor([4]), speaker_id=torch.tensor([1]))
        # طول فریم گوینده‌های متفاوت می‌تواند کمی متفاوت باشد؛ روی بخش مشترک مقایسه می‌کنیم
        self.assertGreater(m0.size(1), 0)
        self.assertGreater(m1.size(1), 0)
        f = min(m0.size(1), m1.size(1))
        self.assertFalse(torch.allclose(m0[:, :f], m1[:, :f]), "different speakers must give different mels")

    def test_deterministic_inference(self):
        cfg = make_cfg()
        tok = CharTokenizer()
        model = AVANativeTTS(cfg, len(tok))
        model.eval()
        ids = torch.tensor([tok.encode("تست")])
        with torch.no_grad():
            a = model.infer(ids, torch.tensor([4]))
            b = model.infer(ids, torch.tensor([4]))
        self.assertTrue(torch.allclose(a, b), "inference must be deterministic (NAR)")

    def test_speed_changes_frame_count(self):
        cfg = make_cfg()
        tok = CharTokenizer()
        model = AVANativeTTS(cfg, len(tok))
        model.eval()
        ids = torch.tensor([tok.encode("یک جمله طولانی‌تر برای تست")])
        with torch.no_grad():
            f1 = model.infer(ids, torch.tensor([ids.size(1)]), speed=1.0).size(1)
            f2 = model.infer(ids, torch.tensor([ids.size(1)]), speed=2.0).size(1)
        self.assertLessEqual(f2, f1, "faster speed must not produce more frames")


if __name__ == "__main__":
    unittest.main(verbosity=2)
