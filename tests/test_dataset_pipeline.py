# -*- coding: utf-8 -*-
"""فاز ۳ — تست‌های پایپ‌لاین دیتاست (معیارهای AC3.1 تا AC3.4)."""
import os
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.datasets import (  # noqa: E402
    ManifestEntry, ValidationLimits, assign_splits, dataset_fingerprint,
    provenance_report, read_manifest, segment_audio, validate_audio, write_manifest,
)


def make_entry(dir_path: str, name: str, text: str = "سلام") -> ManifestEntry:
    return ManifestEntry(
        id=name, audio_path=os.path.join(dir_path, name + ".wav"),
        text=text, speaker_id="spk1",
        source="unit-test", license="test-fixture", consent_ref="UT-001",
    )


def write_wav(path: str, x: np.ndarray, sr: int = 24000, subtype: str = "PCM_16") -> None:
    sf.write(path, x.astype(np.float32), sr, subtype=subtype)


class TestManifest(unittest.TestCase):
    """AC3.1 و AC3.4 — manifest و منشأ."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "manifest.jsonl")

    def test_roundtrip(self):
        e = make_entry(self.tmp, "u1")
        write_manifest(self.path, [e])
        loaded = read_manifest(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, "u1")
        self.assertEqual(loaded[0].license, "test-fixture")

    def test_malformed_line_raises(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{broken json\n")
        with self.assertRaises(ValueError):
            read_manifest(self.path)

    def test_fingerprint_stable(self):
        a, b = make_entry(self.tmp, "u1"), make_entry(self.tmp, "u2")
        self.assertEqual(dataset_fingerprint([a, b]), dataset_fingerprint([b, a]))
        b.text = "متن دیگر"
        self.assertNotEqual(dataset_fingerprint([a, b]), dataset_fingerprint([a, make_entry(self.tmp, "u2")]))

    def test_provenance_report(self):
        entries = [make_entry(self.tmp, "u1"), make_entry(self.tmp, "u2")]
        entries[1].consent_ref = ""
        entries[1].valid = False
        rep = provenance_report(entries)
        self.assertEqual(rep["total"], 2)
        self.assertEqual(rep["consent_missing"], 1)
        self.assertEqual(rep["invalid"], 1)


class TestValidation(unittest.TestCase):
    """AC3.2 — اعتبارسنجی صوت."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.limits = ValidationLimits()

    def test_good_bursts_pass(self):
        sr = 24000
        speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(int(1.2 * sr)) / sr)
        silence = np.zeros(int(0.3 * sr), dtype=np.float32)
        wav = np.concatenate([silence, speech, silence, speech])
        e = make_entry(self.tmp, "good")
        write_wav(e.audio_path, wav, sr)
        issues = validate_audio(e, self.limits)
        self.assertEqual(e.valid, True, issues)
        self.assertEqual(e.sample_rate, sr)
        self.assertEqual(e.channels, 1)

    def test_silent_flagged(self):
        e = make_entry(self.tmp, "silent")
        write_wav(e.audio_path, np.zeros(int(2.0 * 24000)), 24000)
        issues = validate_audio(e, self.limits)
        self.assertFalse(e.valid)
        self.assertIn("silence", " ".join(issues))

    def test_wrong_sample_rate_flagged(self):
        speech = 0.4 * np.sin(2 * np.pi * 220 * np.arange(int(1.5 * 16000)) / 16000)
        e = make_entry(self.tmp, "sr")
        write_wav(e.audio_path, speech, 16000)
        issues = validate_audio(e, self.limits)
        self.assertFalse(e.valid)
        self.assertTrue(any(i.startswith("sample_rate_") for i in issues))

    def test_too_short_flagged(self):
        e = make_entry(self.tmp, "short")
        write_wav(e.audio_path, 0.4 * np.sin(2 * np.pi * 220 * np.arange(int(0.2 * 24000)) / 24000), 24000)
        issues = validate_audio(e, self.limits)
        self.assertIn("too_short", issues)

    def test_clipping_flagged(self):
        sr = 24000
        wav = np.sign(np.sin(2 * np.pi * 220 * np.arange(int(1.2 * sr)) / sr)) * 0.5
        wav = np.clip(wav * 3.0, -1.0, 1.0)  # heavy clipping
        e = make_entry(self.tmp, "clip")
        write_wav(e.audio_path, wav, sr)
        issues = validate_audio(e, self.limits)
        self.assertTrue(any(i.startswith("clipping") for i in issues), issues)

    def test_missing_file(self):
        e = make_entry(self.tmp, "ghost")
        issues = validate_audio(e, self.limits)
        self.assertFalse(e.valid)
        self.assertIn("missing_file", issues)


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_two_bursts_two_segments(self):
        sr = 24000
        sp1 = 0.5 * np.sin(2 * np.pi * 200 * np.arange(int(1.0 * sr)) / sr)
        gap = np.zeros(int(0.8 * sr), dtype=np.float32)
        sp2 = 0.5 * np.sin(2 * np.pi * 300 * np.arange(int(1.0 * sr)) / sr)
        path = os.path.join(self.tmp, "rec.wav")
        write_wav(path, np.concatenate([sp1, gap, sp2]), sr)
        segs = segment_audio(path, sr=sr)
        self.assertEqual(len(segs), 2, segs)
        d1 = segs[0][1] - segs[0][0]
        self.assertGreater(d1, 0.5)

    def test_all_silence_no_segments(self):
        path = os.path.join(self.tmp, "sil.wav")
        write_wav(path, np.zeros(int(2.0 * 24000)), 24000)
        self.assertEqual(segment_audio(path), [])


class TestSplitting(unittest.TestCase):
    """AC3.3 — split قطعی و ناپیوسته بودن گوینده‌ها."""

    def _entries(self, n_speakers=6, per=5):
        out = []
        for s in range(n_speakers):
            for i in range(per):
                out.append(ManifestEntry(id=f"u{s}_{i}", audio_path=f"{s}_{i}.wav", text="متن",
                                         speaker_id=f"spk{s}", source="t", license="t", consent_ref="c"))
        return out

    def test_deterministic(self):
        a, b = self._entries(), self._entries()
        assign_splits(a, seed="s1")
        assign_splits(b, seed="s1")
        self.assertEqual([e.split for e in a], [e.split for e in b])

    def test_speaker_disjoint(self):
        entries = self._entries()
        assign_splits(entries, seed="s2", speaker_disjoint=True)
        by_spk: dict[str, set] = {}
        for e in entries:
            by_spk.setdefault(e.speaker_id, set()).add(e.split)
        for spk, splits in by_spk.items():
            self.assertEqual(len(splits), 1, f"speaker {spk} in {splits}")

    def test_all_three_splits_present(self):
        entries = self._entries(n_speakers=100)
        counts = assign_splits(entries, seed="s3")
        self.assertGreater(counts["train"], 0)
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
