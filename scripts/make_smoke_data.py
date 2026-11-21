#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ساخت دادهٔ دود برای تست خط آموزش — کاملاً بدون وابستگی بیرونی.

با همان مدلِ منتشرشده چند جملهٔ فارسی کوتاه تولید می‌شود؛ متن‌ها (که همین‌جا
نوشته شده‌اند) به‌عنوان ترنسکریپت در مانیفست CSV ذخیره می‌شود. سپس با
training/data/prepare_data.py کش لاتنت ساخته می‌شود.

    python scripts/make_smoke_data.py --n-clips 16 --cache-dir data_cache/smoke

توجه: این داده «برای تست صحت خط آموزش» است، نه برای آموزش واقعی — کیفیت صدا با
۱۶ کلیپ معنا ندارد. برای آموزش واقعی، دیتاست واقعی فارسی را به prepare_data
بدهید (docs/TRAINING.md).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SENTENCES = [
    "سلام دنیا.",
    "امروز هوا آفتابی است.",
    "این یک آزمایش صوت است.",
    "من به بازار رفتم و میوه خریدم.",
    "کتاب‌ها را روی میز گذاشتم.",
    "کوچک‌ترین مدل گفتار فارسی است.",
    "زبان فارسی زیبا و غنی است.",
    "دو دوست به پارک رفتند.",
    "چای تازه دم نوشیدنی خوبی است.",
    "کوه‌های شمال ایران سرسبز هستند.",
    "این جمله برای تست آموزش است.",
    "هر شب پیش از خواب کتاب می‌خوانم.",
    "قطار ساعت شش صبح حرکت می‌کند.",
    "کودکان در حیاط بازی می‌کنند.",
    "زمستان امور برف زیادی آمد.",
    "نوشتن با خودکار آبی آسان است.",
    "عطر گل‌های بهاری دلنشین بود.",
    "خورشید پشت ابر پنهان شد.",
    "او پیانو می‌نوازد و آواز می‌خواند.",
    "دریاچه‌ی ارومیه در غرب ایران است.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="ساخت دادهٔ دود برای تست آموزش")
    ap.add_argument("--n-clips", type=int, default=16, help="تعداد کلیپ (۱ تا ۲۰)")
    ap.add_argument("--cache-dir", default="data_cache/smoke")
    ap.add_argument("--audio-dir", default="data_cache/smoke_audio")
    ap.add_argument("--voice", default="shirin")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "fa.yaml"))
    args = ap.parse_args()

    import numpy as np
    import scipy.io.wavfile

    from ava import TTSModel

    n = max(1, min(args.n_clips, len(SENTENCES)))
    audio_dir = REPO_ROOT / args.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(args.config)
    voice_state = model.get_state_for_audio_prompt(
        str(REPO_ROOT / "weights" / "voices" / f"{args.voice}.safetensors")
    )

    rows = []
    for i, text in enumerate(SENTENCES[:n]):
        audio = model.generate_audio(voice_state, text, copy_state=True)
        wav_path = audio_dir / f"clip-{i:02d}.wav"
        scipy.io.wavfile.write(wav_path, model.sample_rate,
                               (audio.numpy() * 32767).astype(np.int16))
        rows.append((str(wav_path.relative_to(REPO_ROOT)), text))
        print(f"  clip {i + 1}/{n}: «{text}»")

    manifest = REPO_ROOT / args.cache_dir / "smoke_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"مانیفست: {manifest}")
    print(f"قدم بعدی: python -m training.data.prepare_data --manifest {args.cache_dir}/smoke_manifest.csv "
          f"--arch training/configs/smoke_arch.yaml --out {args.cache_dir}")


if __name__ == "__main__":
    main()
