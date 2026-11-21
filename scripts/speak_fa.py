#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""گفتنِ متن فارسی با مدل «آوا» — رابط مشابه speak_fa.py مخزن اصلی.

نمونهٔ استفاده:
    python scripts/speak_fa.py --voice shirin --text "سلام، حال شما چطور است؟" --out hello.wav
    python scripts/speak_fa.py --voice nava --text-file story.txt --out story.wav
    python scripts/speak_fa.py --voice my_voice.wav --text "سلام" --no-normalize

صدا: مسیر یک فایل wav کوتاه (۳ تا ۱۰ ثانیه) به‌عنوان نمونهٔ صدا، یا نام یکی از
صداهای همراه مخزن در weights/voices (nava، shirin، arash).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ava.text_norm import normalize, split_sentences  # noqa: E402

DEFAULT_VOICE = "shirin"


def read_text(args) -> str:
    if args.text_file:
        return pathlib.Path(args.text_file).read_text(encoding="utf-8")
    if not args.text:
        sys.exit("متن را با --text بدهید یا با --text-file مسیر فایل متن را بدهید.")
    return args.text


def resolve_voice(args) -> str:
    """مسیر فایل صدا: مسیر داده‌شده، weights/voices/<name>.safetensors، یا نام صدا."""
    name = args.voice or DEFAULT_VOICE
    p = pathlib.Path(name)
    if p.exists():
        return str(p)
    for suffix in (".safetensors", ".wav"):
        candidate = REPO_ROOT / "weights" / "voices" / f"{name}{suffix}"
        if candidate.exists():
            return str(candidate)
    sys.exit(
        f"صدای «{name}» پیدا نشد. مسیر فایل wav بدهید یا یکی از: "
        "nava، shirin، arash"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="تبدیل متن فارسی به گفتار با آوا")
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help="نام صدا (nava/shirin/arash) یا مسیر فایل نمونهٔ صدا (۳-۱۰ ثانیه)")
    ap.add_argument("--text", default="", help="متن فارسی برای گفتن")
    ap.add_argument("--text-file", default="", help="خواندن متن از فایل (UTF-8)")
    ap.add_argument("--out", default="out.wav", help="مسیر فایل WAV خروجی (پیش‌فرض out.wav)")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "fa.yaml"),
                    help="مسیر فایل پیکربندی مدل")
    ap.add_argument("--temp", type=float, default=None, help="دمای نمونه‌گیری (پیش‌فرض ۰.۳)")
    ap.add_argument("--no-normalize", dest="no_normalize", action="store_true",
                    help="بدون نرمال‌سازی متن (پیش‌فرض: نرمال‌سازی کامل)")
    args = ap.parse_args()

    try:
        import numpy as np
        import scipy.io.wavfile

        from ava import TTSModel
    except ImportError as e:
        sys.exit(f"پکیج‌های لازم نصب نیستند ({e.name}). اول اجرا کنید:\n  pip install -e .")

    text = read_text(args)
    pieces = split_sentences(text.strip())
    if not args.no_normalize:
        pieces = split_sentences(normalize(text.strip()))
    if not pieces:
        sys.exit("چیزی برای گفتن پیدا نشد؛ متن حروف فارسی ندارد.")

    voice = resolve_voice(args)
    model = TTSModel.load_model(args.config, temp=args.temp)
    voice_state = model.get_state_for_audio_prompt(voice)

    t0 = time.time()
    audio = model.generate_audio(voice_state, " ".join(pieces))
    elapsed = time.time() - t0
    duration = audio.numel() / model.sample_rate
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(out, model.sample_rate,
                           (audio.numpy() * 32767).astype(np.int16))
    print(f"«{pieces[0][:40]}{'…' if len(pieces) > 1 or len(pieces[0]) > 40 else ''}»")
    print(f"{duration:.1f} ثانیه صدا در {elapsed:.1f} ثانیه ساخته شد "
          f"(RTF {duration / max(elapsed, 0.01):.2f}x) → {out}")


if __name__ == "__main__":
    main()
