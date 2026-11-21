#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ساخت فایل صدا (voice) از یک نمونهٔ گفتار — معادل export-voice در pocket-tts.

یک فایل wav کوتاه (۳ تا ۱۰ ثانیه) بدهید؛ خروجی یک safetensors شامل بردارهای
conditioning آمادهٔ مصرف است (همان قالب weights/voices/*.safetensors) که هم در
پایتون و هم در دموی مرورگری مستقیماً بارگذاری می‌شود.

    python scripts/export_voice.py --audio my_voice.wav --out weights/voices/my_voice.safetensors
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="ذخیرهٔ صدای نمونه به قالب safetensors")
    ap.add_argument("--audio", required=True, help="فایل wav نمونهٔ صدا (۳-۱۰ ثانیه)")
    ap.add_argument("--out", required=True, help="مسیر خروجی .safetensors")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "fa.yaml"))
    ap.add_argument("--raw-wave", action="store_true",
                    help="به‌جای conditioning، خود موج صوتی (۲۴kHz mono) را ذخیره کن")
    ap.add_argument("--normalized", action="store_true",
                    help="لاتنت‌ها را قبل از projection نرمال کن (برای برخی چک‌پوینت‌های "
                         "خارج از کیوتای؛ پیش‌فرض: همان رفتار pocket-tts رسمی)")
    args = ap.parse_args()

    import safetensors.torch
    import torch

    from ava import TTSModel

    model = TTSModel.load_model(args.config)
    src = pathlib.Path(args.audio)
    if not src.exists():
        sys.exit(f"فایل «{src}» پیدا نشد.")

    if args.raw_wave:
        from ava.data.audio import audio_read
        from ava.data.audio_utils import convert_audio

        wav, sr = audio_read(src)
        wav = convert_audio(wav, sr, model.sample_rate, 1)
        audio_prompt = wav[0].contiguous()
        safetensors.torch.save_file({"audio_prompt": audio_prompt}, args.out)
        print(f"ذخیره شد: {args.out} ({audio_prompt.numel() / model.sample_rate:.1f} ثانیه موج صوتی)")
        return

    # Precompute the voice conditioning (what the transformer consumes as the prompt):
    # codec latents -> speaker projection, identical to TTSModel._encode_audio.
    from ava.data.audio import audio_read
    from ava.data.audio_utils import convert_audio, end_on_pause

    wav, sr = audio_read(src)
    wav = convert_audio(wav, sr, model.sample_rate, 1)
    wav = end_on_pause(wav, model.sample_rate)
    encoded = model.mimi.encode_to_latent(wav.unsqueeze(0).to(model.device)).to(torch.float32)
    if args.normalized:
        encoded = (encoded - model.flow_lm.emb_mean) / model.flow_lm.emb_std
    import torch.nn.functional as F

    prompt = F.linear(encoded, model.flow_lm.speaker_proj_weight)[0].cpu().contiguous()
    safetensors.torch.save_file({"audio_prompt": prompt}, args.out)
    print(f"ذخیره شد: {args.out} ({prompt.shape[0]} فریم conditioning @ 12.5Hz = "
          f"{prompt.shape[0] / 12.5:.1f} ثانیه)")


if __name__ == "__main__":
    main()
