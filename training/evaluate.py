# -*- coding: utf-8 -*-
"""ارزیابی چک‌پوینت آموزش: بازسازی معلم‌اجباری + تولید نمونهٔ صدا.

    python -m training.evaluate --checkpoint runs/ava-fa/checkpoint-latest.pt \
        --arch config/fa.yaml --out-dir runs/ava-fa/eval

شاخص‌ها:
- mean cosine فاصلهٔ پیش‌بینی سر drifting با هدف واقعی (معلم‌اجباری، روی holdout)
- آهنگ EOS درست/نادرست
- ساخت چند WAV نمونه با صدای داده‌شده (پیش‌فرض: shirin مخزن)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAMPLE_TEXTS = [
    "سلام دنیا.",
    "امروز هوا آفتابی است و من به پارک می‌روم.",
    "این صدای مدل آموزش‌دیدهٔ فارسی است.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="ارزیابی چک‌پوینت flow LM")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--arch", default=str(REPO_ROOT / "config" / "fa.yaml"))
    ap.add_argument("--voice", default=str(REPO_ROOT / "weights" / "voices" / "shirin.safetensors"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--texts", nargs="*", default=SAMPLE_TEXTS)
    args = ap.parse_args()

    import scipy.io.wavfile as wav
    import torch

    from ava import TTSModel

    arch_path = Path(args.arch)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TTSModel.load_model(str(arch_path), checkpoint=args.checkpoint)
    report: dict = {"checkpoint": str(args.checkpoint), "samples": []}

    # 1) Reconstruction quality on the model's own text chunking (teacher-forced cosine)
    try:
        from training.data.dataset import Collator, LatentDataset
        from training.flow_lm_training import teacher_forced_reconstruction

        cache_dirs = sorted((REPO_ROOT / "data_cache").glob("*"))
        if cache_dirs:
            ds = LatentDataset(cache_dirs[-1], model.flow_lm.conditioner.tokenizer,
                               max_frames=250, max_tokens=64, min_frames=10,
                               holdout_ratio=0.2, holdout=True)
            if len(ds):
                dl = torch.utils.data.DataLoader(
                    ds, batch_size=4, collate_fn=Collator(ldim=model.flow_lm.ldim))
                cos = sum(float(teacher_forced_reconstruction(model.flow_lm, b)) for b in dl) / len(dl)
                report["holdout_cosine"] = cos
                print(f"holdout cosine: {cos:.4f} ({len(ds)} کلیپ)")
            else:
                print("holdout خالی — فقط تولید نمونه انجام می‌شود.")
    except Exception as e:  # noqa: BLE001 — evaluation must not die without a cache
        print(f"ارزیابی بازسازی رد شد: {e}")

    # 2) Sample generation
    voice_state = model.get_state_for_audio_prompt(str(args.voice))
    for i, text in enumerate(args.texts):
        t0 = time.time()
        audio = model.generate_audio(voice_state, text, copy_state=True)
        dur = audio.numel() / model.sample_rate
        path = out_dir / f"sample-{i:02d}.wav"
        wav.write(path, model.sample_rate, (audio.numpy() * 32767).astype("int16"))
        report["samples"].append({"text": text, "path": path.name,
                                  "seconds": round(dur, 2),
                                  "rtf": round(dur / max(time.time() - t0, 0.01), 2)})
        print(f"  {path.name}: «{text[:30]}» ({dur:.1f}s)")

    (out_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"گزارش: {out_dir / 'eval_report.json'}")


if __name__ == "__main__":
    main()
