#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ساخت بستهٔ استاتیک دموی مرورگری (web_dist/) برای میزبانی استاتیک.

خروجی: یک پوشهٔ مستقل شامل همهٔ فایل‌های web/ به‌علاوهٔ model/ (فایل merge‌شدهٔ
وزن‌ها، tokenizer.json، config.json، voices/) — آمادهٔ آپلود روی هر هاست استاتیک
(سرویس‌دهنده باید هدرهای COOP/COEP را پشتیبانی کند؛ GitHub Pages این کار را
نمی‌کند — برای Pages از scripts/serve.py یا یک worker معادل استفاده کنید).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="ساخت بستهٔ استاتیک دموی وب")
    ap.add_argument("--out", default=str(REPO_ROOT / "web_dist"))
    ap.add_argument("--skip-weights", action="store_true",
                    help="فایل model.safetensors را کپی نکن (وزن‌ها را جداگانه آپلود می‌کنید)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # The demo layout: entry files at the root, model/ next to them.
    shutil.copytree(REPO_ROOT / "web", out,
                    ignore=shutil.ignore_patterns("model.safetensors", "__pycache__"))
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "weights" / "tokenizer.json", model_dir / "tokenizer.json")
    shutil.copy(REPO_ROOT / "weights" / "model_config.json", model_dir / "config.json")
    shutil.copytree(REPO_ROOT / "weights" / "voices", model_dir / "voices", dirs_exist_ok=True)

    if not args.skip_weights:
        sys.path.insert(0, str(REPO_ROOT))
        from ava.utils.weights_io import load_file

        import safetensors.torch

        state = load_file(REPO_ROOT / "weights" / "model.safetensors")
        safetensors.torch.save_file(state, model_dir / "model.safetensors")

    n = sum(1 for _ in out.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"بستهٔ دمو ساخته شد: {out} ({n} فایل، {size / 1e6:.1f} MB)")
    print("یادآوری: هاست باید هدرهای Cross-Origin-Opener-Policy/Embedder-Policy بفرستد.")


if __name__ == "__main__":
    main()
