# -*- coding: utf-8 -*-
"""خروجی‌گرفتن بستهٔ نهایی مدل از یک چک‌پوینت آموزش.

    python -m training.export --checkpoint runs/ava-fa/checkpoint-latest.pt \
        --arch config/fa.yaml [--out weights/model.safetensors] [--sharded]

بستهٔ خروجی دقیقاً همان قالب Pocket TTS است: کلیدهای flow_lm.* (از چک‌پوینت، با
وزن‌های EMA اگر موجود باشند) + کلیدهای mimi.* (کدک منجمد از بستهٔ مرجع).
خروجی هم مستقیماً با ava.TTSModel.load_model و هم با دموی مرورگری سازگار است.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ava.utils.config import load_config as load_arch_config  # noqa: E402
from ava.utils.utils import resolve_path  # noqa: E402
from ava.utils.weights_io import load_file, save_file_sharded  # noqa: E402
from ava.utils.weights_loading import get_training_checkpoint_state_dicts  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="خروجی بستهٔ مدل از چک‌پوینت آموزش")
    ap.add_argument("--checkpoint", required=True, help="فایل .pt مرحلهٔ آموزش")
    ap.add_argument("--arch", default=str(REPO_ROOT / "config" / "fa.yaml"))
    ap.add_argument("--out", default=str(REPO_ROOT / "weights" / "model.safetensors"),
                    help="مسیر بستهٔ خروجی (بدون پسوند شمارهٔ shard)")
    ap.add_argument("--no-sharded", action="store_true", help="یک فایل واحد بنویس")
    ap.add_argument("--no-ema", action="store_true", help="از وزن‌های خام استفاده کن (نه EMA)")
    args = ap.parse_args()

    import torch

    ckpt_path = Path(args.checkpoint).resolve()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    flow_state, _ = get_training_checkpoint_state_dicts(ckpt_path, use_ema=not args.no_ema)
    step = payload.get("step", "?")

    arch = load_arch_config(args.arch)
    base_path = resolve_path(arch.weights_path, REPO_ROOT)
    base = load_file(base_path)
    mimi_state = {k.removeprefix("mimi."): v for k, v in base.items() if k.startswith("mimi.")}

    bundle: dict[str, torch.Tensor] = {}
    for k, v in flow_state.items():
        bundle[f"flow_lm.{k}"] = v.float()
    for k, v in mimi_state.items():
        bundle[f"mimi.{k}"] = v.float()

    out = Path(args.out)
    written = save_file_sharded(
        bundle, out,
        max_shard_size_mb=1e9 if args.no_sharded else 95.0,
        metadata={"format": "pt", "source_checkpoint": str(ckpt_path.name), "step": str(step)},
    )
    total = sum(t.numel() for t in bundle.values())
    print(f"بستهٔ خروجی: {len(bundle)} تنسور ({total / 1e6:.2f}M پارامتر) از گام {step}")
    for p in written:
        print(f"  {p.relative_to(out.parent)}  {p.stat().st_size / 1e6:.1f} MB")
    if not args.no_sharded:
        print(f"  {out.parent / (out.name + '.safetensors.index.json')}")

    # Verify the exported bundle loads strictly into the architecture.
    from pathlib import Path as _Path

    from ava.models.tts_model import TTSModel

    model_config = load_arch_config(args.arch)
    model_config.weights_path = str(out)
    model = TTSModel._from_pydantic_config_with_weights(
        model_config, temp=0.3, sampler_decode_steps=1, noise_clamp=None,
        eos_threshold=-2.0, origin=_Path(args.arch),
    )
    print("تأیید: بسته با load_state_dict(strict=True) بارگذاری شد ✓")
    n_mimi = sum(t.numel() for n, t in model.state_dict().items() if n.startswith("mimi."))
    print(f"یادآوری: {n_mimi / 1e6:.2f}M پارامتر کدک Mimi از بستهٔ مرجع آمده و "
          "محدودیت مجوز CC BY-NC 4.0 را به ارث می‌برد.")


if __name__ == "__main__":
    main()
