# -*- coding: utf-8 -*-
"""آموزش اختیاری کدک Mimi (خودکدگذار) — خارج از روایت اصلی کیوتای.

روایت اصلی Ava: کدک منجمد است و فقط flow LM آموزش می‌بیند
(training.train_flow_lm). این اسکریپت برای حالتی است که بخواهید کدک را روی
دادهٔ خودتان فاین‌تیون کنید: بازسازی موج صوتی با تلفیق L1 موج + STFT چند-تفکیکی.
مدل‌های کدکی صنعتی، به‌جز این، از آموزش تخاصمی (GAN) هم استفاده می‌کنند که در
این مخزن گنجانده نشده — برای کیفیت بهتر به داده و گام‌های بیشتری نیاز است.

    python -m training.train_codec --config training/configs/codec_cpu.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def multi_res_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on log-magnitude STFT at several resolutions + waveform L1."""
    loss = F.l1_loss(pred, target)
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        win = torch.hann_window(n_fft, device=pred.device)
        spec_p = torch.stft(pred.squeeze(1), n_fft, hop, window=win, return_complex=True).abs()
        spec_t = torch.stft(target.squeeze(1), n_fft, hop, window=win, return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log10(spec_p.clamp_min(1e-7)), torch.log10(spec_t.clamp_min(1e-7)))
    return loss


def main() -> None:
    ap = argparse.ArgumentParser(description="فاین‌تیون کدک Mimi (اختیاری)")
    ap.add_argument("--manifest", required=True, help="CSV (مسیر صوت، متن) — متن اینجا بی‌اثر است")
    ap.add_argument("--arch", default=str(REPO_ROOT / "config" / "fa.yaml"))
    ap.add_argument("--out-dir", default="runs/codec")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-seconds", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()

    from training.data.prepare_data import read_manifest

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in read_manifest(Path(args.manifest)) if Path(r[0]).exists()]
    if not rows:
        sys.exit("هیچ فایل صوتی معتبری در مانیفست نیست.")

    from ava.data.audio import audio_read
    from ava.data.audio_utils import convert_audio
    from ava.models.mimi import build_mimi
    from ava.utils.config import load_config as load_arch_config
    from ava.utils.weights_io import load_file
    from ava.utils.utils import resolve_path

    arch = load_arch_config(args.arch)
    mimi = build_mimi(arch.mimi)
    base = load_file(resolve_path(arch.weights_path, REPO_ROOT))
    mimi.load_state_dict({k.removeprefix("mimi."): v for k, v in base.items() if k.startswith("mimi.")})
    mimi.train().to(device)
    n_params = sum(p.numel() for p in mimi.parameters())

    optimizer = torch.optim.AdamW(mimi.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    sr = arch.mimi.sample_rate
    frame = int(args.batch_seconds * sr)
    t0 = time.time()
    losses = []
    print(f"codec fine-tune: {n_params / 1e6:.1f}M params, {len(rows)} clips, device={device}")
    for step in range(1, args.steps + 1):
        batch = []
        for _ in range(args.batch_size):
            path, _ = rows[rng.integers(len(rows))]
            wav, wav_sr = audio_read(path)
            wav = convert_audio(wav, wav_sr, sr, 1)[0]
            if wav.numel() <= frame:
                wav = torch.cat([wav, torch.zeros(frame - wav.numel())])
            start = int(rng.integers(0, wav.numel() - frame + 1))
            batch.append(wav[start:start + frame])
        x = torch.stack(batch).unsqueeze(1).to(device)  # [B, 1, T]

        latents = mimi.encode_to_latent(x)             # [B, T', 32]
        state = None
        emb = mimi.quantizer(latents.transpose(-1, -2))
        emb = mimi._to_encoder_framerate(emb, state)
        (emb,) = mimi.decoder_transformer(emb, state)
        recon = mimi.decoder(emb, state)

        T = min(recon.shape[-1], x.shape[-1])
        loss = multi_res_stft_loss(recon[..., :T], x[..., :T])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mimi.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))

        if step % 50 == 0:
            print(f"step {step:5d} | recon loss {np.mean(losses[-50:]):.4f} | {time.time() - t0:.0f}s")
        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"model": {f"mimi.{k}": v for k, v in mimi.state_dict().items()}, "step": step},
                       out_dir / "codec-checkpoint-latest.pt")
    print(f"تمام شد — چک‌پوینت کدک: {out_dir / 'codec-checkpoint-latest.pt'}")
    print("برای استفاده: مینی-state کدک را با training/export.py ترکیب کنید (mimi.* جایگزین می‌شود).")


if __name__ == "__main__":
    main()
