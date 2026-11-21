# -*- coding: utf-8 -*-
"""آماده‌سازی داده برای آموزش flow LM — مانیفست → کش لاتنت‌ها.

مانیفست: فایل CSV/TSV با سرستون یا بدون سرستون، دو ستون اول: مسیر صوت، متن.

    python -m training.data.prepare_data \
        --manifest data/my_fa_dataset.csv \
        --arch config/fa.yaml \
        --out data_cache/ava-fa

برای هر کلیپ: بارگذاری → ۲۴kHz mono → انکد با Mimi منجمد → ذخیرهٔ لاتنت خام
[T, 32] به‌همراه متن نرمال‌شده. آمارهٔ mean/std کل دیتاست هم در stats.json
ذخیره می‌شود (مقدار اولیهٔ بافرهای emb_mean/emb_std آموزش).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ava.text_norm import normalize  # noqa: E402


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        first_line = sample.splitlines()[0] if sample else ""
        dialect = csv.excel
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        sniff = csv.Sniffer().sniff(sample, delimiters=",\t") if "," in first_line or "\t" in first_line else None
        reader = csv.reader(f, dialect, delimiter=delim) if sniff is None else csv.reader(f, sniff)
        for i, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            audio_path, text = row[0].strip(), row[1].strip()
            if i == 0 and audio_path.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")) is False:
                continue  # header row
            rows.append((audio_path, text))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="آماده‌سازی کش لاتنت برای آموزش")
    ap.add_argument("--manifest", required=True, help="CSV/TSV با ستون‌های (مسیر صوت، متن)")
    ap.add_argument("--arch", default=str(REPO_ROOT / "config" / "fa.yaml"),
                    help="پیکربندی مدل (کدک منجمد از weights_path همان فایل)")
    ap.add_argument("--out", default="data_cache/ava-fa", help="پوشهٔ خروجی کش")
    ap.add_argument("--no-normalize", action="store_true", help="بدون نرمال‌سازی متن فارسی")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from training.data.latents import LatentEncoder

    manifest = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    print(f"{len(rows)} ردیف در مانیفست {manifest}")

    enc = LatentEncoder(args.arch, device=args.device)
    items, skipped, all_latents = [], [], []
    for i, (audio_path, text) in enumerate(rows):
        src = Path(audio_path)
        if not src.exists():
            alt = manifest.parent / src
            src = alt if alt.exists() else src
        if not src.exists():
            skipped.append((str(audio_path), "missing file"))
            continue
        text = text.strip()
        if not text:
            skipped.append((str(audio_path), "empty text"))
            continue
        try:
            latents = enc.encode_file(src)
        except Exception as e:  # noqa: BLE001 — keep going on corrupt files
            skipped.append((str(audio_path), f"encode error: {e}"))
            continue
        if latents.shape[0] < 4:
            skipped.append((str(audio_path), "too short"))
            continue
        norm_text = text if args.no_normalize else normalize(text)
        stem = f"item-{i:06d}"
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            latents=latents.astype(np.float32),
            text=norm_text,
            raw_text=text,
            source=str(src),
        )
        items.append({
            "stem": stem,
            "frames": int(latents.shape[0]),
            "text": norm_text,
            "source": str(src),
        })
        all_latents.append(latents)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} …")

    if all_latents:
        stacked = np.concatenate(all_latents, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = stacked.std(axis=0).astype(np.float32)
        std = np.maximum(std, 1e-4)
    else:
        mean, std = np.zeros(32, np.float32), np.ones(32, np.float32)

    stats = {
        "n_items": len(items),
        "n_skipped": len(skipped),
        "total_frames": int(sum(it["frames"] for it in items)),
        "latent_mean": mean.tolist(),
        "latent_std": std.tolist(),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "items.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    if skipped:
        (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")

    hours = stats["total_frames"] / 12.5 / 3600
    print(f"تمام شد: {stats['n_items']} کلیپ ({hours:.3f} ساعت)، {stats['n_skipped']} رد شده")
    print(f"کش: {out_dir} — stats: mean[min {mean.min():.2f} max {mean.max():.2f}], "
          f"std[min {std.min():.2f} max {std.max():.2f}]")


if __name__ == "__main__":
    main()
