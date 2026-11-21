# راهنمای آموزش AVA-N1 (TRAINING GUIDE) — فاز ۴

## 1. اجرای سریع (smoke واقعی روی CPU — همین امروز اجرا و تأیید شده)

```bash
# دیتاست ساختگی (fixture؛ صراحتاً غیرگفتار واقعی)
python3 scripts/make_native_smoke_data.py data/smoke_native 20

# آموزش ۳۰۰ مرحله‌ای CPU (~۴ دقیقه روی ۲ هسته)
python3 -m training.train_native --manifest data/smoke_native/manifest.jsonl \
    --out runs/native_smoke_v1 --steps 300 --batch-size 8
# خروجی: runs/native_smoke_v1/{checkpoint.pt, log.jsonl, bundle/}
# resume: همان دستور + --resume
```

نتیجهٔ ثبت‌شده (runs/native_smoke_v1): loss از ~19.3 به ~0.35، dur_loss به ~0.04، validation در log.jsonl.

## 2. آموزش واقعی (نیازمند GPU و دیتاست مجاز)

| مورد | مقدار پیشنهادی v0 | توضیح |
| --- | --- | --- |
| داده | ≥ ۵ ساعت گفتار تمیز تک‌گوینده | فاز ۳ (manifest با منشأ/مجوز/رضایت) |
| GPU | ≥ 12GB (RTX 4090/A5000) | برآورد v0: ۲-۴ روز |
| d_model | 256 → 384 برای v1 | `AVANativeConfig` |
| batch | 32 (پویا با طول) | در sandbox: 8 روی CPU |
| lr | 2e-3 با warmup 500 | cosine decay |
| گام | ~150k تا 300k | تا plateau در val_loss |

```bash
python3 -m training.train_native --manifest data/real/manifest.jsonl \
    --out runs/native_v0 --steps 300000 --batch-size 32 --lr 2e-3 --resume
```

- **Resume**: از `checkpoint.pt` ادامه می‌یابد؛ TrainConfig فراخوانauthoritative است.
- **Validation**: هر `val_every` گام روی split=val لاگ می‌شود.
- **Log**: `log.jsonl` شامل step/loss/mel_l1/dur_loss/val_loss — تجسم با هر ابزار JSONL.

## 3. Duration alignment (نکتهٔ مهم)

v1 از طول یکنواخت وزن‌دار (علائم = مکث ۳×) استفاده می‌کند. برای کیفیت بالاتر، durations واقعی از Forced Alignment (مثلاً MFA با مدل فارسی) تولید و به‌عنوان `durations_path` به دیتاست اضافه شود — trainer بدون تغییر از آن استفاده می‌کند (ساختار pluggable).

## 4. Export

```bash
# باندل استنتاج (weights+tokenizer+config) — خودکار پایان آموزش
runs/native_v0/bundle/

# ONNX برای Windows/Android
python3 -m training.export.onnx_export --bundle runs/native_v0/bundle
# → bundle/ava_n1.onnx + گزارش parity/structural
```

## 5. قواعد

- مدل smoke/آزمایشی هرگز «مدل نهایی» معرفی نمی‌شود.
- هر آموزش با `dataset_fingerprint` ثبت می‌شود (تکرارپذیری).
- Loss تنها اثبات کیفیت نیست — ارزیابی صوتی الزامی است (MODEL_EVALUATION.md).
