# پایپ‌لاین دیتاست فارسی (DATASET PIPELINE) — فاز ۳

## 1. ساختار داده

```
data/
  raw/            فایل‌های صوتی خام + متن‌های متناظر (در git نیست)
  manifests/      manifest.jsonl (نسخه‌بندی‌شده)
  reports/        گزارش اعتبارسنجی و منشأ
  processed/      خروجی آماده برای trainer
```

هر جمله = یک خط JSON در `manifest.jsonl` با schema کاملاً تعریف‌شده (`training/datasets/manifest.py`):

| فیلد | توضیح | الزامی |
| --- | --- | --- |
| `id` | شناسه یکتا | ✓ |
| `audio_path` | مسیر WAV | ✓ |
| `text` | متن مرجع خام | ✓ |
| `speaker_id` | شناسه گوینده | ✓ |
| `sample_rate/duration_s/channels` | توسط validate پر می‌شود | - |
| `language` | پیش‌فرض `fa` | ✓ |
| `source` | منشأ (مجموعه یا ضبط اختصاصی) | ✓ |
| `license` | مجوز همین قطعه صوت | ✓ |
| `consent_ref` | مرجع رضایت‌نامهٔ گوینده | ✓ برای ضبط اختصاصی |
| `split/valid/issues` | خروجی split/validate | - |

قانون سخت: **هیچ قطعه‌ای بدون `source` و `license` وارد آموزش نمی‌شود**؛ گزارش `provenance_report()` تعداد قطعه‌های بدون رضایت/نامعتبر را صریح نشان می‌دهد.

## 2. ابزارها (تست‌شده با ۱۵ تست)

| ابزار | فایل | کار |
| --- | --- | --- |
| Manifest | `training/datasets/manifest.py` | خواندن/نوشتن JSONL، خطای فرمت، fingerprint نسخه، گزارش منشأ |
| اعتبارسنجی | `training/datasets/validate.py` | فرمت، SR، کانال، طول، سکوت، SNR (برآورد انرژی‌محور)، کلیپینگ |
| قطعه‌بندی | `training/datasets/segment.py` | VAD-lite انرژی‌محور؛ قطعه‌های طولانی در کم‌انرژی‌ترین نقطه بریده می‌شوند |
| تقسیم | `training/datasets/split.py` | قطعی (hash-based)؛ حالت speaker-disjoint برای سنجش تعمیم واقعی |

## 3. منابع دادهٔ مجاز (برای آموزش واقعی)

| منبع | مجوز | وضعیت |
| --- | --- | --- |
| ضبط اختصاصی گوینده با رضایت‌نامه کتبی | متعلق به ما | مسیر اصلی صداهای اختصاصی؛ فرم رضایت: `data/raw/consents/` |
| Mozilla Common Voice — fa | **CC0-1.0** | کاندید v0 (پیش از استفاده، صفحهٔ مجموعه بازبینی شود؛ کیفیت متن آن بررسی شود) |
| هر منبع دیگر | باید بررسی شود | بدون مجوز شفاف، ممنوع |

نیاز حجمی برای v0 قابل دفاع: ≥ ۵ ساعت گفتار تمیز تک‌گوینده (v0)، ≥ ۲۰ ساعت چندگوینده (v1)، ≥ ۱۰۰ ساعت (کیفیت تولیدی). هر ساعت داده ≈ ۸۵۰ جملهٔ میانگین ۴ ثانیه‌ای.

## 4. جریان کاری استاندارد

```bash
# 1) وارد کردن ضبط خام و قطعه‌بندی
python3 -m training.datasets.segment data/raw/session01.wav > segments.json

# 2) ساخت manifest اولیه + متن‌ها
#    (اسکریپت data/import_session.py در فاز ۴ به CLI متصل می‌شود)

# 3) اعتبارسنجی کامل
python3 -c "from training.datasets import *; ..."   # یا ava-cli validate-dataset

# 4) تقسیم قطعی
assign_splits(entries, seed="<نسخه>", speaker_disjoint=True)

# 5) ثبت نسخه
dataset_fingerprint(entries)  # هویت نسخهٔ دیتاست در گزارش آموزش
```

## 5. حریم خصوصی

- Metadata فقط شامل شناسه گویندهٔ مستعار است؛ نام/سن/آدرس هرگز وارد manifest نمی‌شود.
- فایل‌های رضایت‌نامه خارج از repo نگهداری می‌شوند و فقط `consent_ref` ثبت می‌شود.
- صدای گویندگان فقط روی دیتاست آموزشی می‌ماند؛ خروجی مدل، صدای جدید تولید می‌کند نه کپی گوینده (بدون رضایت، voice-cloning فعال نمی‌شود — فاز ۶).
