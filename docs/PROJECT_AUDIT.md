# ممیزی کامل پروژه (PROJECT AUDIT) — فاز صفر AVA Native

تاریخ ممیزی: 2026-11-21 · شاخه: `feature/ava-native` · کامیت مبدأ: `5822ade`

این سند وضعیت واقعی تک‌تک اجزای مخزن را پیش از بازسازی ثبت می‌کند. برچسب‌ها:
`KEEP` (نگه‌داری) · `REFACTOR` (بازطراحی) · `REPLACE` (جایگزینی) · `REMOVE` (حذف) · `INVESTIGATE` (بررسی بیشتر).

## 1. اجزای کد

| جزء | مسیر | وضعیت | توضیح |
| --- | --- | --- | --- |
| معماری استنتاج Pocket-TTS (اقتباس MIT از Kyutai) | `ava/models/flow_lm.py`, `ava/models/mimi.py`, `ava/modules/*` | KEEP (به‌عنوان **آداپتور Legacy**) | کد MIT است؛ استفاده تجاری آزاد است. در AVA Native پشت رابط پلاگین مدل قرار می‌گیرد، نه هسته محصول. سرنویس‌های MIT حفظ می‌شود. |
| واسط مدل TTS | `ava/models/tts_model.py`, `ava/models/model_state.py` | REFACTOR | به رابط `IModel` در `ava/inference/` منتقل می‌شود؛ بارگذاری صداها و مسیرها پارامتریک می‌شود. |
| نرمال‌سازی فارسی | `ava/text_norm.py` | REFACTOR | به موتور ماژولار `ava/language/` (فاز ۱) تبدیل می‌شود؛ shim سازگاری باقی می‌ماند تا وزن‌ها و تست‌های فعلی کار کنند. |
| تقسیم متن طولانی | `ava/models/text_chunking.py` | REFACTOR | به `ava/language/text_chunking.py` منتقل و توسعه داده می‌شود (تقسیم جمله‌محور با شمارش توکن). |
| ورودی/خروجی وزن‌های shard شده | `ava/utils/weights_io.py`, `weights_loading.py`, `utils/config.py` | KEEP | کد اصلی ما، تست‌شده، مستقل از فرمت. |
| ابزار صوت | `ava/data/audio_utils.py` | REFACTOR | به `ava/audio/` منتقل می‌شود (خواندن/نوشتن WAV، resample، اندازه‌گیری). |
| پایپ‌لاین آموزش فرمت Legacy | `training/` | KEEP | صحتش با smoke training واقعی تأیید شده؛ برای آموزش مدل اختصاصی جدید (فاز ۴) از آن الگو می‌گیریم و trainer مستقل `native_trainer.py` اضافه می‌شود. |
| اسکریپت‌های CLI | `scripts/*.py` | REFACTOR | به `ava-cli` (entry point واحد) با زیرفرمان‌ها مهاجرت می‌کنند؛ اسکریپت‌ها سازگار باقی می‌مانند. |
| دموی مرورگر (WASM) | `web/` + `web/vendor/phonon-tts/` | KEEP (تاریخی) | مسیر Legacy بدون سرور؛ برای مدل اختصاصی، UI جدید `web-native/` روی API ساخته می‌شود. |
| CI | `.github/workflows/ci.yml` | REFACTOR | باید تست‌های وابسته به وزن‌های خارج‌شده از git را با skip شرطی مدیریت کند. |
| نمونه‌های صوتی | `samples/*.wav` | KEEP | تولید خودِ پروژه (MIT). |

## 2. مدل‌ها، وزن‌ها و داده

| جزء | وضعیت | تصمیم برای AVA Native |
| --- | --- | --- |
| چک‌پوینت ۱۸.۲M فارسی (`weights/model*.safetensors`, `tokenizer.json`, `model_config.json`) | REPLACE (به‌عنوان مدل محصول) / KEEP (به‌عنوان مرجع مقایسه) | مجوز CC BY-NC 4.0 — استفاده تجاری ندارد. از git حذف‌ردیابی می‌شود؛ فقط از طریق آداپتور اختیاری Legacy (خارج از بستهٔ تجاری) قابل استفاده است. |
| بردارهای صدا (`weights/voices/*.safetensors`: nava, shirin, arash) | REPLACE | مانند وزن‌ها؛ صداهای اختصاصی باید با دادهٔ مجاز خودمان ساخته شوند (فاز ۶). |
| دادهٔ smoke (`data_cache/`) | KEEP (gitignored) | برای تست پایپ‌لاین تولید می‌شود، در git نیست. |
| نمونهٔ دیتاست واقعی | REMOVE (فعلاً خارج از repo) | تهیه دادهٔ مجاز فارسی = فاز ۳؛ منابع مجاز مستند می‌شود. |

## 3. معماری مدل فعلی (منشأ کد)

- **منشأ اقتباس**: Kyutai pocket-tts v4.0.1 (MIT © 2025 Kyutai) — معماری Flow-LM یک‌مرحله‌ای روی latent کدک Mimi با شرط‌سازی صوتی، plus تفکیک encoder/decoder. تغییرات ما: تغییر نام importها، بارگذاری فقط-محلی، پشتیبانی وزن shard، حذف دانلود HF/سرور/کوانتیزاسیون. این تغییرات در `NOTICE.md` ثبت و به‌صورت diff قابل بازبینی در تاریخ git است.
- **MLP** اقتباسی از MAR (MIT).
- **runtime مرورگر**: phonon-tts 0.7.1 (Apache-2.0/MIT) — vendor نشده تغییریافته، فقط vendored.
- نتیجهٔ ممیزی: کد اقتباسی همگی MIT/Apache (سازگار با تجاری)، اما **وزن‌ها** محدودیت تجاری دارند؛ بنابراین مسیر استقلال محصول = معماری اختصاصی + آموزش روی دادهٔ مجاز (فاز ۲ و ۴).

## 4. وابستگی‌های نرم‌افزاری

| وابستگی | مجوز | سازگار تجاری | یادداشت |
| --- | --- | --- | --- |
| torch (CPU) | BSD-3 | ✅ | هسته آموزش/استنتاج |
| numpy / scipy | BSD-3 | ✅ | |
| soundfile | BSD-3 | ✅ | libsndfile داخل wheel از نوع LGPL-2.1+ است — لینک داینامیک با رعایت NOTICE مجاز است؛ در سند مجوزها ثبت شد |
| sentencepiece | Apache-2.0 | ✅ | توکنایز آموزشی |
| safetensors | Apache-2.0 | ✅ | |
| librosa | ISC | ✅ | فقط ابزارهای ارزیابی/آنالیز |
| fastapi / uvicorn / httpx | MIT / BSD-3 | ✅ | فاز ۷ |
| onnx / onnxruntime | MIT | ✅ | مسیر Android/Windows |
| pytest / PyYAML | MIT | ✅ | |
| phonon-tts (vendor) | Apache-2.0/MIT | ✅ | فقط دموی Legacy مرورگر |

سند کامل: `docs/DEPENDENCY_LICENSE_AUDIT.md`.

## 5. تاریخچهٔ Git و منشأ

- مخزن یک کامیت پایه دارد (`5822ade`) که کل کد را یک‌جا ثبت کرده؛ تاریخچهٔ پیش از آن در sandbox قبلی از بین رفته اما منشأ اقتباس‌ها در سرنویس فایل‌ها + `NOTICE.md` + این سند مستند است. از این پس commits کوچک و هدفمند بر اساس فازها ثبت می‌شود.
- شاخهٔ توسعه: `feature/ava-native`. merge به `main` فقط با اجازهٔ کارفرما.

## 6. جمع‌بندی و موانع واقعی

1. **موانع محیطی**: بدون GPU؛ آموزش واقعی تولیدی در sandbox ممکن نیست (فقط smoke آموزش روی CPU واقعاً اجرا می‌شود). بدون اعتبارنامهٔ GitHub؛ push نیازمند مجوز/توکن کارفرماست.
2. **نتیجهٔ فاز صفر**: مسیر بازسازی — موتور زبان اختصاصی (فاز ۱)، معماری عصبی اختصاصی (فاز ۲)، پایپ‌لاین داده (فاز ۳)، آموزش (فاز ۴)، runtime/API/CLI/Web (فاز ۵ و ۷)، صداها (فاز ۶)، قرارداد Raya (فاز ۸)، امنیت/کیفیت (فاز ۹).
