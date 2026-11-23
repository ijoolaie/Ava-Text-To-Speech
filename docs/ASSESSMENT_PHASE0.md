# ارزیابی فاز صفر — وضعیت واقعی پروژه AVA (ASSESSMENT)

تاریخ: 2026-11-23 · شاخه: `feature/ava-native` · آخرین کامیت مبدأ: `a1fb636`
مبنای ارزیابی: کد واقعی مخزن + اجرای واقعی تست‌ها (docs/TEST_REPORT.md). هیچ قابلیتی بدون اثبات اجرایی «پیاده‌سازی‌شده» شمرده نشده است.

راهنما: ✅ پیاده‌سازی‌شده و تست‌شده · 🟡 پیاده‌سازی جزئی/اولیه · ❌ انجام‌نشده

## 1. معماری AVA Native

- وضعیت: ✅ طراحی و پیاده‌سازی کامل + تست
- فایل‌ها: `docs/NEURAL_ARCHITECTURE.md` (طراحی + مقایسهٔ گزینه‌ها)، `ava/models/native/` (model/config/tokenizer)، `tests/test_native_model.py`
- جزئیات: AVA-N1 غیرخودبازگشتی؛ encoder→duration→decoder→80-mel در 24kHz؛ speaker-embedding برای چندصدایی؛ استنتاج قطعی (NAR)
- کمبودها: vocoder عصبی (v2)، streaming فریم‌محور داخل مدل، pitch/energy adaptor

## 2. پردازش و نرمال‌سازی متن فارسی

- وضعیت: ✅ کامل و ماژولار + تست گسترده (۴۴ تست، شامل خروجی طلاییِ رفتار)
- فایل‌ها: `ava/language/` — chars/numbers/datetime_speech/language_detect/pronunciation/segmentation/prosody/pipeline/token_counter
- پوشش: عربی→فارسی، نویسه‌های نامرئی، نیم‌فاصله، اعداد/ترتیبی/اعشاری، تاریخ شمسی و میلادی (تبدیل تقویم رفت‌وبرگشت تست‌شده)، ساعت/دقیقه/ثانیه، تلفن، واحدها و ارز، متن ترکیبی (سه سیاست)، جمله‌بندی، جدول مکث
- کمبودها: متن محاوره‌ای (محاوره‌نویسی→رسمی)، اعراب‌گذاری، واژگان تلفظ بزرگ‌تر، اختصارات فارسی بیشتر

## 3. تبدیل متن به واج یا توکن

- وضعیت: 🟡 توکن کاراکتری کامل؛ واج‌سازی (G2P) واقعی انجام‌نشده
- فایل‌ها: `ava/models/native/tokenizer.py` (الفبای کاراکتری + specials)، `ava/language/pronunciation.py` (واژگان تلفظ + قواعد)
- توضیح: مدل AVA-N1 به توکن کاراکتری آموزش می‌بیند؛ تلفظ از مسیر lexicon+قواعد انجام می‌شود نه G2P واج‌محور
- کمبودها: نگاشت حرف→واج فارسی، تنش/استرس، G2P برای وام‌واژه‌ها

## 4. دیتاست و ابزار آماده‌سازی داده

- وضعیت: ✅ ابزارها کامل + تست؛ 🟡 خود داده: صفر ساعت واقعی
- فایل‌ها: `training/datasets/` (manifest/validate/segment/split)، `scripts/make_native_smoke_data.py`، `docs/DATASET_PIPELINE.md`
- پوشش: JSONL با منشأ/مجوز/رضایت، اعتبارسنجی صوت (SR/کانال/طول/سکوت/SNR/کلیپینگ)، قطعه‌بندی انرژی‌محور، split قطعی speaker-disjoint، fingerprint نسخه
- کمبودها: acquisition واقعی (ضبط اختصاصی یا Common Voice-fa CC0)، اتصال import به CLI

## 5. هم‌ترازی متن و صوت

- وضعیت: ❌ انجام‌نشده (fallback یکنواخت وزن‌دار با مکث ۳× برای علائم)
- فایل‌ها: `training/train_native.py::NativeDataset._duration_targets`
- کمبودها: اتصال MFA (مدل فارسی) یا ASR-based alignment؛ ساختار trainer از الان pluggable است و بدون تغییر می‌پذیرد

## 6. مدل و معماری شبکه عصبی

- وضعیت: ✅ کد معماری کامل + تست (۱۴ تست)؛ ❌ **وزن واقعی فارسی: ندارد**
- فایل‌ها: `ava/models/native/model.py`، `tests/test_native_model.py`
- واقعیت آموزش: smoke ۳۰۰ گام CPU روی دادهٔ ساختگی tone-burst (loss 19.32→0.141، val 0.196) — اثبات پایپ‌لاین، نه گفتار واقعی
- کمبودها: آموزش واقعی روی GPU با دادهٔ مجاز (نیاز: ≥۵ ساعت داده v0)

## 7. آموزش و ذخیره وزن‌ها

- وضعیت: ✅ کامل + تست (۴ تست آموزش شامل resume واقعی)
- فایل‌ها: `training/train_native.py` (train/checkpoint/resume/export/load_bundle)، `ava/utils/weights_io.py` (sharded I/O)
- پوشش: checkpoint کامل مدل+optimizer+config، bundle استنتاج (weights+tokenizer+config+meta)
- کمبودها: experiment tracking خارجی (اختیاری)، EMA

## 8. Vocoder (تولید موج صوتی)

- وضعیت: 🟡 v1 Griffin-Lim اجرا و تست‌شده (کیفیت صحت‌سنجی)؛ ❌ vocoder عصبی
- فایل‌ها: `ava/audio/mel.py` (mel filterbank، griffin-lim، librosa/ISC)
- کمبودها: vocoder عصبی سبک (طراحی در NEURAL_ARCHITECTURE §5 ثبت شده) — برای کیفیت شنیداری واقعی لازم است

## 9. Runtime و استنتاج

- وضعیت: ✅ کامل + تست (۱۴ تست)
- فایل‌ها: `ava/runtime/` (engine/backends)، `tests/test_runtime.py`
- پوشش: load/unload، synthesize/synthesize_stream/batch، stop با event per-request، WAV 24kHz mono PCM-16، رجیستری فرمت‌مدل (دو معماری هم‌زمان تست شده)، RTF 0.93 اندازه‌گیری‌شده
- کمبودها: بهینه‌سازی int8/کوانتیزاسیون، پروفایل حافظه روی موبایل، streaming فریم‌محور

## 10. مدیریت صداهای مختلف

- وضعیت: ✅ زیرساخت کامل + تست؛ ❌ صدای مستقل واقعی: صفر
- فایل‌ها: `ava/voices/` (registry/quality)، `tests/test_voice_system.py`
- پوشش: VoiceProfile با الزام رضایت‌نامه (کلون بدون consent رد می‌شود)، سنجش کیفیت prompt، صدا در runtime با مالکیت بک‌اند
- کمبودها: ≥۳ صدای اختصاصی — مشروط به دادهٔ فاز ۲؛ صداهای فعلی legacy و غیرتجاری‌اند

## 11. API و CLI

- وضعیت: ✅ کامل + تست (۱۲ تست)
- فایل‌ها: `ava/api/server.py`، `ava/cli.py` (+ console script در pyproject)، `web-native/index.html`
- پوشش: /v1/synthesis (WAV)، /v1/synthesis/stream (NDJSON)، voices/models/health، محدودیت اندازه/سرعت، خطاهای تمیز، وب فارسی RTL با پخش/دانلود/توقف
- کمبودها: احراز هویت (سند Raya صراحتاً reverse-proxy با توکن را توصیه کرده)، rate limiting، WebSocket

## 12. پشتیبانی Windows

- وضعیت: 🟡 کد cross-platform + راهنما + launcher؛ ❌ آزمون روی دستگاه واقعی
- فایل‌ها: `desktop/windows/README.md`، `scripts/windows/run_ava.bat`، `ava/cli.py` (pathlib/UTF-8 صریح)
- کمبودها: نصب و اجرای واقعی ویندوز، بستهٔ نصبی (installer)

## 13. پشتیبانی Android

- وضعیت: 🟡 مسیر فنی اثبات‌شده تا سطح export؛ ❌ اجرا روی دستگاه
- فایل‌ها: `training/export/onnx_export.py` (export + parity تست‌شده)، `android/README.md`
- پوشش: ONNX opset-17 static-shape export با صحت‌سنجی دوگانه (parity با durations ثابت + structural)
- کمبودها: روی-device benchmark، mel→PCM در JNI، مدیریت دانلود مدل، نمونهٔ app

## 14. وابستگی‌ها و مجوزها

- وضعیت: ✅ audit کامل و ثبت‌شده
- فایل‌ها: `docs/DEPENDENCY_LICENSE_AUDIT.md`، `NOTICE.md`، `LICENSE`، سرنویس‌های MIT در فایل‌های اقتباسی
- وضعیت: وابستگی‌های همگی BSD/MIT/Apache؛ libsndfile (LGPL) با لینک داینامیک ثبت شده
- تصمیم باز برای کارفرما: وزن‌های Legacy (CC BY-NC، 108MB) در git history مخزن هستند — برای بستهٔ تجاری باید خارج شوند؛ حذف از history یا untrack فقط با تصمیم کارفرما (قانون: بدون دلیل واقعی تاریخچه بازنویسی نمی‌شود)

## قابلیت‌های واقعاً پیاده‌سازی‌شده (خلاصهٔ قابل دفاع)

1. موتور زبان فارسی ماژولار با پوشش گسترده سناریو — مستقل و تست‌شده
2. معماری عصبی اختصاصی AVA-N1 با کد اصل و تست — **بدون وزن واقعی**
3. پایپ‌لاین دیتاست کامل با انطباق منشأ/مجوز/رضایت — **بدون دادهٔ واقعی**
4. trainer با checkpoint/resume/export — اثبات‌شده با smoke واقعی
5. runtime پلاگین‌پذیر + API + CLI + وب — تست‌شده؛ صدای واقعی فعلاً فقط از بک‌اند غیرتجاری legacy
6. export ONNX با صحت‌سنجی — اثبات‌شده روی CPU
7. vocoder v1 (Griffin-Lim) — برای صحت‌سنجی؛ کیفیت نهایی نیازمند vocoder عصبی

## جمع‌بندی

هستهٔ مستقل (زبان+معماری+دیتاست+آموزش+runtime) به‌صورت کدِ واقعی و تست‌شده موجود است؛ **حلقهٔ مفقود مرکزی، آموزش وزن واقعی روی دادهٔ مجاز است** — پس از آن قابلیت‌های runtime/API/صدا بلافاصله «مستقل و واقعی» می‌شوند، چون مسیرشان از قبل از مدل پلاگین‌پذیر عبور می‌کند.
