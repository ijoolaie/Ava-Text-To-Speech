# ممیزی مجوزها (DEPENDENCY LICENSE AUDIT) — AVA Native

قانون طلایی: هیچ جزئی با مجوز ناسازگار با استفادهٔ تجاری، **به‌عنوان وابستگی محصول** پذیرفته نمی‌شود. اجزای دارای محدودیت فقط به‌صورت اختیاری، خارج از بستهٔ تجاری و با برچسب صریح قابل استفاده‌اند.

## 1. اجزای داخل مخزن

| جزء | فایل‌ها | مجوز | تجاری؟ | تعهد ما |
| --- | --- | --- | --- | --- |
| کد معماری Pocket-TTS (اقتباس Kyutai) | `ava/models/`, `ava/modules/` (به‌جز native) | MIT (© 2025 Kyutai) | ✅ | حفظ سرنویس MIT هر فایل + `NOTICE.md` |
| کد MLP (اقتباس MAR) | `ava/modules/mlp.py` | MIT | ✅ | همانند بالا |
| runtime مرورگر phonon-tts | `web/vendor/phonon-tts/` | Apache-2.0 / MIT | ✅ | vendored دست‌نخورده + `THIRD_PARTY_LICENSES.md` |
| کد اصلی AVA (موتور زبان، runtime، آموزش، تست‌ها) | بقیهٔ repo | MIT (خودِ ما) | ✅ | `LICENSE` |
| **چک‌پوینت وزن Legacy** | `weights/model*.safetensors`, `tokenizer.json`, `model_config.json` | **CC BY-NC 4.0** | ❌ | از ردیابی git حذف می‌شود؛ فقط آداپتور اختیاری خارج از بستهٔ تجاری؛ attribution الزامی حفظ می‌شود |
| **بردارهای صدای Legacy** | `weights/voices/*.safetensors` | CC BY-NC 4.0 (همان چک‌پوینت) | ❌ | مانند بالا |

نکته: CC BY-NC بازتوزیع با attribution را اجازه می‌دهد اما استفادهٔ تجاری را نه؛ بنابراین در بیلد تجاری این فایل‌ها وجود ندارند و در بیلد توسعه/مقایسه به‌صورت خارجی بارگذاری می‌شوند.

## 2. وابستگی‌های Python

| بسته | مجوز | تجاری؟ | یادداشت |
| --- | --- | --- | --- |
| torch | BSD-3-Clause | ✅ | |
| numpy | BSD-3-Clause | ✅ | |
| scipy | BSD-3-Clause | ✅ | |
| soundfile | BSD-3-Clause | ✅ | libsndfile هم‌بسته (LGPL-2.1+) — لینک داینامیک؛ متن LGPL در توزیع نهایی ضمیمه شود |
| sentencepiece | Apache-2.0 | ✅ | |
| safetensors | Apache-2.0 | ✅ | |
| librosa | ISC | ✅ | ارزیابی |
| fastapi | MIT | ✅ | |
| uvicorn | BSD-3-Clause | ✅ | |
| httpx | BSD-3-Clause | ✅ | تست API |
| onnx | MIT | ✅ | |
| onnxruntime | MIT | ✅ | مسیر Android/Windows |
| pytest | MIT | ✅ | فقط توسعه |
| PyYAML | MIT | ✅ | |

## 3. اجزای مرجع بیرونی (برای معماری اختصاصی)

| مرجع | مجوز | استفادهٔ ما |
| --- | --- | --- |
| Kyutai pocket-tts (کد) | MIT | آداپتور Legacy + الگوهای معماری؛ attribution کامل در NOTICE |
| HiFi-GAN (کد مرجع NVIDIA/official، MIT) | MIT | اگر از معماری vocoder مشابه استفاده شود، پیاده‌سازی خودمان با attribution؛ کد copy نمی‌شود |
| Mozilla Common Voice — fa | **CC0-1.0** | کاندید دیتاست v0 (بدون محدودیت؛ پیش از استفاده، شرایط صفحهٔ مجموعه بازبینی می‌شود) |
| ضبط‌های اختصاصی گوینده | قرارداد/رضایت‌نامه | مسیر اصلی صداهای اختصاصی؛ فاز ۳ |

## 4. چک‌لیست انطباق برای انتشار

- [x] `LICENSE` (MIT) برای کد مخزن
- [x] `NOTICE.md`: attribution Kyutai/MAR/phonon + منشأ وزن‌های Legacy (الزام CC BY-NC)
- [x] سرنویس MIT در فایل‌های اقتباسی
- [x] حذف وزن‌های NC از ردیابی git روی شاخهٔ AVA Native
- [ ] در توزیع باینری نهایی: ضمیمهٔ متن مجوزهای همهٔ وابستگی‌ها (`training/export/bundle_licenses.py` در فاز ۹ تولید می‌شود)
- [ ] پیش از انتشار تجاری: بازبینی نهایی توسط کارفرما
