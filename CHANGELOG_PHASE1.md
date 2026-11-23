# گزارش تغییرات فاز ۱ — Independent Native Core (CHANGELOG)

- تاریخ: 2026-11-23
- نسخهٔ پایه: **AVA_Phase-0_Roadmap_v1.0.0** (کامیت مبنای شروع: `6bf7930`)
- شاخه: `feature/ava-native` · نسخهٔ جدید پروژه: **1.1.0** (`pyproject.toml`)
- دامنه: استقلال هستهٔ Native و مرزبندی Legacy — بدون بازنویسی معماری، بدون آموزش، بدون قابلیت جدید

## ۱. هدف و نتیجه

مسیر اجرای AVA Native اکنون **کاملاً از Legacy (استک اقتباسی pocket-tts) مستقل است**: import ماژول‌های Native هیچ کد اقتباسی را بار نمی‌کند، Runtime بدون Legacy سنتز می‌کند، و وضعیت هر جزء اقتباسی به‌همراه جایگزین آن مستند شده است. قراردادهای API/CLI/ورودی-خروجی بدون تغییر ماندند.

## ۲. فایل‌های ایجادشده

| فایل | محتوا |
| --- | --- |
| `ava/audio/vocoder.py` | مرز مستقل Vocoder: کلاس انتزاعی `Vocoder`، پیاده‌سازی موقت `GriffinLimVocoder` (v1، بدون وزن)، رجیستری `get_vocoder` برای vocoder عصبی آینده |
| `tests/test_core_independence.py` | ۱۳ تست استقلال: import هسته در پروسهٔ بلوکه‌شدهٔ Legacy، پویش AST سطح ماژول مسیر Native، قرارداد frontend/model/vocoder، سنتز کامل بدون Legacy، CLI بدون Legacy، ایزوله‌سازی وب Legacy |
| `docs/CORE_INDEPENDENCE_AUDIT.md` | کلاسه‌بندی کامل A/B/C/D همهٔ اجزا + قرارداد و جایگزین موردنیاز هر جزء اقتباسی |
| `docs/NATIVE_CONTRACTS.md` | قرارداد پایدار چهار مرز: Text Frontend / Acoustic Model / Vocoder / Runtime |
| `docs/RUNTIME_INSTALL.md` | نصب مسیر Native (بدون وابستگی Legacy)، باندل مدل، CLI/API/وب، خطاهای نبود وزن، محدودیت‌های فعلی |
| `CHANGELOG_PHASE1.md` | همین سند |

## ۳. فایل‌های تغییرکرده

| فایل | تغییر | دلیل |
| --- | --- | --- |
| `ava/__init__.py` | **اصلاح بحرانی استقلال:** importهای سطح ماژول `TTSModel`/`export_model_state` (استک Legacy) → دسترسی تنبل PEP 562 | قبلاً هر `import ava.*` کد اقتباسی pocket-tts را بار می‌کرد؛ سازگاری اسکریپت‌های Legacy حفظ شد |
| `ava/runtime/backends.py` | `NativeN1Backend` از مرز `get_vocoder()` استفاده می‌کند (به‌جای فراخوانی مستقیم griffin-lim) | جایگزینی vocoder آینده بدون تغییر قرارداد؛ رفتار خروجی یکسان |
| `ava/audio/__init__.py` | export نمادهای vocoder | دسترس‌پذیری مرز |
| `pyproject.toml` | `version = "1.1.0"` | نسخه‌بندی فاز ۱ |
| `docs/ROADMAP.md` | فاز ۱ → 100% ✅ + جمع موزون ≈ 51% + ثبت تکمیل‌ها | ثبت رسمی |
| `PROJECT_STATUS.md` | ردیف فاز ۱ → 100% + دستاورد فاز ۱ | ثبت رسمی |
| `NOTICE.md` | بخش «Phase 1 independence note» | شفاف‌سازی منشأ کد مسیر محصول |
| `docs/DEPENDENCY_LICENSE_AUDIT.md` | بخش 1.1 وضعیت استقلال اجزا (مجاز/خارج از محصول) | §۵ مأموریت |
| `docs/NEURAL_ARCHITECTURE.md` | جدول vocoder: v1 موقت پشت مرز مستقل | شفاف‌سازی |
| `docs/TEST_REPORT.md` | بخش نتایج فاز ۱ (۱۲۸/۱۲۸) | §۶ مأموریت |

## ۴. تصمیم‌های فنی این فاز

1. **PEP 562 lazy import** برای نمادهای Legacy در `ava/__init__.py` — حداقل‌ترین تغییر ممکن که هم استقلال کامل می‌دهد و هم سازگاری اسکریپت‌های مقایسه/ارزیابی Legacy را نگه می‌دارد (قانون §۱-۵: حذف ناگهانی ممنوع).
2. **مرز Vocoder به‌جای vocoder عصبی** — طبق محدودیت مأموریت، وارد پیاده‌سازی vocoder عصبی نشدیم؛ فقط مرز مستقل آن ثبت شد تا در فاز ۳ بدون دست‌زدن به مدل/runtime بنشیند.
3. **حذف فیزیکی فایل‌های Legacy/وندور موکول به فاز ۶** — کد و تاریخچه حفظ شد؛ در بستهٔ محصول هرگز وارد نمی‌شوند و از مسیر اجرا جدا شده‌اند.
4. **پروفایل بیلد محصول** به فاز ۶ (بسته‌بندی) منتقل شد — طبیعی‌ترین جای آن.

## ۵. چیزهایی که تغییر نکردند (حفظ قراردادها)

API (`/v1/synthesis`، `/v1/synthesis/stream`، voices/models/health)، CLI (`ava-cli`)، فرمت WAV 24kHz mono PCM-16، معماری AVA-N1، قرارداد باندل استنتاج، رفتار موتور زبان (تست طلایی ثابت) — همگی بدون تغییر.
