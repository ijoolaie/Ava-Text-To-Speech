# گزارش تست‌ها — فاز صفر و فاز ۱ (TEST REPORT)

## بخش فاز ۱ — استقلال هسته (2026-11-23)

### چه چیزی اجرا شد و چرا

مطابق §۶ مأموریت فاز ۱ فقط تست‌های مرتبط با تغییرات اجرا شد. تغییرات این فاز: `ava/__init__.py` (تنبل‌سازی)، `ava/audio/vocoder.py` (جدید)، `ava/runtime/backends.py` (اتصال به مرز vocoder)، `ava/audio/__init__.py`، `tests/test_core_independence.py` (جدید) و مستندات.

### تست‌های جدید — `tests/test_core_independence.py` (۱۳ تست در اجرای فاز ۱؛ پس از اصلاح closeout: ۱۴ تست — تست «سازگاری فرمت باندل مستندات↔کد» اضافه شد)

| الزام §۶ | تست |
| --- | --- |
| ۱) import هسته بدون Legacy | `test_native_core_imports_without_legacy` — پروسهٔ جدا با بلوکه‌سازی ۸ ماژول Legacy، import کامل هسته |
| ۲) Text Frontend فارسی | `test_text_frontend_contract` — قطعیت، idempotency، واژگان |
| ۳) قرارداد ورودی/خروجی مدل | `test_acoustic_model_io_contract` — shape `(80, F)`، قطعیت NAR |
| ۴) inference با وزن آزمایشی داخلی مجاز | `test_full_inference_with_internal_test_weights` — باندل fixture خود پروژه → WAV معتبر (RIFF) |
| ۵) Runtime بدون وابستگی Legacy | `test_runtime_synthesis_without_legacy_modules` — سنتز کامل در پروسهٔ بلوکه‌شده |
| ۶) API و CLI موجود | `test_cli_import_and_help_without_legacy` + اجرای `tests/test_platform.py` موجود |
| ۷) عدم import اجباری pocket-tts | `test_module_level_imports_of_native_path_are_clean` (پویش AST)، `test_ava_package_init_does_not_load_legacy`، `test_backends_legacy_imports_are_lazy_only`، `test_legacy_symbol_access_is_lazy` |
| — (تکمیلی) | `test_vocoder_boundary_contract`، `test_native_web_ui_is_self_contained`، `test_legacy_web_vendor_is_marked_and_licensed` |

### نتیجهٔ کلی

```
python3 -m pytest tests/ -q -p no:warnings   (ترتیب استاندارد الفبایی)
=> 128 passed (exit 0) — ۱۱۵ تست قبلی + ۱۳ تست استقلال؛ صفر ناموفق
```

- اجرای مرتبطِ میانی: `tests/test_core_independence.py + test_runtime.py + test_platform.py` → 39 سبز.
- **نکتهٔ صداقت:** در یک ترکیب‌بندی دستیِ غیرالفبایی (runtime→platform)، تست `TestOnnxExport::test_export_and_parity` یک‌بار ناموفق شد؛ همان تست به‌تنهایی و در ترتیب استاندارد کل مجموعه **سبز** است — تداخل ترتیب اجرا بین فایل‌ها، نه شکست ناشی از تغییرات فاز ۱ (کد export هیچ تغییری نکرده است).
- تولید صدای واقعی محصول ادعا نشده است؛ وزن‌های استفاده‌شده در تست‌ها fixture آزمایشیِ تولیدشده توسط خود پروژه‌اند (§۶ مأموریت رعایت شد).

### هنوز تست‌نشده (بدون تغییر نسبت به فاز صفر)

کیفیت گفتار واقعی (نیازمند آموزش فاز ۳)، دستگاه واقعی Windows/Android (فاز ۶)، API زیر بار هم‌زمانی واقعی. مانع بحرانی برای این فاز: **نداشت**.

---

## بخش closeout فاز ۱ — اصلاحات بازبینی (نسخهٔ 1.1.1 — 2026-11-23)

### چه چیزی اجرا شد و چرا

فقط تست‌های مرتبط با پنج اصلاح closeout (مطابق مأموریت closeout §تست و پذیرش). تغییرات: `ava/runtime/backends.py` (وضعیت مجوز صدا)، `tests/test_runtime.py` (ادعای تجاری → آزمایشی)، `pyproject.toml` (هویت بسته + نسخهٔ 1.1.1)، مستندات، و افزودن یک تست پایدار برای سازگاری مستندات باندل با کد.

### نتایج واقعی

```
python3 -m pytest tests/test_core_independence.py tests/test_runtime.py tests/test_platform.py
=> 37 passed, 3 skipped (exit 0) در 19.3s
```

- **۳ skip، همگی علت‌دار:** سه تست آداپتور Legacy در `test_runtime.py` (خطوط 138–151) با پیام `legacy weights/config not present (commercial-clean checkout)` — وزن‌های Legacy در این چک‌اوت عمداً موجود نیستند (مطابق محدودیت closeout: بازیابی ممنوع) و این تست‌ها به‌درستی skip می‌شوند، نه fail.
- **تست جدید** `test_bundle_format_matches_documentation` سبز: باندل واقعی دقیقاً `{ava_n1.pt, tokenizer.json, meta.json}` + `meta["format"] == "ava-native-n1"` + هر دو سند مستندات همین فرمت را توصیف می‌کنند و عبارت فرمت فرضی قدیمی در آن‌ها نیست.
- **اصلاح مجوز صدا** تأیید شد: `test_load_and_list_voices` (نسخهٔ جدید) → `commercial_ok=False` برای همهٔ صداهای Native + مجوز شامل «آزمایشی».
- **اعتبارسنجی metadata بسته** (`scripts/validate_pyproject.py`): ۱۲/۱۲ PASS — نسخهٔ 1.1.1، بدون pocket-tts/kyutai در metadata، گروه‌های واقعی serve/audio/dev، بدون گروه فرضی [api]، اسکریپت ava-cli.
- **سازگاری گروه نصب واقعی:** `pip install -e ".[serve]"` با exit 0 نصب شد (fastapi/uvicorn/python-multipart).
- **شمارش واقعی جمع‌آوری (`pytest --collect-only`):** کل مجموعه **۱۲۹ تست** (۱۱۵ قبلی + ۱۴ استقلال)؛ در اجرای تاریخی فاز ۱ همین مجموعه ۱۲۸ تست داشت (قبل از افزودن تست closeout) و ۱۲۸/۱۲۸ سبز بود.
- مجموعهٔ کامل در closeout بازاجرای نشد (طبق محدودیت «تست غیرمرتبط ممنوع»)؛ آخرین اجرای کامل معتبر: ۱۲۸/۱۲۸ در فاز ۱.

### خطاها و محدودیت‌های اجرا

- محیط توسعه بین دو پیام بازیافت (recycle) شد؛ مخزن از ZIP فاز ۱ (شامل تاریخچهٔ git) بازسازی و وابستگی‌ها نصب مجدد شدند — هیچ اثری روی محتوا نداشت.
- نصب اولیهٔ torch دو بار به دلیل کمبود فضای دیسک ناقص ماند؛ با نسخهٔ CPU-only (`torch 2.15.0+cpu`) حل شد.
- بدون وزن‌های Legacy، سه تست آداپتور Legacy قابل اجرا نیستند (skip صدق می‌کند) — همین وضعیت در چک‌اوت تجاری محصول طبیعی است.

---

## بخش فاز صفر — گواهی سلامت وضعیت موجود (2026-11-23)

### چه چیزی اجرا شد و چرا

پس از بازیابی مخزن از آرشیو (`Text-To-Speech-repo.zip`، کامیت `a1fb636`) و نصب مجدد وابستگی‌ها در محیط تازه، **یک بار کامل مجموعهٔ تست‌ها** اجرا شد تا «سلامت وضعیت موجود» که در `docs/ASSESSMENT_PHASE0.md` ادعا می‌شود، گواهی شود. طبق قانون پروژه، تست دیگری تکرار نشد؛ تغییرات این فاز صرفاً مستنداتی بوده‌اند و تست جداگانه‌ای نمی‌خواستند.

### نتیجه

```
python3 -m pytest tests/
=> 115 passed, 8 warnings in 72.18s
```

| مجموعه | فایل | تعداد |
| --- | --- | --- |
| موتور زبان فارسی (سازگاری Legacy + قابلیت‌های جدید) | `tests/test_language_engine.py` | 32 |
| معماری AVA-N1 (forward/backward، کاهش loss، استنتاج قطعی، WAV) | `tests/test_native_model.py` | 14 |
| پایپ‌لاین دیتاست (manifest، اعتبارسنجی، قطعه‌بندی، split) | `tests/test_dataset_pipeline.py` | 15 |
| آموزش (smoke واقعی، checkpoint/resume/export باندل) | `tests/test_native_training.py` | 4 |
| runtime (دو بک‌اند هم‌زمان، stream/stop/WAV واقعی) | `tests/test_runtime.py` | 14 |
| سیستم صداها (پروفایل، رضایت‌نامه، کیفیت prompt) | `tests/test_voice_system.py` | 7 |
| پلتفرم (API، CLI، export ONNX + parity) | `tests/test_platform.py` | 12 |
| امنیت و کیفیت (ورودی، path، secrets، سناریوهای فارسی) | `tests/test_security_quality.py` | 9 |
| تست‌های قدیمی Legacy (وزن‌ها، توکنایزر، آموزش قبلی) | `test_model_and_weights.py`, `test_tokenizer_and_chunking.py`, `test_training_smoke.py`, `test_train_real.py` | 8 |
| **جمع** | | **115 — همگی موفق** |

هیچ تستی ناموفق نبود. هشدارهای ۸گانهٔ ثبت‌شده صرفاً کتابخانه‌ای‌اند (librosa: n_fft بزرگ برای سیگنال کوتاه در تست‌های گراف‌لیم — بی‌ضرر).

## مواردی که هنوز تست نشده‌اند (شفاف‌سازی)

1. **کیفیت گفتار واقعی مدل اختصاصی** — وزن واقعی هنوز آموزش ندیده؛ شنیداری/MOS پس از فاز ۳ معنا دارد.
2. **اجرا روی دستگاه واقعی Windows و Android** — sandbox لینوکسی است؛ کد و export آماده، اجرای روی دستگاه تحویل‌نشده.
3. **API زیر هم‌زمانی/بار واقعی** — فقط با TestClient تست شده (تک‌کاربره).
4. **بنچمارک روی باندل واقعی** — اعداد ثبت‌شده (RTF 0.93 / legacy 0.22) مربوط به باندل smoke و محیط قبلی است؛ پس از آموزش واقعی مجدد اندازه‌گیری می‌شود.
5. **import دیتاست انتهابه‌انتها با دادهٔ واقعی** — به محض تأمین دادهٔ مجاز.

## مانع بحرانی؟

**نه برای این فاز.** مانع واحد مسیر جلو: تأمین دادهٔ مجاز + GPU (فاز ۲ و ۳). هیچ مانع فنی در کد، تست یا export شناسایی نشد.
