# PHASE1_CLOSEOUT — اصلاحات بازبینی فاز ۱

- تاریخ: 2026-11-23 · نسخه: **1.1.1** (`pyproject.toml`)
- مبنا: تحویل `AVA_Phase-1_Independent-Core_v1.1.0.zip` (کامیت `3e0981e`) · شاخه: `feature/ava-native`
- دامنه: رفع ناهماهنگی‌های مشخص‌شده در بازبینی فاز ۱ — بدون توسعه قابلیت، بدون تغییر معماری، بدون دست‌زدن به تاریخچهٔ Git

## ۱. موارد اصلاح‌شده

### ۱) فرمت باندل — مستندات اکنون با کد واقعی هماهنگ است
مبنا: `training/train_native.py` (کد تغییر نکرد).
- `export_native_bundle()` تولید می‌کند: **`ava_n1.pt`** (`torch.save({"model": state_dict, "config": dict(cfg)})` — چک‌پوینت Torch، نه safetensors) + **`tokenizer.json`** (`{"type": "ava-n1-char", "vocab": [...]}`) + **`meta.json`** (`{"format": "ava-native-n1", "params_m", "sample_rate", "mel": {"n_mels", "hop"}}`).
- `load_native_bundle()` همان `ava_n1.pt` را با `torch.load(map_location="cpu")` بار می‌کند.
- فرمت فرضی `weights.safetensors + config.json` از `docs/NATIVE_CONTRACTS.md` و `docs/RUNTIME_INSTALL.md` حذف و فرمت واقعی جدول‌وار ثبت شد.
- تثبیت پایدار: تست جدید `test_bundle_format_matches_documentation` — باندل واقعی دقیقاً همین سه فایل است و مستندات باید همین را توصیف کنند.

### ۲) دستور نصب API — با optional dependencyهای واقعی سازگار شد
- گروه واقعی در `pyproject.toml` «**serve**» است (fastapi + uvicorn + python-multipart)؛ `.[api]` فرضی از `docs/RUNTIME_INSTALL.md` حذف و به `.[serve]` اصلاح شد (+`.[dev]` برای تست).
- اعتبارسنجی واقعی: `pip install -e ".[serve]"` با exit 0 نصب شد؛ بررسی اسکریپت `scripts/validate_pyproject.py`: گروه‌های serve/audio/dev موجود، گروه [api] وجود ندارد.
- ضمناً در همان سند: متغیر env واقعی **`AVA_NATIVE_BUNDLES`** (قالب `id=/path,...` طبق `ava/api/server.py`) جایگزین `AVA_BUNDLE_DIRS` فرضی شد، مثال CLI به فلگ واقعی `speak --text ... --bundle ...` اصلاح شد و دستور smoke-training به فرم واقعی `scripts/make_native_smoke_data.py` + `python3 -m training.train_native --manifest ... --out ... --steps ...` (طبق `docs/TRAINING_GUIDE.md`) تصحیح شد.

### ۳) مجوز و وضعیت تجاری صدای Native — دقیق و بدون ادعای زودهنگام
- `ava/runtime/backends.py` (متن و خروجی API): صداهای `ava-n1-*` اکنون با
  - license = **«آزمایشی — مجوز وزن نهایی هنوز تعیین نشده»**،
  - description = «صدای آزمایشی AVA-N1 — وزن آموزش‌ندیده (fixture داخلی)؛ پس از تأمین دیتاست مجاز و آموزش واقعی، وضعیت نهایی مجوز تعیین می‌شود»،
  - `commercial_ok=False`
  ارائه می‌شوند. ادعای قبلی «MIT (کد/وزن‌های مخزن)» و `commercial_ok=True` حذف شد.
- تفکیک شفاف در مستندات: مجوز MIT فقط **کد** مخزن را پوشش می‌دهد؛ **وزن‌های** نهایی هنوز ساخته نشده‌اند و مجوزشان پس از آموزش (فاز ۳) تعیین می‌شود. وزن‌های Legacy (CC BY-NC) خارج از مسیر محصول‌اند.
- هماهنگ‌سازی: `tests/test_runtime.py::test_load_and_list_voices` اکنون دقیقاً همین وضعیت را اثبات می‌کند (commercial_ok=False + «آزمایشی» در مجوز).
- هیچ مجوزی حدس زده نشد؛ هیچ ادعای تجاری بدون پشتوانه باقی نماند.

### ۴) هویت بسته Python — AVA Native مستقل
- `pyproject.toml`: توضیح → «Ava (AVA Native): independent, offline Persian text-to-speech engine with the original AVA-N1 architecture — CPU inference and a full training pipeline.» و کلیدواژه‌ها → `tts, text-to-speech, persian, speech, offline, ava-n1`؛ **pocket-tts و kyutai از metadata محصول حذف شدند**.
- منشأ اجزای Legacy عمداً و کامل حفظ شده است: `NOTICE.md` (جدول attribution + یادداشت استقلال فاز ۱)، `docs/CORE_INDEPENDENCE_AUDIT.md` و سرنویس فایل‌های اقتباسی — این اصلاح پنهان‌سازی نیست؛ فقط هویت *محصول* اصلاح شد.

### ۵) شمارش تست‌ها — تطبیق با واقعیت
شمارش واقعی با `pytest --collect-only`:
- `tests/test_core_independence.py`: **۱۴ تست** (۱۳ در اجرای فاز ۱ + ۱ تست closeout). عدد «۱۱» قبلی در `docs/CORE_INDEPENDENCE_AUDIT.md` خطای نگارشی بود؛ اصلاح شد.
- کل مجموعه: **۱۲۹ تست** (۱۱۵ قبلی + ۱۴)؛ اجرای تاریخی فاز ۱ روی ۱۲۸ تست (قبل از افزودن تست closeout) و نتیجه ۱۲۸/۱۲۸ سبز — هر دو عدد با ذکر زمان ثبت شده‌اند تا مبهم نباشد.
- اسناد هماهنگ‌شده: `docs/CORE_INDEPENDENCE_AUDIT.md`، `docs/TEST_REPORT.md`، `CHANGELOG_PHASE1.md` (+`PROJECT_STATUS.md`، `docs/ROADMAP.md`).

## ۲. تست‌های اجراشده و نتایج واقعی

| بررسی پذیرش | روش | نتیجه |
| --- | --- | --- |
| ۱) metadata بسته | `scripts/validate_pyproject.py` (tomllib) | **۱۲/۱۲ PASS** |
| ۲) سازگاری دستور نصب | `pip install -e ".[serve]"` واقعی | **exit 0** |
| ۳) سازگاری مستندات باندل با کد | `test_bundle_format_matches_documentation` (باندل واقعی + متن دو سند) | **سبز** |
| ۴) مجوز/metadata صدای Native | `test_load_and_list_voices` (نسخهٔ جدید) | **سبز** — commercial_ok=False، «آزمایشی» در مجوز |
| ۵) تست‌های مرتبط قبلی | `test_core_independence.py + test_runtime.py + test_platform.py` | **۳۷ passed + 3 skipped، exit 0** در 19.3s |

- سه skip علت‌دار و شفاف: سه تست آداپتور Legacy در `tests/test_runtime.py` به دلیل نبود عمدی وزن Legacy در چک‌اوت (مطابق محدودیت «بازیابی وزن ممنوع») با پیام `legacy weights/config not present (commercial-clean checkout)` رد می‌شوند.
- مجموعهٔ کامل بازاجرای نشد (محدودیت closeout: تست غیرمرتبط ممنوع)؛ آخرین اجرای کامل معتبر ۱۲۸/۱۲۸ در فاز ۱ است.
- محیط توسعه بین پیام‌ها بازیافت شد؛ مخزن از ZIP فاز ۱ (با تاریخچهٔ git کامل) بازسازی شد و اصلاحات روی همان اعمال گردید.

## ۳. چیزهایی که عمداً تغییر نکرد

کد آموزش/بارگذاری باندل (`training/train_native.py`)، معماری AVA-N1، API/CLI و قراردادها، فایل‌های Legacy و `web/vendor/phonon-tts/` (حذف ممنوع)، `weights/` و `samples/` (بازیابی ممنوع)، تاریخچهٔ Git (فقط یک کامیت جدید closeout اضافه می‌شود).

## ۴. موارد باقی‌مانده / محدودیت‌ها (شفاف)

- صدای Native همچنان **آزمایشی** است: تا تأمین دیتاست مجاز + آموزش واقعی (فاز ۳)، هیچ ادعای کیفیت/تجاری وجود ندارد.
- مجوز وزن‌های نهایی: تعیین‌نشده — پس از آموزش تصمیم کارفرما لازم دارد.
- حذف فیزیکی فایل‌های Legacy/وندور: فاز ۶ (بسته‌بندی)؛ تا آن‌زمان فقط از مسیر اجرا جدا هستند.
- پروفایل بیلد «محصول» بدون `weights/`: فاز ۶.
- اجرای کامل مجموعه در این مرحله انجام نشد (فقط تست‌های مرتبط، طبق مأموریت).
