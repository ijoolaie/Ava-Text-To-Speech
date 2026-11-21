# معماری پایه (ARCHITECTURE BASELINE) — وضعیت قبل از بازسازی

این سند معماریِ موجود را طوری ثبت می‌کند که بازسازی AVA Native بتواند با یک نقشهٔ مرجع شروع کند.

## 1. مسیر استنتاج Legacy (فعلاً تنها مسیر تولید صدای واقعی)

```
متن فارسی
  └─ ava/text_norm.py          نرمال‌سازی (نیم‌فاصله، ارقام، تاریخ/درصد/واحد، علائم)
  └─ ava/models/text_chunking  تقسیم به تکه‌های ≤ حد توکن
  └─ ava/models/text_conditioner (Kyutai MIT)  توکن‌سازی/شرط‌سازی متن
  └─ ava/models/flow_lm.py     Flow-LM یک‌مرحله‌ای روی latent کدک؛ 15.56M پارامتر
       │   ورودی شرطی: بردار صدا (10s audio_prompt → Mimi encode، 12.5Hz)
       └─ خروجی: latent های Mimi (75Hz درون‌یابی‌شده، 12.5Hz مؤثر)
  └─ ava/models/mimi.py        Mimi codec — decoder سمی-ضد؛ 2.62M decoder + 9.75M encoder
  └─ خروجی: PCM 24kHz مونو
```

- کل چک‌پوینت: 27.93M پارامتر؛ 302 تانسور؛ shardهای < 100MB؛ بارگذاری صحیح با تست `test_model_and_weights.py` تأیید شده.
- سرعت روی CPU (2 هسته سندباکس): RTF ≈ 4.5–5.1× (گفتار ~3s در ~0.6s)؛ روی CPU دسکتاپ سریع‌تر.
- صداهای فعلی: nava، shirin، arash (بردارهای شرطی از چک‌پوینت Legacy — مجوز NC).

## 2. پایپ‌لاین آموزش Legacy (تأییدشده با smoke واقعی)

`training/data/prepare_data.py` (latent cache از صوت + متن) → `training/train_flow_lm.py` (هدف drifting یک‌مرحله‌ای + سر EOS، EMA، stats-EMA، AMP اختیاری) → `training/export.py` (باندل) → `training/evaluate.py`. smoke آموزش CPU: loss 1.33→0.52 (همبستگی 0.70) — این پایپ‌لاین به‌عنوان مرجع مهندسی برای trainer اختصاصی نگه داشته می‌شود.

## 3. وب/CLI/API فعلی

- CLI: `scripts/speak_fa.py` (تک‌فایل، ذخیرهٔ WAV)، `scripts/serve.py` (HTTP ساده برای دموی WASM).
- وب: دموی WASM آفلاین (`web/`) با vendor phonon-tts — فقط فرمت Legacy.
- API واقعی تولید صدا با کنترل صدا/سرعت/جریان: **وجود ندارد** → فاز ۵ و ۷.

## 4. شکاف‌های مشخص‌شده (ورودی بازسازی)

| شکاف | فاز پاسخگو |
| --- | --- |
| موتور زبان ماژولار و تست‌شده (اعداد/تاریخ/ساعت/واحد/تلفظ/بخش‌بندی) | فاز ۱ |
| معماری عصبی اختصاصیِ قابل‌آموزش روی دادهٔ مجاز | فاز ۲ |
| پایپ‌لاین دیتاست با اعتبارسنجی و منشأ | فاز ۳ |
| trainer اختصاصی با checkpoint/resume/val | فاز ۴ |
| Runtime ماژولار با رابط پایدار (stream/batch/stop) | فاز ۵ |
| سیستم صدا با پروفایل/متادیتا/انتخاب | فاز ۶ |
| API + CLI یکپارچه + مسیر Android/ONNX + Windows | فاز ۷ |
| قرارداد اتصال Raya-Hybrid | فاز ۸ |
| تست امنیت/کارایی و بنچمارک رسمی | فاز ۹ |
