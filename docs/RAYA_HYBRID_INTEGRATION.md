# اتصال AVA Native به Raya-Hybrid (فاز ۸)

اصل طراحی: **AVA هیچ وابستگی به Raya ندارد** — Raya به‌عنوان یک کلاینت HTTP استاندارد با AVA صحبت می‌کند. قرارداد زیر همان API تست‌شدهٔ فاز ۷ است (`ava/api/server.py`، تست: `tests/test_platform.py::TestAPI`).

## 1. Endpoints (REST)

| متد | مسیر | ورودی | خروجی |
| --- | --- | --- | --- |
| GET | `/v1/health` | - | `{status, models, voices}` |
| GET | `/v1/voices` | - | لیست صداها با `commercial_ok` و مجوز |
| GET | `/v1/models` | - | مدل‌های بارگذاری‌شده + محدودیت متن |
| POST | `/v1/synthesis` | `{"text", "voice", "speed"?}` | **audio/wav** + هدرهای `X-Duration-Seconds`، `X-Sample-Rate` |
| POST | `/v1/synthesis/stream` | همان بالا | NDJSON: هر خط `{voice, text, duration_s, audio_wav_b64}` یا `{"status":"cancelled"}` |

## 2. کنترل‌ها

- **سرعت**: `speed` ∈ [0.5, 2.5] (خارج از بازه → 422)
- **طول متن**: ≤ `AVA_NATIVE_MAX_TEXT` (پیش‌فرض 5000 نویسه → خطای 422)
- **صدا**: نام نامعتبر → 404 با پیام فارسیِ قابل نمایش
- **توقف**: Raya درخواست fetch را با `AbortController` لغو می‌کند؛ برای stream سمت سرور، مکانیزم `SynthesisOptions.stop_event` (per-request) لغو در مرز جمله می‌دهد.
- **خطاها**: JSON `{"detail": "..."}` با کد 400/404/422/500؛ `detail` فارسی و قابل نمایش به کاربر نهایی است.

## 3. جریان پیشنهادی Raya

```
کاربر → Raya: «بخوان: …»
Raya → GET /v1/voices (کش ۵ دقیقه‌ای)
Raya → POST /v1/synthesis (voice انتخابی؛ speed از تنظیمات کاربر)
      ← audio/wav
Raya: پخش یا ارسال به کانال صوتی
```

برای پاسخ‌های طولانی (تک‌جمله‌ای > 5 ثانیه) از `/v1/synthesis/stream` استفاده شود تا جملهٔ اول قبل از تولید بقیه پخش شود.

## 4. قرارداد مستقل از معماری

- Raya هیچ فرضی دربارهٔ مدل داخلی (AVA-N1 یا آداپتور Legacy) ندارد؛ فقط `voice_id` از `/v1/voices`.
- نمونه‌شناسی صدا ثابت است: **24kHz، مونو، PCM-16** — مستند در هر پاسخ با هدر `X-Sample-Rate`.
- آینده (Voice Cloning / لحن): فیلدهای `style` و `emotion` به `SynthesisRequest` اضافه می‌شوند؛ رفتار پیش‌فرض نادیده‌گرفتن با کد 200 (سازگاری رو به جلو).

## 5. استقرار

- سرویس AVA کنار Raya (همان ماشین): `ava-cli serve --host 127.0.0.1 --port 8300 --bundle <path>`
- تأخیر مورد انتظار روی CPU: RTF ≈ 1-3× (گفتار ۵ ثانیه‌ای در ۲-۵ ثانیه) — برای تعامل صوتی زنده، stream و مدل v2 vocoder عصبی لازم است.
- امنیت: API فقط روی localhost/شبکهٔ داخلی bind شود؛ اگر مصرف عمومی لازم شد، یک reverse proxy با توکن جلوی آن قرار گیرد (AVA خودش احراز هویت ندارد — ثبت‌شده درKNOWN-ISSUES).
