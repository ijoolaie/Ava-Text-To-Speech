# Runtime و نصب — AVA Native (فاز ۱)

مسیر محصول کاملاً آفلاین، CPU-only و مستقل از Legacy است.

## نصب (مسیر Native)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # هسته: torch/numpy/scipy/safetensors/pydantic/pyyaml/…
pip install -e ".[audio]"   # خروجی WAV روی دیسک (soundfile)
pip install -e ".[api]"     # سرویس HTTP (fastapi/uvicorn)
```

برای اجرای Native هیچ وابستگی Legacy لازم نیست و هیچ ماژول اقتباسی (pocket-tts) هنگام import بار نمی‌شود (تست‌شده: `tests/test_core_independence.py`).

## باندل مدل

استنتاج به یک باندل AVA-N1 نیاز دارد: `weights.safetensors + tokenizer.json + config.json + meta.json`.

- تا قبل از فاز ۳ (آموزش واقعی)، باندل‌های موجود فقط **fixture آزمایشیِ خود پروژه** هستند (کیفیت گفتار ندارند): تولید با `python scripts/make_native_smoke_data.py` + `python -m training.train_native --smoke` (راهنما: `docs/TRAINING_GUIDE.md`)
- اگر باندل موجود نباشد، Runtime/API/CLI خطای شفاف می‌دهند؛ هیچ fallback به Legacy انجام نمی‌شود

## استفاده

```bash
# CLI
ava-cli voices                          # فهرست صداها
ava-cli speak "سلام دنیا" --out out.wav # متن → WAV 24kHz
ava-cli serve --host 127.0.0.1 --port 8000

# API
uvicorn ava.api.server:app --port 8000
POST /v1/synthesis      {"text": "...", "voice": "ava-n1-ava"}   → WAV bytes
GET  /v1/voices | /v1/models | /v1/health

# وب (بدون وابستگی خارجی)
python -m http.server 8080 -d web-native   # http://localhost:8080
```

## متغیرهای محیطی

- `AVA_BUNDLE_DIRS` — مسیرهای باندل برای بارگذاری خودکار API (جداکنندهٔ `:`)

## Windows / Android

- راهنمای ویندوز: `desktop/windows/README.md` + `scripts/windows/run_ava.bat` (کد cross-platform؛ تست دستگاه واقعی در فاز ۶)
- اندروید: export ONNX (`training/export/onnx_export.py`) + `android/README.md` (فاز ۶)

## محدودیت‌های فعلی (شفاف)

- صدای واقعی محصول: پس از آموزش وزن اختصاصی (فاز ۳) — الان فقط mechanics تست‌شده است
- vocoder فعلی v1 (Griffin-Lim) موقت است — مرز `Vocoder` مستقل ثبت شده تا جایگزین بدون تغییر قرارداد بنشیند
- وزن‌های Legacy (CC BY-NC) فقط برای مقایسه‌اند؛ در بستهٔ محصول هرگز قرار نمی‌گیرند
