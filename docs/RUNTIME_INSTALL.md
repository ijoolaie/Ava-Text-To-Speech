# Runtime و نصب — AVA Native (فاز ۱)

مسیر محصول کاملاً آفلاین، CPU-only و مستقل از Legacy است.

## نصب (مسیر Native)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .             # هسته: torch/numpy/scipy/safetensors/pydantic/pyyaml/…
pip install -e ".[audio]"    # خروجی WAV روی دیسک (soundfile)
pip install -e ".[serve]"    # سرویس HTTP (fastapi/uvicorn/python-multipart)
pip install -e ".[dev]"      # تست (pytest + soundfile)
```

برای اجرای Native هیچ وابستگی Legacy لازم نیست و هیچ ماژول اقتباسی (pocket-tts) هنگام import بار نمی‌شود (تست‌شده: `tests/test_core_independence.py`).

## باندل مدل

استنتاج به یک باندل AVA-N1 نیاز دارد. **فرمت واقعی باندل (طبق کد `training/train_native.py`) سه فایل است:**

| فایل | محتوا | تولید/خوانش |
| --- | --- | --- |
| `ava_n1.pt` | `torch.save({"model": state_dict, "config": dict(cfg)})` — چک‌پوینت Torch، نه safetensors | `export_native_bundle()` / `load_native_bundle()` |
| `tokenizer.json` | `{"type": "ava-n1-char", "vocab": [...]}` واژگان کاراکتری | `CharTokenizer.save()/load()` |
| `meta.json` | `{"format": "ava-native-n1", "params_m", "sample_rate", "mel": {"n_mels", "hop"}}` | `export_native_bundle()` |

- تا قبل از فاز ۳ (آموزش واقعی)، همهٔ باندل‌ها **fixture آزمایشیِ آموزش‌ندیدهٔ خود پروژه** هستند (کیفیت گفتار ندارند و برای استفادهٔ تولیدی/تجاری مجاز نیستند). ساخت باندل آزمایشی طبق `docs/TRAINING_GUIDE.md`:
  ```bash
  python3 scripts/make_native_smoke_data.py data/smoke_native 20
  python3 -m training.train_native --manifest data/smoke_native/manifest.jsonl \
      --out runs/native_smoke_v1 --steps 300 --batch-size 8
  # باندل در runs/native_smoke_v1/bundle/ ساخته می‌شود
  ```
- اگر باندل موجود نباشد، Runtime/API/CLI خطای شفاف می‌دهند؛ هیچ fallback به Legacy انجام نمی‌شود

## استفاده

```bash
# CLI
ava-cli voices                                # فهرست صداها
ava-cli speak --text "سلام دنیا" --out out.wav --bundle runs/native_smoke_v1/bundle
ava-cli serve --host 127.0.0.1 --port 8000

# API
uvicorn ava.api.server:app --port 8000
POST /v1/synthesis      {"text": "...", "voice": "ava-n1-ava"}   → WAV bytes
GET  /v1/voices | /v1/models | /v1/health

# وب (بدون وابستگی خارجی)
python -m http.server 8080 -d web-native   # http://localhost:8080
```

## متغیرهای محیطی

- `AVA_NATIVE_BUNDLES` — بارگذاری خودکار باندل‌ها در API/CLI، قالب `id1=/path1,id2=/path2` (همان‌طور که در `ava/api/server.py::load_bundles_from_env` و فلگ `--bundle` CLI استفاده می‌شود)

## Windows / Android

- راهنمای ویندوز: `desktop/windows/README.md` + `scripts/windows/run_ava.bat` (کد cross-platform؛ تست دستگاه واقعی در فاز ۶)
- اندروید: export ONNX (`training/export/onnx_export.py`) + `android/README.md` (فاز ۶)

## محدودیت‌های فعلی (شفاف)

- **مجوز و وضعیت تجاری صدای Native:** همهٔ صداهای `ava-n1-*` **آزمایشی**اند (وزن آموزش‌ندیده) و `commercial_ok=False`؛ مجوز وزن‌های نهایی پس از آموزش واقعی (فاز ۳) تعیین می‌شود — مجوز MIT فقط کد مخزن را پوشش می‌دهد، نه وزن‌ها را
- صدای واقعی محصول: پس از آموزش وزن اختصاصی (فاز ۳) — الان فقط mechanics تست‌شده است
- vocoder فعلی v1 (Griffin-Lim) موقت است — مرز `Vocoder` مستقل ثبت شده تا جایگزین بدون تغییر قرارداد بنشیند
- وزن‌های Legacy (CC BY-NC) فقط برای مقایسه‌اند؛ در بستهٔ محصول هرگز قرار نمی‌گیرند
