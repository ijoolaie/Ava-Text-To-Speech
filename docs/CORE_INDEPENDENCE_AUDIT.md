# ممیزی استقلال هستهٔ Native — فاز ۱ (CORE INDEPENDENCE AUDIT)

تاریخ: 2026-11-23 · مبنا: `AVA_Phase-0_Roadmap_v1.0.0` (کامیت `6bf7930`) · شاخه: `feature/ava-native`

طبق §۲ مأموریت فاز ۱، هر جزء پروژه در یکی از این دسته‌ها قرار گرفته است:

- **A** — پیاده‌سازی مستقل AVA (کد اصل، تست‌شده)
- **B** — کد اقتباس‌شده یا وابسته به پروژه خارجی
- **C** — وابستگی عمومی با مجوز سازگار (بدون کد اقتباسی)
- **D** — جزء Legacy که نباید در مسیر اجرای Native حضور داشته باشد

## ۱. کد پایتون مخزن

| جزء | فایل‌های دقیق | دسته | توضیح / وضعیت جایگزینی |
| --- | --- | --- | --- |
| موتور زبان فارسی (Text Frontend) | `ava/language/*.py` (chars, numbers, datetime_speech, language_detect, pronunciation, segmentation, prosody, pipeline, token_counter) | **A** | پیاده‌سازی کامل اصل؛ قرارداد در `docs/NATIVE_CONTRACTS.md` |
| سازگاری نام قدیمی نرمال‌سازی | `ava/text_norm.py` | **A** | فقط re-export به `ava.language`؛ رفتار با تست طلایی تثبیت شده |
| مدل عصبی Native | `ava/models/native/{model,config,tokenizer}.py` | **A** | معماری اصل AVA-N1؛ هیچ کد/وزن خارجی |
| I/O وزن Native | `ava/utils/weights_io.py` | **A** | ذخیره/خواندن shard خود پروژه |
| mel و vocoder v1 | `ava/audio/mel.py`، `ava/audio/vocoder.py` | **A** | v1 Griffin-Lim (الگوریتمی، بدون وزن) — **موقت و غیرنهایی**؛ پشت مرز `Vocoder` مستقل؛ vocoder عصبی اختصاصی بعداً جایگزین می‌شود (فاز ۳) |
| Runtime | `ava/runtime/engine.py`، `ava/runtime/__init__.py` | **A** | پلاگین‌پذیر، آفلاین، CPU، WAV |
| آداپتور Legacy | `ava/runtime/backends.py::LegacyPocketTTSBackend` | **B/D** | آداپتورِ اختیاری؛ import اقتباسی فقط **lazy داخل متد**؛ خروجی آن هرگز در بستهٔ محصول نیست (سرگیت commercial_ok=False) |
| استک مدل Legacy | `ava/models/{tts_model,flow_lm,mimi,text_chunking,model_state}.py`، `ava/modules/*.py`، `ava/data/` | **B/D** | اقتباس از Kyutai pocket-tts v4.0.1 (MIT) — فقط مقایسه/ارزیابی؛ دیگر به‌صورت اجباری بار نمی‌شود (اصلاح فاز ۱) |
| I/O وزن Legacy | `ava/utils/weights_loading.py` | **B/D** | نگاشت نام flow_lm/mimi؛ فقط توسط `tts_model.py` و `training/export.py` |
| اسکیمای config Legacy | `ava/utils/config.py` (FlowLMConfig/MimiConfig + resolve_path) | **B/D** | اقتباس pocket-tts؛ مسیر Native از `ava/models/native/config.py` (اصل) استفاده می‌کند |
| ابزار عمومی | `ava/utils/utils.py` | **A** | توابع عمومی خود پروژه |
| API | `ava/api/server.py` | **A** | FastAPI؛ فقط `ava.runtime` |
| CLI | `ava/cli.py` | **A** | پنج زیرفرمان؛ فقط مسیر Native |
| صداها | `ava/voices/{registry,quality}.py` | **A** | الزام رضایت‌نامه؛ بدون وابستگی خارجی |
| دیتاست | `training/datasets/*.py` | **A** | manifest/validate/segment/split اصل |
| آموزش Native | `training/train_native.py`، `training/data/latents.py` (بخش Native) | **A** | استقلال آموزش و استنتاج رعایت شده |
| ابزار Legacy training | `training/train_codec.py`، `training/export.py`، `scripts/{make_smoke_data,export_voice,speak_fa}.py`، `training/evaluate.py` | **B/D** | ابزار مقایسه/توسعهٔ Legacy؛ خارج از بستهٔ محصول؛ در فاز ۱ فقط برچسب‌گذاری شدند، حذف نشدند (قانون §۱-۵) |
| export ONNX | `training/export/onnx_export.py` | **A** | از مسیر Native؛ parity تست‌شده |

## ۲. فرانت‌اند وب

| جزء | فایل‌ها | دسته | توضیح |
| --- | --- | --- | --- |
| وب Native (محصول) | `web-native/index.html` | **A** | کاملاً self-contained؛ بدون هیچ وندور خارجی (تست شد) |
| دموی وب Legacy | `web/{index.html,app.js,persian.js,persian_tables.js,timing_model.json}` | **B/D** | رابط دموی قدیمی؛ به استک Legacy وابسته است؛ برچسب Legacy |
| وندور phonon-tts | `web/vendor/phonon-tts/` (device.js + مجوزهای Apache-2.0/MIT + THIRD_PARTY_LICENSES.md) | **B** | وندور مرورگری پروژهٔ [phonon-tts (gradium-ai)](https://github.com/gradium-ai) — WASM runtime Rust؛ فقط در دموی Legacy استفاده می‌شود؛ **هرگز وارد بستهٔ محصول نمی‌شود**؛ حذف کامل در فاز ۶ (بسته‌بندی) انجام می‌شود، نه اکنون (قانون عدم حذف ناگهانی) |

## ۳. وابستگی‌های پایتون (همه دسته C — مجاز)

`torch`, `numpy`, `scipy`, `safetensors`, `soundfile` (اختیاری)، `sentencepiece`, `tokenizers`, `pydantic`, `pyyaml`, `typing-extensions`, `fastapi`, `uvicorn`, `httpx` (تست)، `onnx`, `onnxruntime`, `onnxscript`، کتابخانه‌های تست (`pytest`, `librosa`, `soundfile`). جزئیات مجوزها: `docs/DEPENDENCY_LICENSE_AUDIT.md`. **هیچ وابستگی‌ای به بستهٔ pocket-tts یا phonon وجود ندارد.**

## ۴. تنظیمات و مستندات

- `pyproject.toml` — دسته C؛ ورودی محصول فقط `ava-cli`
- `config/fa.yaml` — تنظیمات Legacy (برای ابزار مقایسه)؛ کانفیگ Native در باندل است
- `NOTICE.md`، `docs/DEPENDENCY_LICENSE_AUDIT.md`، `LICENSE` — به‌روزرسانی فاز ۱
- `docs/CORE_INDEPENDENCE_AUDIT.md` (این سند)، `docs/NATIVE_CONTRACTS.md`، `docs/RUNTIME_INSTALL.md` — جدید فاز ۱

## ۵. قراردادهای بخش‌های اقتباسی و جایگزین موردنیاز

| جزء اقتباسی | عملکرد | قرارداد ورودی→خروجی | وابستگی فنی | جایگزین مستقل موردنیاز |
| --- | --- | --- | --- | --- |
| `ava/modules/` + استک مدل Legacy | مدل گفتار فرمت pocket-tts | متن → `TTSModel.generate_audio(state, text)` → tensor [samples] | torch + وزن‌های CC BY-NC | **جایگزین کامل = AVA-N1 + آموزش واقعی (فاز ۳)** — زیرساختش همین حالا آماده و تست‌شده است |
| `web/vendor/phonon-tts/` | TTS مرورگری WASM برای دموی وب Legacy | checkpoint WASM → stream صوت | مرورگر، وندور WASM | دموی Native خود (`web-native/`) جایگزین کاربردی است؛ حذف فایل‌ها در فاز ۶ |
| `ava/utils/weights_loading.py` + `config.py` (بخش اقتباسی) | نگاشت نام و schema وزن Legacy | checkpoint .pt → state_dict flow_lm/mimi | safetensors/torch | با بازنشستگی Legacy حذف می‌شود؛ هیچ بخشی از مسیر محصول به آن نیازمند نیست |

## ۶. اصلاحات استقلال‌سازی انجام‌شده در فاز ۱

1. **`ava/__init__.py`** — قبلاً `TTSModel` (استک Legacy) را در سطح ماژول import می‌کرد؛ یعنی هر `import ava.*` کد اقتباسی pocket-tts را بار می‌کرد. اکنون با **PEP 562 lazy** فقط در صورت درخواست صریح بار می‌شود. سازگاری اسکریپت‌های Legacy حفظ شد.
2. **مرز Vocoder مستقل** — `ava/audio/vocoder.py` با کلاس انتزاعی `Vocoder`، پیاده‌سازی موقت `GriffinLimVocoder` و رجیستری `get_vocoder`؛ `NativeN1Backend` از این مرز استفاده می‌کند (رفتار خروجی بدون تغییر).
3. **قراردادهای مرزی مستند** — `docs/NATIVE_CONTRACTS.md` برای چهار مرز Text Frontend / Acoustic Model / Vocoder / Runtime.
4. **تست‌های استقلال** — `tests/test_core_independence.py` (۱۴ تست پس از اصلاح closeout؛ در اجرای فاز ۱: ۱۳ تست): import هسته بدون Legacy در پروسهٔ جدا، پویش AST فهرست ممنوع در مسیر Native، قرارداد frontend/model/vocoder، سنتز کامل در محیط بدون Legacy، CLI، سازگاری فرمت باندل مستندات↔کد، ایزوله‌سازی وب.
5. **برچسب‌گذاری Legacy** — سندها و NOTICE به‌روزرسانی شدند؛ هیچ فایل Legacy حذف نشد (قانون §۱-۵).

## ۷. مواردی که هنوز جایگزین نشده‌اند (شفاف)

- صدای واقعی محصول: هنوز وابسته به آموزش وزن اختصاصی است (فاز ۳) — استقلال کد ≠ استقلال صدای آموزش‌دیده.
- vocoder نهایی: v1 Griffin-Lim موقت است؛ vocoder عصبی اختصاصی بعداً پشت همان مرز نشسته می‌شود.
- حذف فیزیکی فایل‌های Legacy/وندور از مخزن و بستهٔ نصبی: به فاز ۶ (بسته‌بندی) موکول شد تا تاریخچه و امکان مقایسه حفظ شود؛ در بستهٔ محصول هرگز وارد نمی‌شوند.
