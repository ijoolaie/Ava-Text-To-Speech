# قراردادهای مرزی هستهٔ AVA Native (NATIVE CONTRACTS)

نسخهٔ 1.1.0 · فاز ۱ · این قراردادها پایدارند و فقط با نسخهٔ جدید API تغییر می‌کنند.

چهار مرز مستقل هستهٔ Native و قرارداد دقیق هر یک. هیچ مرزی به کد یا وزن خارجی وابسته نیست؛ جزئیات استقلال در `docs/CORE_INDEPENDENCE_AUDIT.md`.

## ۱. Text Frontend (`ava.language`)

- ورودی: متن خام فارسی/ترکیبی (str، UTF-8)
- خروجی: متن نرمال‌شدهٔ آمادهٔ توکن‌سازی (`LanguageEngine.normalize(text) -> str`) — قطعی و idempotent روی خروجی خودش
- وظایف: نرمال‌سازی نویسه‌ها (عربی→فارسی، حذف نویسهٔ نامرئی)، نیم‌فاصله، اعداد/ترتیبی/اعشاری، تاریخ شمسی و میلادی، ساعت/دقیقه/ثانیه، تلفن، واحدها و ارز، سیاست متن ترکیبی، واژگان تلفظ، جمله‌بندی، جدول مکث
- توکن‌سازی: `CharTokenizer.encode(str) -> list[int]` با BOS/EOS خودکار؛ واژگان کاراکتری ثابت (`tokenizer.vocab`)
- تضمین: هیچ حالتی خروجی خالی/بیت‌به‌بیت غیرقطعی نمی‌دهد؛ نویسه‌های ناشناخته → `<unk>` نه خطا

## ۲. Acoustic Model (`ava.models.native.AVANativeTTS`)

- معماری: AVA-N1 (ثابت — تصمیم پروژه؛ تغییر فقط با دلیل فنی و نسخهٔ مدل جدید)
- ورودی استنتاج: `infer(tokens [1,T] int, lengths [1] int, speed float, speaker_id [1] int)`
- خروجی: mel `(n_mels=80, F)` float32 در بازهٔ [-1, 1]؛ `F ≤ cfg.max_frames`
- تضمین: استنتاج NAR قطعی (با ورودی یکسان، خروجی یکسان؛ eval/no_grad)
- آموزش و استنتاج جدا هستند: `training/train_native.py` → `export_native_bundle()` → باندل استنتاج (`ava-native-bundle`: weights.safetensors + tokenizer.json + config.json + meta.json)
- نکتهٔ سازگاری: `infer` خروجی batch-1 را فشرده می‌کند `(n_mels, F)`؛ نسخه‌های بعدی API با نام متفاوت و batch صریح ارائه می‌شوند

## ۳. Vocoder (`ava.audio.vocoder`)

- مرز: کلاس انتزاعی `Vocoder` با `mel_to_audio(mel (80,F) float, cfg) -> np.ndarray float32`
- رجیستری: `get_vocoder(name)` — فعلاً فقط `griffin-lim-v1`
- وضعیت v1: Griffin-Lim، الگوریتمی، بدون وزن — **موقت و غیرنهایی**؛ کیفیت نهایی با vocoder عصبی اختصاصی AVA (فاز ۳) که پشت همین مرز نشسته می‌شود، تأمین می‌شود
- ممنوع: استفاده از vocoder خارجی یا وزن آمادهٔ دیگران در مسیر محصول

## ۴. Runtime (`ava.runtime.SynthesisEngine`)

- بک‌اندها پلاگین‌پذیرند (`register_backend`)؛ مسیر محصول فقط `NativeN1Backend` است
- `load_model` بدون باندل معتبر → خطای شفاف «وزن آموزش‌دیده موجود نیست» (فایل یافت نشد/فرمت نامعتبر) — نه سکوت و نه fallback به Legacy
- ورودی: متن + voice_id + `SynthesisOptions(speed, sample_rate)`؛ خروجی: `SynthesisResult(audio float32, sample_rate=24000 پیش‌فرض, voice, text, duration_s, elapsed_s, rtf)`
- استریم: `synthesize_stream(text, voice)` → یک chunk برای هر جمله؛ توقف: `stop()` + event درخواستی → `SynthesisCancelled`
- خروجی صوتی: `wav_bytes(result)` — WAV PCM-16 مونو 24kHz
- آفلاین: هیچ I/O شبکه در هیچ مسیری؛ CPU-only؛ بدون import اجباری Legacy (تست: `tests/test_core_independence.py`)
- وزن‌های Legacy (CC BY-NC) در مسیر Native هرگز استفاده نمی‌شوند؛ آداپتور `LegacyPocketTTSBackend` فقط برای مقایسه است و صداهایش `commercial_ok=False`
