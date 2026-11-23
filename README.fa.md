# آوا نیتیو (AVA Native)

**پلتفرم مستقل تبدیل متن فارسی به گفتار — موتور زبان خودمان، معماری خودمان (AVA-N1)، runtime پلاگین‌پذیر، پایپ‌لاین آموزش، API/CLI/وب و ONNX برای اندروید و ویندوز.**

آوا نیتیو نسل بعدی پروژهٔ آوا است: مسیر محصول، **مدل اختصاصی AVA-N1** است (غیرخودبازگشتی، آموزش‌پذیر روی دادهٔ مجاز خودمان) که از طریق runtime مستقل از معماری سرو می‌شود. قالب Legacy (اقتباس MIT از کیوتای / وزن‌های CC BY-NC) به‌عنوان بک‌اند اختیاری مقایسه و مرجع تاریخی باقی می‌ماند — [NOTICE.md](NOTICE.md) و [docs/PROJECT_AUDIT.md](docs/PROJECT_AUDIT.md).

## ساختار مخزن

| مسیر | محتوا |
| --- | --- |
| `ava/language/` | موتور زبان فارسی: نرمال‌سازی، اعداد/ترتیبی، تاریخ شمسی/میلادی، ساعت، تلفن، واحدها/ارز، سیاست متن ترکیبی، واژگان تلفظ، جمله‌بندی، prosody |
| `ava/models/native/` | **AVA-N1** — معماری اصلِ غیرخودبازگشتی (encoder → duration → decoder → 80-mel در 24kHz) |
| `ava/audio/` | خط mel + vocoder نسخهٔ اول (Griffin-Lim) |
| `ava/runtime/` | موتور سنتز پلاگین‌پذیر (synthesize/stream/batch/stop) با registry بک‌اند |
| `ava/voices/` | پروفایل صداها، فروشگاه با الزام رضایت‌نامه، سنجش کیفیت |
| `ava/api/` + `web-native/` | API سبک HTTP + رابط وب فارسی RTL |
| `ava/cli.py` | `ava-cli` — speak / voices / serve / validate-dataset / voice-quality |
| `training/` | پایپ‌لاین دیتاست (manifest/اعتبارسنجی/قطعه‌بندی/tقسیم)، trainer با checkpoint/resume/export، export ONNX |
| `tests/` | ~۱۱۹ تست خودکار (همگی سبز روی CPU) |
| `docs/` | ممیزی، معماری، نقشهٔ راه، معیارهای پذیرش، راهنماهای دیتاست/آموزش/ارزیابی، اتصال Raya-Hybrid |

## شروع سریع (مدل Native)

```bash
pip install -e ".[serve]"

# اجرای کامل پایپ‌لاین آموزش روی CPU (~۴ دقیقه) و export باندل
python scripts/make_native_smoke_data.py data/smoke_native 20
python -m training.train_native --manifest data/smoke_native/manifest.jsonl --out runs/native_smoke_v1 --steps 300

# تولید صدا با باندل آموزش‌دیده
ava-cli speak --text "سلام دنیا" --voice ava-n1-ava --bundle runs/native_smoke_v1/bundle --out hello.wav

# API و رابط وب فارسی روی http://127.0.0.1:8300/
ava-cli serve --bundle runs/native_smoke_v1/bundle
```

وضعیت واقعی و شکاف‌ها: [PROJECT_STATUS.md](PROJECT_STATUS.md). نقشهٔ راه رسمی هفت‌فازی: [docs/ROADMAP.md](docs/ROADMAP.md)؛ ارزیابی فاز صفر: [docs/ASSESSMENT_PHASE0.md](docs/ASSESSMENT_PHASE0.md). آموزش صدای فارسی واقعی به GPU و دادهٔ مجاز نیاز دارد ([docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md))؛ مدل smoke صرفاً fixture پایپ‌لاین است، نه صدای محصول.

## قالب Legacy (تاریخی/مقایسه — وزن‌های غیرتجاری)

مجموعهٔ خوداتکای قبلی آوا (چک‌پوینت Pocket-TTS با ۱۸.۲M پارامتر، ۲۴kHz مونو، RTF ≈ ۴× روی CPU دو هسته‌ای) از طریق آداپتور اختیاری اجرا می‌شود:

```bash
python scripts/speak_fa.py --voice shirin --text "سلام، حال شما چطور است؟" --out hello.wav
```

| | |
|---|---|
| مدل زبانی جریانی (Flow LM) | ۱۵.۶M پارامتر — ۸ لایه، d_model=512، سرِ نمونه‌گیر «drifting» یک‌مرحله‌ای |
| دیکودر صوت (Mimi) | ۲.۶M پارامتر — ترنسفورمر ۱۲ لایه در ۱۲.۵ هرتز |
| انکودر گوینده (Mimi) | ۹.۷M پارامتر — کدگذاری نمونهٔ صدا |
| نرخ نمونه / نرخ فریم | ۲۴kHz مونو / ۱۲.۵ هرتز |
| کل چک‌پوینت | ۲۷.۹M پارامتر (~۱۰۷MB float32، در ۳ قطعهٔ زیر ۱۰۰MB) |
| مجوز وزن‌ها | **CC BY-NC 4.0 (غیرتجاری)**؛ کد MIT — [NOTICE.md](NOTICE.md) |

API پایتون (سازگار با `pocket_tts`):

```python
from ava import TTSModel

model = TTSModel.load_model("config/fa.yaml")
voice = model.get_state_for_audio_prompt("weights/voices/nava.safetensors")
audio = model.generate_audio(voice, "سلام دنیا. این مدل فارسی روی پردازنده اجرا می‌شود.")
# audio: تنسور float32 [تعداد نمونه] با نرخ model.sample_rate (24000)
```

دموی مرورگری (اجرای مدل داخل مرورگر با رانتایم WASM):

```bash
python scripts/serve.py          # http://localhost:8000
```

## آموزش مدل خودتان

خط آموزش دقیقاً روایت کیوتای است: کدک Mimi **منجمد** می‌ماند و فقط flow LM روی لاتنت‌های پیوستهٔ کدک آموزش می‌بیند (هدف «drifting» یک‌مرحله‌ای + سر EOS). همهٔ مسیر در همین محیط به‌صورت سرتاسری تأیید شده است:

```bash
# ۱) آماده‌سازی داده: مانیفست CSV (مسیر صوت، متن) → کش لاتنت با کدک منجمد
python -m training.data.prepare_data --manifest my_data.csv --out data_cache/ava-fa

# ۲) آموزش flow LM (از صفر، یا فاین‌تیون با model.init_from: weights/model.safetensors)
python -m training.train_flow_lm --config training/configs/ava_fa_flow_lm.yaml

# ۳) خروجی گرفتن هر مرحلهٔ آموزش در قالب Pocket TTS
python -m training.export --checkpoint runs/ava-fa/checkpoint-latest.pt

# ۴) ارزیابی و ساخت نمونهٔ صدا
python -m training.evaluate --checkpoint runs/ava-fa/checkpoint-latest.pt
```

اجرای دود روی CPU (`make smoke-train`) با خود مدلِ منتشرشده ۱۶ کلیپ می‌سازد، یک flow LM کوچک را روی آن‌ها آموزش می‌دهد و کاهش خطا را نشان می‌دهد (اندازه‌گیری‌شده: MSE لاتنت ۱.۳۳ → ۰.۵۲ و همبستگی پیش‌بینی/هدف ۰.۰۰ → ۰.۷۰ در ۱۵۰ گام / ۲ ثانیه). جزئیات هدف آموزش، زمان‌بندی‌ها و محدودیت‌های صادقانهٔ این تأیید در [docs/TRAINING.md](docs/TRAINING.md) و راهنمای دیتاست در [docs/DATA.md](docs/DATA.md).

## نکته‌های ساختار مخزن

- `weights/model.safetensors` قطعه‌قطعه شده (index + سه shard) چون گیت‌هاب فایل بزرگ‌تر از ۱۰۰MB را نمی‌پذیرد. پکیج `ava` شفاف از index استفاده می‌کند؛ `scripts/split_weights.py merge` فایل واحدِ دموی وب را می‌سازد (در `scripts/serve.py` خودکار انجام می‌شود).
- سه فایل صدا، بردارهای conditioning از پیش‌محاسبه‌شده هستند — همان قالبی که رانتایم مرورگر مصرف می‌کند — پس برای آن‌ها هیچ انکد صوتی در زمان اجرا لازم نیست.
- `ava/text_norm.py` متن فارسی را پیش از سنتز نرمال می‌کند (حروف عربی→فارسی، عدد→واژه، درصد، نقطه‌گذاری، نیم‌فاصله) — همان رفتار `persian.js` دمو.

## مستندات

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — معماری مدل، تنسوربه‌تنسور، و روش تأیید این مخزن با چک‌پوینت
- [docs/TRAINING.md](docs/TRAINING.md) — هدف آموزش، خط داده، زمان‌بندی‌ها، خروجی‌گیری و شواهد اجرای دود
- [docs/DATA.md](docs/DATA.md) — دیتاست‌ها، قالب مانیفست، نرمال‌سازی متن برای آموزش
- [docs/PUBLISH.md](docs/PUBLISH.md) — انتشار روی گیت‌هاب (قطعه‌ها، LFS، محدودیت Pages)
- [NOTICE.md](NOTICE.md) — منشأ و مجوز تک‌تک فایل‌ها (کد MIT، وزن‌ها CC BY-NC)

## سپاسگزاری

- [Kyutai pocket-tts](https://github.com/kyutai-labs/pocket-tts) (MIT) — معماری مدل، رانتایم استریمینگ و قالب؛ ماژول‌های `ava/` از آن اقتباس شده‌اند.
- کدک [Mimi](https://github.com/kyutai-labs/moshi) کیوتای — کدک داخل چک‌پوینت.
- [phonon-tts](https://www.npmjs.com/package/phonon-tts) (Apache-2.0/MIT) — رانتایم WASM مرورگر در `web/vendor/`.
- وزن‌های منتشرشدهٔ مدل فارسی (CC BY-NC 4.0)، همراه مخزن در `weights/` — چک‌پوینت مرجع: [`mehdi-hf/cilla-mini`](https://huggingface.co/mehdi-hf/cilla-mini).
