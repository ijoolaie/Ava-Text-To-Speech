# آوا

**متن‌به‌گفتار فارسی در قالب Pocket TTS — یک مخزن کاملاً خوداتکا.**

آوا یک مدل گفتار فارسی با ۱۸.۲ میلیون پارامتر است که بر پایهٔ دستورِ «مدل کوچک» کیوتای (Pocket TTS) ساخته شده؛ روی CPU، گوشی و مرورگر اجرا می‌شود و با یک نمونهٔ صوتی کوتاه، به صدای همان گوینده حرف می‌زند.

این مخزن **بدون هیچ وابستگی بیرونی** است: وزن‌های مدل، توکنایزر، صداها، دموی مرورگری و خط کامل آموزش همه همین‌جا هستند — در زمان اجرا هیچ چیزی دانلود نمی‌شود و هیچ وابستگی‌ای به مخزن gated هاگینگ‌فیس وجود ندارد.

| | |
|---|---|
| مدل زبانی جریانی (Flow LM) | ۱۵.۶M پارامتر — ۸ لایه، d_model=512، سرِ نمونه‌گیر «drifting» یک‌مرحله‌ای |
| دیکودر صوت (Mimi) | ۲.۶M پارامتر — ترنسفورمر ۱۲ لایه در ۱۲.۵ هرتز |
| انکودر گوینده (Mimi) | ۹.۷M پارامتر — کدگذاری نمونهٔ صدا |
| نرخ نمونه / نرخ فریم | ۲۴kHz مونو / ۱۲.۵ هرتز |
| کل چک‌پوینت | ۲۷.۹M پارامتر (~۱۰۷MB float32، در ۳ قطعهٔ زیر ۱۰۰MB) |
| سرعت (CPU دو هسته‌ای) | ≈ ۴ برابر زمان واقعی (اندازه‌گیری: ۴.۳ ثانیه صدا در ۱.۰ ثانیه) |
| مجوز وزن‌ها | CC BY-NC 4.0 (غیرتجاری)؛ کد MIT — [NOTICE.md](NOTICE.md) |

## شروع سریع

```bash
pip install -e .

# با صداهای همراه مخزن (nava / shirin / arash)
python scripts/speak_fa.py --voice shirin --text "سلام، حال شما چطور است؟" --out hello.wav

# یا کلون صدای دلخواه از یک wav کوتاه (۳ تا ۱۰ ثانیه)
python scripts/speak_fa.py --voice my_voice.wav --text "این یک آزمایش است." --out mine.wav
```

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
