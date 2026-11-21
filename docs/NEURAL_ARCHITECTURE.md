# طراحی معماری عصبی اختصاصی — AVA-N1 (فاز ۲)

## 1. الزامات طراحی

| الزام | نتیجهٔ طراحی |
| --- | --- |
| آموزش‌پذیر روی دادهٔ مجاز خودمان (بدون وزن شخص ثالث) | همهٔ ماژول‌ها پیاده‌سازی اصلی؛ فقط ایده‌های عمومی (attention، mel) |
| قابل اجرا روی CPU و موبایل، ONNX-exportable | معماری غیرخودبازگشتی (NAR) با عملگرهای استاندارد |
| پشتیبانی آینده: چند گوینده، cloning، احساس، streaming | جداسازی speaker-conditioning و adaptor از هسته |
| فارسی | توکنایز کاراکتری روی الفبای موتور زبان فاز ۱ |
| کیفیت صوتی | هدف mel استاندارد + مسیر vocoder قابل تعویض |

## 2. بررسی گزینه‌ها

| معماری | نوع | کیفیت | هزینهٔ آموزش | پیچیدگی | مجوز کد مرجع | سازگاری ONNX | جمع‌بندی |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Tacotron2 | AR + attention | متوسط | بالا (attention ناپایدار) | متوسط | BSD | ضعیف (attention پویا) | رد |
| FastSpeech2 | NAR + duration | خوب | متوسط | کم | MIT | خوب | **پایهٔ انتخاب** |
| VITS | end-to-end (flow + AR prior) | عالی | بالا (هم‌آموزش vocoder) | زیاد | MIT | متوسط | برای v2 |
| Matcha-TTS (flow-matching NAR) | NAR + flow | عالی | متوسط-بالا | متوسط | MIT | متوسط | کاندید v2 |
| Pocket-TTS (مسیر Legacy فعلی) | flow-LM روی latent کدک | خوب (با وزن موجود) | بالا + نیاز به کدک از پیش آموزش‌دیده | زیاد | MIT (کد ما اقتباسی) | ضعیف (WASM خاص) | مرجع مقایسه، نه محصول |
| AVA-N1 (انتخاب) | NAR encoder→duration→decoder→mel | متوسط-خوب (v1) | کم-متوسط | کم | اصل | خوب | **انتخاب v1** |

دلیل انتخاب: FastSpeech-style NAR کم‌ترین ریسک آموزش را دارد، روی CPU آموزش‌پذیر است، خروجی mel استاندارد دارد (vocoder قابل تعویض)، و برای ONNX/اندروید ایده‌آل است. مسیر ارتقا مشخص است: vocoder عصبی (HiFi-GAN-like) → flow-matching decoder (Matcha-style) → cloning.

## 3. معماری AVA-N1

```
متن فارسی
  └─ ava.language.LanguageEngine        نرمال‌سازی کامل فاز ۱
  └─ CharTokenizer (ava/models/native/tokenizer.py)   الفبای کاراکتری + ZWNJ + علائم
  └─ TextEncoder                        ۴ لایه pre-norm Transformer
       │   d=256، heads=4، FFN=1024، pos-emb آموزش‌پذیر + speaker-embedding (بخش‌پذیر)
  └─ DurationPredictor                  ۲ لایه Conv1D + ReLU + LayerNorm → log-duration (MSE)
  └─ LengthRegulator                    تکرار فریم‌ها بر اساس duration پیش‌بینی‌شده/واقعی
  └─ Decoder                            ۴ لایه pre-norm Transformer → linear → 80 mel
  └─ PostNet                            ۵ لایه Conv1D برای پالایش mel
  └─ Vocoder (پلاگین)                   v1: Griffin-Lim؛ v2: vocoder عصبی آموزش‌دیده
  └─ خروجی PCM 24kHz مونو
```

- **هدف آموزش**: `L1(mel_pred, mel_target)` + `MSE(log_dur_pred, log_dur_true)`.
- **شرط‌سازی گوینده**: جدول embedding با جمع‌شدن در خروجی encoder — v1 یک گوینده، ساختار برای چند گوینده آماده.
- **احساس/لحن**: ورودی conditioning رزرو (بردار صفر در v1) — فعال‌سازی بعد از داده.
- **Streaming**: decoder فریم‌محور است؛ شکستن طول‌ها به پنجره‌های ۵۰ فریم با هم‌پوشانی در v2 فعال می‌شود (طراحی امروز سازگار است).
- **long-form**: متن با فاز ۱ جمله‌بندی می‌شود؛ هر جمله یک نمونهٔ آموزشی/استنتاج.

## 4. پارامترهای mel (استاندارد، ثابت محصول)

| پارامتر | مقدار |
| --- | --- |
| sample rate | 24000 Hz |
| n_fft / win / hop | 2048 / 1200 / 300 (۱۲.۵ms) |
| mel bins | 80 |
| fmin / fmax | 80 / 12000 Hz |
| محدوده | log-mel، نرمال‌شده به [-4, 4] |

## 5. مسیر vocoder

| نسخه | vocoder | وضعیت |
| --- | --- | --- |
| v1 (اکنون) | Griffin-Lim (librosa، ISC) | اجرا و تست شده — کیفیت متوسط، برای صحت‌سنجی پایپ‌لاین |
| v2 | vocoder عصبی (generator مشابه HiFi-GAN؛ پیاده‌سازی اصلی + آموزش روی همان دیتاست) | طراحی‌شده؛ نیاز به GPU |

## 6. اثبات صحت در این repo

- تست‌های `tests/test_native_model.py`: roundtrip توکنایزر، forward/backward، کاهش واقعی loss در چند گام، خروجی infer با طول فریم منطقی، تولید WAV واقعی با Griffin-Lim.
- smoke training واقعی در فاز ۴ (`training/train_native.py`) روی دادهٔ ساختگی و گزارش کاهش loss.
- export ONNX واقعی در فاز ۷.
