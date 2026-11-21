# AVA Native — Android Runtime (فاز ۷)

وضعیت: **مسیر فنی طراحی و اثبات‌شده در سطح export؛ اجرای روی دستگاه واقعی = تحویل‌نشده** (در sandbox دستگاه اندروید وجود ندارد). همهٔ اجزای لازم آماده است: `ava_n1.onnx` با موفقیت export و با onnxruntime اجرا و با مسیر torch مقایسه شد.

## 1. معماری پیشنهادی روی Android

| لایه | انتخاب | دلیل |
| --- | --- | --- |
| موتور استنتاج | **ONNX Runtime Mobile** (Java/Kotlin API) | export ما استاندارد ONNX opset-17 است؛ اجرا با CPU EP؛ EP خطی NNAPI/XNNPACK برای سرعت |
| بلوک‌های صوتی | mel→WAV با کد JNI باریک (Griffin-Lim متن‌باز خودمان به C++) یا vocoder عصبی ONNX در v2 | Griffin-Lim روی موبایل سبک است (~۱۰۰ms) |
| موتور زبان | همان `ava.language` به‌صورت port شده به Kotlin (نرمال‌سازی/اعداد/تاریخ) یا اجرای آن در سمت سرور قبل از export | توکنایز کاراکتری ساده است |
| دانلود مدل | DownloadManager + checksum (SHA-256 فایل باندل) | مدل‌ها خارج از APK برای کاهش حجم |
| پخش | ExoPlayer با `ProgressiveMediaSource` روی WAV حاصل از جمله‌ها؛ streaming جمله‌به‌جمله | `synthesize_stream` سمت سرور/دستگاه جمله‌ها را جدا تولید می‌کند |
| باتری | batch synthesis غیرفعال؛ WakeLock فقط حین تولید؛ تولید در Doze ممنوع | |

## 2. الزامات منابع (برآورد)

- مدل AVA-N1 v0: ~2-8MB ONNX (بسته به d_model) + توکنایز ۲KB
- حافظهٔ استنتاج: < 150MB RSS برای متن ≤ 128 توکن (بودجهٔ ثابت export)
- تأخیر: < 1s برای جملهٔ معمول روی گوشی میان‌رده (XNNPACK، ۴ رشته)

## 3. مراحل تحویل واقعی Android (نیازمند دستگاه)

1. `onnxruntime-android` (aar) به پروژه اضافه شود.
2. `ava_n1.onnx` + `tokenizer.json` از باندل دانلود و با checksum صحت‌سنجی شود.
3. `OrtSession.run(tokens, lengths, speed)` → mel → mel→PCM (JNI) → ExoPlayer.
4. بنچمارک روی دستگاه واقعی + آزمون آفلاین ( airplane mode).

قرارداد API همان `docs/RAYA_HYBRID_INTEGRATION.md` است؛ اپ اندروید می‌تواند هم محلی (session داخلی) و هم به API سرور وصل شود.
