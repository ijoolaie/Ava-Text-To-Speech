# AVA Native — Windows Runtime (فاز ۷)

وضعیت: کد پایپ‌لاین کاملاً cross-platform نوشته شده (pathlib، بدون مسیر لینوکسی، UTF-8 صریح)؛ **آزمون روی ویندوز واقعی = تحویل‌نشده** (sandbox لینوکسی است). مسیرهای زیر برای اجرا روی Windows آماده است.

## نصب

```bat
:: Python 3.10+ و محیط مجاز
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e ".[serve]"
```

## اجرا

```bat
:: CLI با باندل آموزش‌دیده
set AVA_NATIVE_BUNDLES=n1=runs\native_v1\bundle
ava-cli voices
ava-cli speak --text "سلام دنیا" --voice ava-n1-ava --out out.wav

:: سرور API + رابط وب
ava-cli serve --host 127.0.0.1 --port 8300 --bundle runs\native_v1\bundle
:: سپس در مرورگر: http://127.0.0.1:8300/
```

`scripts/windows/run_ava.bat` همین کار را خودکار می‌کند.

## نکات ویندوز

- کدنویسی فایل‌ها: همهٔ خواندن/نوشتن‌ها با `encoding="utf-8"` صریح انجام می‌شود.
- مسیرها: فقط `pathlib`/`os.path` — جداکنندهٔ `\` پشتیبانی می‌شود.
- صداها و باندل‌ها مسیر نسبی به CWD دارند؛ در سرویس ویندوز مسیر مطلق بدهید (`AVA_NATIVE_BUNDLES=n1=C:\ava\bundle`).
- صوت: خروجی 24kHz mono PCM-16 — پخش با هر player یا `sounddevice`.
