# -*- coding: utf-8 -*-
"""اعتبارسنجی metadata بسته پس از اصلاحات closeout فاز ۱.

بررسی‌ها:
1) نسخه، توضیح و کلیدواژه‌ها با هویت AVA Native مستقل هماهنگ‌اند
   (بدون pocket-tts/kyutai در metadata محصول؛ منشأ Legacy فقط در NOTICE/ممیزی).
2) گروه‌های optional واقعی (serve/audio/dev) موجود و قابل نصب‌اند.
3) اسکریپت ورودی ava-cli ثبت شده است.
"""
import tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

proj = data["project"]
name, version = proj["name"], proj["version"]
desc = proj.get("description", "")
kw = [str(k) for k in proj.get("keywords", [])]
meta_blob = (name + " " + desc + " " + " ".join(kw)).lower()

checks = [
    (version == "1.1.1", f"version == 1.1.1 (got {version})"),
    ("pocket" not in meta_blob, "no 'pocket-tts' in package metadata"),
    ("kyutai" not in meta_blob, "no 'kyutai' in package metadata"),
    ("ava-n1" in " ".join(kw), "keywords introduce the product (ava-n1 present)"),
    ("offline" in " ".join(kw), "keywords introduce the product (offline present)"),
    ("pocket" not in desc.lower(), "description has no Pocket-TTS identity"),
    ("serve" in proj.get("optional-dependencies", {}), "optional group [serve] exists"),
    ("audio" in proj.get("optional-dependencies", {}), "optional group [audio] exists"),
    ("dev" in proj.get("optional-dependencies", {}), "optional group [dev] exists"),
    ("api" not in proj.get("optional-dependencies", {}), "no fictitious [api] group"),
    ("ava-cli" in proj.get("scripts", {}), "entry point ava-cli registered"),
]
failed = 0
for ok, label in checks:
    print(("PASS " if ok else "FAIL ") + label)
    failed += 0 if ok else 1

serve = proj["optional-dependencies"]["serve"]
assert any(d.startswith("fastapi") for d in serve) and any(d.startswith("uvicorn") for d in serve)
print("PASS [serve] provides fastapi + uvicorn (matches RUNTIME_INSTALL.md)")
raise SystemExit(1 if failed else 0)
