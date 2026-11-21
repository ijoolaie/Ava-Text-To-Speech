# -*- coding: utf-8 -*-
"""AVA Native benchmark — RTF, latency and memory (AC9.3).

Usage: python3 scripts/benchmark.py --bundle runs/native_smoke_v1/bundle
Reports real numbers only; legacy adapter measured when weights exist.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TEXTS = ["سلام.", "این یک جملهٔ آزمایشی است.", "قیمت این کتاب دویست و پنجاه هزار تومان است و دوازده درصد تخفیف دارد."]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # linux: KB


def bench_backend(engine, voice_id: str, name: str) -> dict:
    rows = []
    for text in TEXTS:
        t0 = time.time()
        r = engine.synthesize(text, voice_id)
        rows.append({"chars": len(text), "audio_s": round(r.duration_s, 2),
                     "elapsed_s": round(r.elapsed_s, 3), "rtf": round(r.rtf, 2)})
    return {"backend": name, "voice": voice_id, "rows": rows,
            "avg_rtf": round(sum(x["rtf"] for x in rows) / len(rows), 2),
            "max_rss_mb": round(rss_mb(), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--legacy", action="store_true", help="also benchmark the legacy adapter (non-commercial)")
    args = ap.parse_args()

    from ava.runtime import NativeN1Backend, SynthesisEngine
    out = {"product": "AVA Native", "sample_rate": 24000, "channels": 1, "results": []}
    engine = SynthesisEngine()
    engine.register_backend("ava-n1", NativeN1Backend)
    engine.load_model("n1", "ava-n1", bundle_dir=args.bundle)
    out["results"].append(bench_backend(engine, "ava-n1-ava", "ava-n1"))

    if args.legacy and os.path.isdir(os.path.join(ROOT, "weights", "voices")):
        from ava.runtime import LegacyPocketTTSBackend
        engine.register_backend("legacy-pocket-tts", LegacyPocketTTSBackend)
        engine.load_model("legacy", "legacy-pocket-tts", config_path=os.path.join(ROOT, "config", "fa.yaml"))
        out["results"].append(bench_backend(engine, "legacy-shirin", "legacy-pocket-tts (CC BY-NC, comparison only)"))

    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
