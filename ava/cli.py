# -*- coding: utf-8 -*-
"""ava-cli — the single command-line entry point of AVA Native.

Subcommands:
    speak            text -> WAV file
    voices           list available voices
    serve            run the HTTP API
    validate-dataset dataset manifest validation + provenance report
    voice-quality    measure a voice prompt WAV

Cross-platform: pure pathlib/argparse; works on Linux, macOS and Windows.
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_engine(bundle_dir: str | None):
    import os
    from ava.runtime import NativeN1Backend, SynthesisEngine
    engine = SynthesisEngine()
    engine.register_backend("ava-n1", NativeN1Backend)
    spec = os.environ.get("AVA_NATIVE_BUNDLES", "")
    paths: dict[str, str] = {}
    for part in spec.split(","):
        if "=" in part:
            k, _, v = part.strip().partition("=")
            paths[k.strip()] = v.strip()
    if bundle_dir:
        paths.setdefault("n1", bundle_dir)
    for mid, path in paths.items():
        engine.load_model(mid, "ava-n1", bundle_dir=path)
    return engine


def cmd_serve(args) -> int:
    if args.bundle:
        import os
        os.environ.setdefault("AVA_NATIVE_BUNDLES", f"n1={args.bundle}")
    from ava.api.server import app, load_bundles_from_env
    load_bundles_from_env()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_speak(args) -> int:
    engine = _load_engine(args.bundle)
    voice = args.voice
    if voice not in {v.voice_id for v in engine.list_voices()}:
        available = ", ".join(v.voice_id for v in engine.list_voices())
        print(f"voice {voice!r} not found. available: {available}", file=sys.stderr)
        return 2
    result = engine.synthesize(args.text, voice)
    import soundfile as sf
    sf.write(args.out, result.audio, result.sample_rate)
    print(f"{result.duration_s:.2f}s audio in {result.elapsed_s:.2f}s (RTF {result.rtf:.2f}x) -> {args.out}")
    return 0


def cmd_voices(args) -> int:
    engine = _load_engine(args.bundle)
    for v in engine.list_voices():
        flag = "commercial" if v.commercial_ok else "NON-COMMERCIAL"
        print(f"{v.voice_id}\t{v.display_name}\t[{v.backend}] {flag} {v.license}")
    if not engine.list_voices():
        print("no voices — pass --bundle or set AVA_NATIVE_BUNDLES", file=sys.stderr)
    return 0


def cmd_validate_dataset(args) -> int:
    from training.datasets.manifest import provenance_report, read_manifest
    from training.datasets.validate import ValidationLimits, validate_audio
    entries = read_manifest(args.manifest)
    limits = ValidationLimits()
    for e in entries:
        validate_audio(e, limits)
    report = provenance_report(entries)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def cmd_voice_quality(args) -> int:
    from ava.voices import check_voice_prompt
    rep = check_voice_prompt(args.wav)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ava-cli", description="AVA Native — Persian TTS")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("speak", help="text -> WAV")
    p.add_argument("--text", required=True)
    p.add_argument("--voice", default="ava-n1-ava")
    p.add_argument("--out", default="out.wav")
    p.add_argument("--bundle", default="", help="path to an AVA-N1 bundle (or AVA_NATIVE_BUNDLES env)")
    p.add_argument("--speed", type=float, default=1.0)
    p.set_defaults(fn=lambda a: (setattr(a, "speed", a.speed), cmd_speak(a))[1])

    p = sub.add_parser("voices", help="list voices")
    p.add_argument("--bundle", default="")
    p.set_defaults(fn=cmd_voices)

    p = sub.add_parser("serve", help="run HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--bundle", default="")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("validate-dataset", help="manifest validation + provenance")
    p.add_argument("--manifest", required=True)
    p.set_defaults(fn=cmd_validate_dataset)

    p = sub.add_parser("voice-quality", help="measure a voice prompt")
    p.add_argument("--wav", required=True)
    p.set_defaults(fn=cmd_voice_quality)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
