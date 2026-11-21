# -*- coding: utf-8 -*-
"""AVA Native HTTP API (FastAPI) — the stable contract used by Raya-Hybrid too.

Endpoints:
    GET  /v1/health          liveness + loaded models
    GET  /v1/voices          voice list with license/commercial flags
    GET  /v1/models          loaded model info
    POST /v1/synthesis       {text, voice, speed?} -> audio/wav (or JSON stream metadata)
    POST /v1/synthesis/stream  -> sentence-by-sentence JSON lines (streaming)

Security defaults: text size limit, voice allow-list, no path exposure.
Model loading happens at startup from AVA_NATIVE_BUNDLES env ("id=bundle_dir")
plus optional AVA_NATIVE_LEGACY=1 (non-commercial comparison backend).
"""
from __future__ import annotations

import json
import os
import threading
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from ava.runtime import SynthesisCancelled, SynthesisEngine, SynthesisOptions, wav_bytes

MAX_TEXT_CHARS = int(os.environ.get("AVA_NATIVE_MAX_TEXT", "5000"))
MAX_TEXTS_BATCH = 8

app = FastAPI(title="AVA Native TTS API", version="1.0.0")
engine = SynthesisEngine()
_models_lock = threading.Lock()


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.5)


@app.post("/v1/synthesis", responses={200: {"content": {"audio/wav": {}}}})
def synthesis(req: SynthesisRequest):
    try:
        result = engine.synthesize(req.text, req.voice, SynthesisOptions(speed=req.speed))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'")) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 — surface a clean 500
        raise HTTPException(status_code=500, detail="synthesis failed") from e
    return Response(content=wav_bytes(result), media_type="audio/wav",
                    headers={"X-Duration-Seconds": f"{result.duration_s:.2f}",
                             "X-Sample-Rate": str(result.sample_rate)})


@app.post("/v1/synthesis/stream")
def synthesis_stream(req: SynthesisRequest) -> StreamingResponse:
    def gen() -> Iterator[bytes]:
        try:
            for chunk in engine.synthesize_stream(req.text, req.voice, SynthesisOptions(speed=req.speed)):
                payload = {"voice": chunk.voice, "text": chunk.text,
                           "duration_s": round(chunk.duration_s, 3),
                           "audio_wav_b64": __import__("base64").b64encode(wav_bytes(chunk)).decode()}
                yield (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        except SynthesisCancelled:
            yield (json.dumps({"status": "cancelled"}) + "\n").encode("utf-8")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/v1/voices")
def voices():
    return [{"voice_id": v.voice_id, "display_name": v.display_name, "language": v.language,
             "gender": v.gender, "backend": v.backend, "license": v.license,
             "commercial_ok": v.commercial_ok} for v in engine.list_voices()]


@app.get("/v1/models")
def models():
    return {"models": engine.get_model_info(), "max_text_chars": MAX_TEXT_CHARS}


@app.get("/v1/health")
def health():
    return {"status": "ok", "models": len(engine.get_model_info()),
            "voices": len(engine.list_voices())}


@app.get("/", include_in_schema=False)
def web_ui():
    """Serve the Persian RTL web UI (web-native/index.html)."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    ui = Path(__file__).resolve().parents[2] / "web-native" / "index.html"
    if not ui.is_file():
        raise HTTPException(status_code=404, detail="web UI not found")
    return FileResponse(ui)


def load_bundles_from_env() -> None:
    """AVA_NATIVE_BUNDLES='id1=/path1,id2=/path2' — product bundles (MIT)."""
    spec = os.environ.get("AVA_NATIVE_BUNDLES", "")
    from ava.runtime import NativeN1Backend
    engine.register_backend("ava-n1", NativeN1Backend)
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        mid, _, path = part.partition("=")
        if not path:
            raise ValueError(f"AVA_NATIVE_BUNDLES entry malformed: {part!r}")
        engine.load_model(mid.strip(), "ava-n1", bundle_dir=path)
    if os.environ.get("AVA_NATIVE_LEGACY") == "1":
        from ava.runtime import LegacyPocketTTSBackend
        engine.register_backend("legacy-pocket-tts", LegacyPocketTTSBackend)
        engine.load_model("legacy", "legacy-pocket-tts", config_path="config/fa.yaml")


@app.on_event("startup")
def _startup() -> None:
    try:
        load_bundles_from_env()
    except Exception as e:  # noqa: BLE001 — start anyway; health shows 0 models
        print(f"[ava-api] model loading skipped: {e}")
