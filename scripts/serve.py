#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""سرور محلی دموی مرورگری آوا.

پوشهٔ web/ را روی http://localhost:8000 سرو می‌کند و مسیرهای /model/* را به
پوشهٔ weights/ نگاشت می‌دهد (فایل merge‌شدهٔ model.safetensors را در صورت نبود
می‌سازد). هدرهای COOP/COEP برای نسخهٔ چندنخی WASM ارسال می‌شوند.

اجرا:  python scripts/serve.py [پورت]
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = REPO_ROOT / "web"
MODEL_MERGED = WEB_ROOT / "model.safetensors"

# URL path (under /model/) -> file in weights/
MAPPING = {
    "/model/tokenizer.json": REPO_ROOT / "weights" / "tokenizer.json",
    "/model/config.json": REPO_ROOT / "weights" / "model_config.json",
}

HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "cross-origin",
}


def ensure_merged_weights() -> Path:
    """Merge the sharded weights into the single file the WASM runtime expects."""
    if MODEL_MERGED.exists():
        return MODEL_MERGED
    sys.path.insert(0, str(REPO_ROOT))
    from ava.utils.weights_io import load_file

    import safetensors.torch

    state = load_file(REPO_ROOT / "weights" / "model.safetensors")
    safetensors.torch.save_file(state, MODEL_MERGED)
    print(f"model.safetensors merge شد ({MODEL_MERGED.stat().st_size / 1e6:.1f} MB)")
    return MODEL_MERGED


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        for k, v in HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        resolved = MAPPING.get(self.path.split("?")[0])
        if resolved is not None and resolved.exists():
            return self._serve_file(resolved)
        if self.path.startswith("/model/voices/"):
            name = Path(self.path).name
            candidate = REPO_ROOT / "weights" / "voices" / name
            if candidate.exists():
                return self._serve_file(candidate)
        if self.path.rstrip("/") == "/model/model.safetensors":
            return self._serve_file(ensure_merged_weights())
        return super().send_head()

    def _serve_file(self, path: Path):
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return None
        self.send_response(http.HTTPStatus.OK)
        ctype = {
            ".safetensors": "application/octet-stream",
            ".json": "application/json",
            ".wasm": "application/wasm",
        }.get(path.suffix, "application/octet-stream")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        return f


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "8000"))
    url = f"http://localhost:{port}/"
    print(f"دموی آوا روی {url} — برای توقف Ctrl+C")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with Server(("0.0.0.0", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nمتوقف شد.")


if __name__ == "__main__":
    main()
