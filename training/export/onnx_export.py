# -*- coding: utf-8 -*-
"""ONNX export of the AVA-N1 native model (phase 7 — Android/Windows path).

Exports text tokens + lengths + speed -> mel + total_frames, using a
vectorized (scatter-free) length regulator that ONNX can represent.
Verified against the torch implementation with onnxruntime.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

from ava.models.native import AVANativeConfig, AVANativeTTS


class VectorizedLengthRegulator(nn.Module):
    """Frame expansion via index computation — no per-batch Python loop.

    log_dur must already be masked (padded positions = 0 → reps = 0).
    """

    def forward(self, enc: torch.Tensor, log_dur: torch.Tensor, max_frames: int):
        # enc: (1, T, D), log_dur: (1, T) — masked (pads have log_dur == 0)
        # every real token keeps >= 1 frame (matching the torch path); pads give 0.
        reps = (torch.round(torch.exp(log_dur)).clamp(min=1) * (log_dur != 0).float()).long()  # (1, T)
        ends = torch.cumsum(reps, dim=1)                                     # (1, T)
        total = ends[:, -1:]                                                 # (1, 1)
        idx = torch.arange(max_frames, device=enc.device).view(1, max_frames, 1)
        # token index for every frame: number of token ends <= frame index
        tok_index = torch.sum(idx < ends.unsqueeze(1), dim=-1).clamp(max=enc.size(1) - 1)  # (1, F)
        out = torch.gather(enc, 1, tok_index.unsqueeze(-1).expand(-1, -1, enc.size(2)))
        frame_mask = (torch.arange(max_frames, device=enc.device).view(1, max_frames) < total).float()
        return out * frame_mask.unsqueeze(-1), total


class AVAN1ExportWrapper(nn.Module):
    def __init__(self, model: AVANativeTTS):
        super().__init__()
        self.model = model
        self.reg = VectorizedLengthRegulator()

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, speed: torch.Tensor):
        enc = self.model.encoder(tokens, lengths)
        log_dur = self.model.duration(enc) - torch.log(speed.clamp(min=0.05))
        # padded tokens contribute no frames in the export path
        tmask = (torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        log_dur = log_dur * tmask
        reg, total = self.reg(enc, log_dur, self.model.cfg.max_frames)
        _, mel_post = self.model.decoder(reg)
        return mel_post, total


class AVAN1ParityWrapper(nn.Module):
    """Same graph with EXTERNAL durations — for bit-close parity verification.

    The two float engines (torch / ONNX) may round a duration near .5 differently,
    so exact parity is verified by feeding identical durations; the deployment
    export (predictor inside) is checked structurally.
    """

    def __init__(self, model: AVANativeTTS):
        super().__init__()
        self.model = model
        self.reg = VectorizedLengthRegulator()

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor, log_dur_in: torch.Tensor):
        enc = self.model.encoder(tokens, lengths)
        tmask = (torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        reg, total = self.reg(enc, log_dur_in * tmask, self.model.cfg.max_frames)
        _, mel_post = self.model.decoder(reg)
        return mel_post, total


def export_onnx(bundle_dir: str, out_path: str | None = None, opset: int = 17,
                max_tokens: int = 128) -> str:
    """Static-shape export (mobile-friendly). Falls back to the legacy exporter.

    `max_tokens` fixes the input budget; callers pad token ids to it.
    """
    from training.train_native import load_native_bundle
    model, tok, cfg = load_native_bundle(bundle_dir)
    wrapper = AVAN1ExportWrapper(model).eval()
    out_path = out_path or os.path.join(bundle_dir, "ava_n1.onnx")
    tokens = torch.randint(0, len(tok), (1, max_tokens))
    lengths = torch.tensor([8])
    speed = torch.tensor([1.0])
    common = dict(
        input_names=["tokens", "lengths", "speed"],
        output_names=["mel", "total_frames"],
        opset_version=opset, do_constant_folding=True,
    )
    try:
        torch.onnx.export(wrapper, (tokens, lengths, speed), out_path, **common)
    except Exception:
        torch.onnx.export(wrapper, (tokens, lengths, speed), out_path, dynamo=False, **common)
    return out_path


def verify_onnx(bundle_dir: str, onnx_path: str | None = None, text: str = "سلام دنیا") -> dict:
    """Two-part verification:
    1. bit-close parity of the exported graph with identical duration inputs;
    2. structural check of the deployment export (predictor inside the graph).
    """
    import onnxruntime as ort
    from training.train_native import load_native_bundle
    model, tok, cfg = load_native_bundle(bundle_dir)
    onnx_path = onnx_path or os.path.join(bundle_dir, "ava_n1.onnx")
    ids = tok.encode(text)

    # ---- part 1: parity with fixed durations --------------------------------
    parity_path = os.path.join(bundle_dir, "ava_n1_parity.onnx")
    parity = AVAN1ParityWrapper(model).eval()
    max_tokens = 64
    ids_pad = ids[:max_tokens] + [tok.pad_id] * max(0, max_tokens - len(ids))
    torch.manual_seed(3)
    log_dur_in = torch.randn(1, max_tokens) * 0.4 + 1.3  # ~3.7 frames/token
    try:
        torch.onnx.export(parity, (torch.zeros(1, max_tokens, dtype=torch.long),
                                   torch.tensor([len(ids)]), log_dur_in), parity_path,
                          input_names=["tokens", "lengths", "log_dur_in"],
                          output_names=["mel", "total_frames"],
                          opset_version=17, do_constant_folding=True)
        sess_p = ort.InferenceSession(parity_path, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            mel_t, total_t = parity(torch.tensor([ids_pad]), torch.tensor([len(ids)]), log_dur_in)
        mel_o, total_o = sess_p.run(None, {
            "tokens": torch.tensor([ids_pad]).numpy().astype("int64"),
            "lengths": np.array([len(ids)], dtype="int64"),
            "log_dur_in": log_dur_in.numpy().astype("float32")})
        cut = min(int(total_t.reshape(-1)[0]), int(np.asarray(total_o).reshape(-1)[0]))
        parity_diff = float(abs(mel_o[0, :, :cut] - mel_t[0, :, :cut].detach().numpy()).max())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"parity export failed: {e}"}

    # ---- part 2: deployment export runs standalone --------------------------
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dep_max = sess.get_inputs()[0].shape[1]
    ids_pad_dep = ids[:dep_max] + [tok.pad_id] * max(0, dep_max - len(ids))
    with torch.no_grad():
        mel_torch = model.infer(torch.tensor([ids]), torch.tensor([len(ids)]), speed=1.0)
    mel_onnx, total = sess.run(None, {
        "tokens": torch.tensor([ids_pad_dep]).numpy().astype("int64"),
        "lengths": np.array([len(ids)], dtype="int64"),
        "speed": np.array([1.0], dtype="float32")})
    total = int(total.reshape(-1)[0])
    frame_gap = abs(total - int(mel_torch.size(1)))
    return {"parity_max_abs_diff": parity_diff, "parity_ok": parity_diff < 1e-3,
            "torch_frames": int(mel_torch.size(1)), "onnx_frames": total,
            "frame_gap": frame_gap,
            "ok": parity_diff < 1e-3 and frame_gap <= 2 and total > len(ids)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="export AVA-N1 to ONNX")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    path = export_onnx(args.bundle, args.out or None)
    print("exported:", path)
    print(verify_onnx(args.bundle, path))
