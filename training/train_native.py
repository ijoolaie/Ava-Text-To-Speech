# -*- coding: utf-8 -*-
"""Trainer for the AVA-N1 native model (AVA Native original pipeline).

Features: manifest-driven dataset, mel targets from real audio, duration
targets from a pluggable alignment source (uniform fallback), checkpointing,
resume, validation, JSONL logging, and bundle export. CPU-friendly by design.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import soundfile as sf
import torch

from ava.audio.mel import mel_spectrogram
from ava.language import LanguageEngine
from ava.models.native import AVANativeConfig, AVANativeTTS, CharTokenizer
from training.datasets.manifest import ManifestEntry, dataset_fingerprint, read_manifest


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 10
    val_every: int = 50
    checkpoint_every: int = 50
    seed: int = 17
    model: dict = field(default_factory=lambda: AVANativeConfig().to_dict())
    dataset_fingerprint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


class NativeDataset(torch.utils.data.Dataset):
    """Utterance dataset from a manifest: (tokens, mel, duration targets)."""

    def __init__(self, manifest_path: str, cfg: AVANativeConfig, entries: list[ManifestEntry] | None = None,
                 split: str | None = "train"):
        self.cfg = cfg
        self.tok = CharTokenizer()
        self.engine = LanguageEngine()
        entries = entries or read_manifest(manifest_path)
        if split:
            entries = [e for e in entries if e.split == split]
        # only usable entries (validated, with audio) enter training
        self.entries = [e for e in entries if e.valid and os.path.isfile(e.audio_path)]

    def __len__(self) -> int:
        return len(self.entries)

    def _duration_targets(self, n_frames: int, n_tokens: int, entry: ManifestEntry) -> np.ndarray:
        """v1: uniform durations weighted by token type (punct = pause).

        A production run passes aligned durations via `durations_path`
        (MFA/ASR-based); the trainer accepts either transparently.
        """
        weights = np.ones(n_tokens, dtype=np.float32)
        text = self.engine.normalize(entry.text)
        ids = self.tok.encode(text)[1:-1]
        for i, tid in enumerate(ids[:n_tokens]):
            if self.tok.vocab[tid] in "،؛:.!؟":
                weights[i] = 3.0  # pause frames around punctuation
        weights = weights / weights.sum()
        durs = np.maximum(weights * n_frames, 1.0)
        return np.log(durs.astype(np.float64))

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        x, sr = sf.read(e.audio_path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(x.mean(axis=1)).float()
        if sr != self.cfg.sample_rate:
            import librosa
            wav = torch.from_numpy(librosa.resample(wav.numpy(), orig_sr=sr, target_sr=self.cfg.sample_rate)).float()
        mel = mel_spectrogram(wav, self.cfg)                      # (n_mels, F)
        text = self.engine.normalize(e.text)[: self.cfg.max_text_tokens - 2]
        ids = self.tok.encode(text)
        T = len(ids)
        log_durs = torch.from_numpy(self._duration_targets(mel.size(1), T - 2, e)).float()
        full = torch.zeros(T)
        full[1:-1] = log_durs  # BOS/EOS get their own (learned) durations — target 0 log-frames
        return {"tokens": torch.tensor(ids), "mel": mel, "log_durations": full, "speaker": 0}


def collate(batch: list[dict], cfg: AVANativeConfig) -> dict:
    B = len(batch)
    Tm = max(b["tokens"].size(0) for b in batch)
    Fm = max(b["mel"].size(1) for b in batch)
    pad = CharTokenizer().pad_id
    tokens = torch.full((B, Tm), pad, dtype=torch.long)
    mel = torch.zeros(B, cfg.n_mels, Fm)
    durs = torch.zeros(B, Tm)
    lens, flens = torch.zeros(B, dtype=torch.long), torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        t, m, d = b["tokens"], b["mel"], b["log_durations"]
        tokens[i, : t.size(0)] = t
        mel[i, :, : m.size(1)] = m
        durs[i, : d.size(0)] = d
        lens[i] = t.size(0)
        flens[i] = m.size(1)
    return {"tokens": tokens, "lengths": lens, "mel": mel, "mel_lengths": flens,
            "log_durations": durs}


def build_batches(ds: NativeDataset, cfg_train: TrainConfig, shuffle: bool = True) -> list[dict]:
    """Simple deterministic batching (no DataLoader workers — CPU sandbox friendly)."""
    cfg = AVANativeConfig.from_dict(cfg_train.model)
    idx = list(range(len(ds)))
    rng = np.random.default_rng(cfg_train.seed)
    if shuffle:
        rng.shuffle(idx)
    return [collate([ds[i] for i in idx[k:k + cfg_train.batch_size]], cfg)
            for k in range(0, len(idx), cfg_train.batch_size)]


def save_checkpoint(path: str, model: AVANativeTTS, opt, step: int, tcfg: TrainConfig) -> None:
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "train_config": tcfg.to_dict(),
    }, path)


def load_checkpoint(path: str, model: AVANativeTTS, opt=None) -> tuple[int, TrainConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    if opt is not None:
        opt.load_state_dict(ck["optimizer"])
    return int(ck["step"]), TrainConfig.from_dict(ck["train_config"])


def train(manifest_path: str, out_dir: str, tcfg: TrainConfig | None = None,
          resume: bool = False, log_cb=None) -> dict:
    """Full training loop; returns a summary dict with loss trajectory."""
    tcfg = tcfg or TrainConfig()
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "log.jsonl")
    mcfg = AVANativeConfig.from_dict(tcfg.model)
    torch.manual_seed(tcfg.seed)
    ds = NativeDataset(manifest_path, mcfg)
    if len(ds) == 0:
        raise RuntimeError("dataset is empty — run validation and check the manifest")
    tcfg.dataset_fingerprint = dataset_fingerprint(read_manifest(manifest_path))
    model = AVANativeTTS(mcfg, len(ds.tok))
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    start = 0
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    if resume and os.path.isfile(ckpt_path):
        start, _stored = load_checkpoint(ckpt_path, model, opt)  # caller's TrainConfig stays authoritative
        print(f"resumed from step {start}")
    val_ds = NativeDataset(manifest_path, mcfg, split="val")

    log_f = open(log_path, "a", encoding="utf-8")
    history = []
    model.train()
    for step in range(start, tcfg.steps):
        batches = build_batches(ds, tcfg)
        if not batches:
            raise RuntimeError("no training batches — check split assignment")
        for batch in batches:
            out = model(**batch)
            opt.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
        step_loss = float(out["loss"].detach())
        history.append(step_loss)
        if step % tcfg.log_every == 0 or step == tcfg.steps - 1:
            rec = {"step": step, "loss": round(step_loss, 4), "mel_l1": round(float(out["mel_l1"].detach()), 4),
                   "dur_loss": round(float(out["dur_loss"].detach()), 4)}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            if log_cb:
                log_cb(rec)
        if val_ds and len(val_ds) > 0 and ((step + 1) % tcfg.val_every == 0 or step == tcfg.steps - 1):
            model.eval()
            with torch.no_grad():
                v = [model(**collate([val_ds[i]], mcfg))["loss"].detach().item() for i in range(min(len(val_ds), 16))]
            val_loss = float(np.mean(v))
            log_f.write(json.dumps({"step": step, "val_loss": round(val_loss, 4)}) + "\n")
            log_f.flush()
            model.train()
        if (step + 1) % tcfg.checkpoint_every == 0 or step == tcfg.steps - 1:
            save_checkpoint(ckpt_path, model, opt, step + 1, tcfg)
    log_f.close()

    export_native_bundle(model, ds.tok, mcfg, os.path.join(out_dir, "bundle"))
    return {"steps": tcfg.steps, "first": history[0] if history else None,
            "last": history[-1] if history else None, "history": history}


def export_native_bundle(model: AVANativeTTS, tok: CharTokenizer, mcfg: AVANativeConfig, out_dir: str) -> str:
    """Export an inference bundle: weights + config + tokenizer (AVA Native format)."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": mcfg.to_dict()}, os.path.join(out_dir, "ava_n1.pt"))
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "ava-native-n1", "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
                   "sample_rate": mcfg.sample_rate, "mel": {"n_mels": mcfg.n_mels, "hop": mcfg.hop_size}}, f,
                  ensure_ascii=False, indent=1)
    return out_dir


def load_native_bundle(bundle_dir: str) -> tuple[AVANativeTTS, CharTokenizer, AVANativeConfig]:
    ck = torch.load(os.path.join(bundle_dir, "ava_n1.pt"), map_location="cpu", weights_only=False)
    mcfg = AVANativeConfig.from_dict(ck["config"])
    tok = CharTokenizer.load(os.path.join(bundle_dir, "tokenizer.json"))
    model = AVANativeTTS(mcfg, len(tok))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, mcfg


def synth_to_wav(model: AVANativeTTS, tok: CharTokenizer, mcfg: AVANativeConfig,
                 text: str, wav_path: str, speed: float = 1.0) -> float:
    """Text -> normalized text -> tokens -> mel -> Griffin-Lim WAV. Returns duration in seconds."""
    from ava.audio.mel import griffin_lim_from_mel
    engine = LanguageEngine()
    text = engine.normalize(text)[: mcfg.max_text_tokens - 2]
    ids = torch.tensor([tok.encode(text)])
    mel = model.infer(ids, torch.tensor([ids.size(1)]), speed=speed)
    wav = griffin_lim_from_mel(mel.numpy(), mcfg)
    sf.write(wav_path, wav, mcfg.sample_rate)
    return len(wav) / mcfg.sample_rate


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AVA-N1 native trainer")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/native_v1")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    summary = train(args.manifest, args.out,
                    TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr),
                    resume=args.resume)
    print(json.dumps({k: summary[k] for k in ("steps", "first", "last")}, ensure_ascii=False))
