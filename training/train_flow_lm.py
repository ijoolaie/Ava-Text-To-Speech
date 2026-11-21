# -*- coding: utf-8 -*-
"""آموزش flow LM مدل آوا روی لاتنت‌های کدک منجمد.

    python -m training.train_flow_lm --config training/configs/ava_fa_flow_lm.yaml
    python -m training.train_flow_lm --config training/configs/smoke_cpu.yaml

روایت Kyutai: کدک Mimi منجمد است؛ فقط flow LM (و projection صدای گوینده)
روی لاتنت‌های پیوستهٔ ۱۲.۵ هرتزی آموزش می‌بیند. هدف «drifting» یک‌مرحله‌ای
به‌همراه سر EOS دودویی است (توضیح کامل در training/flow_lm_training.py).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ava.models.flow_lm import FlowLMModel  # noqa: E402
from ava.utils.config import load_config as load_arch_config  # noqa: E402
from ava.utils.utils import resolve_path  # noqa: E402
from ava.utils.weights_io import load_file  # noqa: E402
from training.data.dataset import Collator, LatentDataset  # noqa: E402
from training.flow_lm_training import flow_lm_loss, update_stats_ema  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(arch_path: Path, init_from: str | None, train_speaker_proj: bool):
    arch = load_arch_config(arch_path)
    # Resolve the tokenizer path against the repo root so the run is cwd-independent.
    arch.flow_lm.lookup_table.tokenizer_path = str(
        resolve_path(arch.flow_lm.lookup_table.tokenizer_path, REPO_ROOT)
    )
    model = FlowLMModel.from_pydantic_config(
        arch.flow_lm, latent_dim=arch.mimi.quantizer.dimension,
        insert_bos_before_voice=arch.flow_lm.insert_bos_before_voice,
    )
    # The voice-conditioning projection (raw codec latents -> transformer dim). The
    # released recipe trains it with the clip conditioning on its own audio.
    model.speaker_proj_weight = torch.nn.Parameter(
        torch.zeros(
            (arch.flow_lm.transformer.d_model,
             arch.mimi.inner_dim or arch.mimi.seanet.dimension),
            dtype=torch.float32,
        )
    )
    if init_from:
        bundle = load_file(resolve_path(init_from, REPO_ROOT))
        state = {k.removeprefix("flow_lm."): v for k, v in bundle.items() if k.startswith("flow_lm.")}
        model.load_state_dict(state, strict=True)
        print(f"fine-tune: بارگذاری {len(state)} تنسور از {init_from}")
    if not train_speaker_proj:
        model.speaker_proj_weight.requires_grad_(False)
    return model, arch


def param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".embed.weight") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def lr_at(step: int, cfg: dict) -> float:
    warmup = cfg["warmup_steps"]
    total = cfg["total_steps"]
    if step < warmup:
        return cfg["lr"] * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    t = min(t, 1.0)
    min_ratio = cfg.get("cosine_min_ratio", 0.002)
    return cfg["lr"] * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] | None = None

    def update(self, model: torch.nn.Module):
        if self.shadow is None:
            self.shadow = {
                k: v.detach().clone().float() for k, v in model.state_dict().items()
            }
            return
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k not in self.shadow or not torch.is_floating_point(v):
                    continue
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {f"flow_lm.{k}": v for k, v in (self.shadow or {}).items()}


def save_checkpoint(out_dir: Path, step: int, model, ema: EMA, optimizer, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {f"flow_lm.{k}": v for k, v in model.state_dict().items()},
        "ema": ema.state_dict(),
        "step": step,
        "config": cfg["_arch_path_str"],
        "stats": {
            "latent_mean": model.emb_mean.detach().tolist(),
            "latent_std": model.emb_std.detach().tolist(),
        },
        "optimizer": optimizer.state_dict(),
    }
    path = out_dir / f"checkpoint-{step:07d}.pt"
    torch.save(payload, path)
    latest = out_dir / "checkpoint-latest.pt"
    torch.save(payload, latest)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="آموزش flow LM آوا")
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="بازنویسی مقادیر کانفیگ، مثل train.total_steps=50")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for ov in args.override:
        key, _, value = ov.partition("=")
        node = cfg
        keys = key.split(".")
        for k in keys[:-1]:
            node = node[k]
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value  # bare string
        node[keys[-1]] = parsed
    arch_file = resolve_path(cfg["model"]["config"], cfg_path.parent)
    cfg["_arch_path_str"] = str(arch_file.resolve())

    set_seed(cfg["seed"])
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = cfg.get("precision", "fp32")
    autocast_dtype = torch.bfloat16 if (precision == "bf16" and device == "cuda") else None
    print(f"device={device} precision={precision}")

    model, arch = build_model(
        Path(cfg["_arch_path_str"]), cfg["model"].get("init_from"),
        cfg["model"].get("train_speaker_proj", True),
    )
    model.to(device).train()

    tokenizer = model.conditioner.tokenizer

    d = cfg["data"]
    common = dict(
        cache_dir=REPO_ROOT / d["cache_dir"], tokenizer=tokenizer,
        max_frames=d.get("max_frames", 250), max_tokens=d.get("max_tokens", 64),
        min_frames=d.get("min_frames", 10), seed=cfg["seed"],
    )
    train_ds = LatentDataset(holdout_ratio=d.get("holdout_ratio", 0.0), holdout=False, **common)
    eval_ds = LatentDataset(holdout_ratio=d.get("holdout_ratio", 0.0), holdout=True, **common)
    if len(train_ds) == 0:
        sys.exit(f"دیتاست خالی است: {d['cache_dir']} را با training.data.prepare_data بسازید.")
    collate = Collator(ldim=arch.mimi.quantizer.dimension,
                       pad_id=arch.flow_lm.lookup_table.n_bins)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0), collate_fn=collate, drop_last=False,
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate,
    ) if len(eval_ds) else None

    # Latent stats: initialize from the dataset stats, then keep the training EMA.
    mean = torch.tensor(train_ds.stats.get("latent_mean"), dtype=torch.float32)
    std = torch.tensor(train_ds.stats.get("latent_std"), dtype=torch.float32)
    if model.emb_mean.shape == mean.shape and cfg["model"].get("init_from") is None:
        model.emb_mean.copy_(mean.to(device))
        model.emb_std.copy_(std.to(device))
        print("emb_mean/emb_std از آمارهٔ دیتاست مقداردهی شد")

    ocfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        param_groups(model, ocfg["weight_decay"]), lr=ocfg["lr"], betas=tuple(ocfg["betas"])
    )
    ema = EMA(model, cfg["ema"]["decay"])

    out_dir = REPO_ROOT / cfg["train"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, indent=2, default=str),
        encoding="utf-8",
    )
    log_path = out_dir / "train_log.jsonl"

    total_steps = ocfg["total_steps"]
    accum = cfg["train"].get("grad_accum", 1)
    step = 0
    accum_count = 0
    t0 = time.time()
    history = []
    running = {}
    print(f"آموزش: {len(train_ds)} کلیپ، {total_steps} گام، batch {cfg['train']['batch_size']}")

    data_iter = iter(train_dl)
    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, ocfg)

        ctx = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and device == "cuda"
            else torch.enable_grad()
        )
        with ctx:
            loss, metrics = flow_lm_loss(
                model, batch,
                noise_std=cfg["loss"]["noise_std"],
                eos_loss_weight=cfg["loss"]["eos_loss_weight"],
                latent_loss_weight=cfg["loss"]["latent_loss_weight"],
                voice_cond_normalized=bool(arch.voice_cond_normalized),
            )
        (loss / accum).backward()
        accum_count += 1
        if accum_count < accum:
            continue
        accum_count = 0

        grad_clip = ocfg.get("grad_clip")
        if grad_clip:
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Moving latent statistics (only meaningful once gradients moved the head).
        raw_targets = batch["targets"].to(model.bos_emb.device)
        mask = batch["mask"].to(model.bos_emb.device)
        update_stats_ema(model, raw_targets, mask, cfg["stats"]["ema_decay"])

        if step >= cfg["ema"]["warmup_steps"]:
            ema.update(model)

        step += 1
        for k, v in metrics.items():
            running.setdefault(k, []).append(v)
        running.setdefault("lr", []).append(optimizer.param_groups[0]["lr"])
        if grad_clip:
            running.setdefault("grad_norm", []).append(float(gnorm))

        if step % cfg["train"]["log_every"] == 0:
            avg = {k: sum(v) / len(v) for k, v in running.items()}
            elapsed = time.time() - t0
            line = (f"step {step:6d} | loss {avg['loss']:.4f} (latent {avg['latent_loss']:.4f}, "
                    f"eos {avg['eos_loss']:.4f}) | corr {avg['pred_target_corr']:.3f} | "
                    f"lr {avg['lr']:.2e} | {elapsed:.0f}s")
            print(line)
            history.append({"step": step, **{k: float(np.mean(v)) for k, v in running.items()}})
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **{k: float(np.mean(v)) for k, v in running.items()}}) + "\n")
            running = {}

        if eval_dl is not None and step % cfg["train"]["eval_every"] == 0:
            model.eval()
            cos_sum, n = 0.0, 0
            with torch.no_grad():
                for eb in eval_dl:
                    from training.flow_lm_training import teacher_forced_reconstruction

                    cos_sum += float(teacher_forced_reconstruction(model, eb))
                    n += 1
            print(f"  eval: mean cosine(pred, target) = {cos_sum / max(n, 1):.4f}")
            history.append({"step": step, "eval_cosine": cos_sum / max(n, 1)})
            model.train()

        if step % cfg["train"]["ckpt_every"] == 0 or step == total_steps:
            path = save_checkpoint(out_dir, step, model, ema, optimizer, cfg)
            print(f"  checkpoint → {path.name}")

    print(f"تمام شد: {step} گام در {time.time() - t0:.0f}s — خروجی در {out_dir}")


if __name__ == "__main__":
    main()
