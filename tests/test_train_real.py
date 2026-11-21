"""End-to-end training test on the real latent cache (skipped if absent).

Run `make smoke-data` first (it generates clips with the released model, then
runs training/data/prepare_data.py). This test then verifies that the tiny flow
LM actually learns on real Persian speech latents and that the exported bundle
loads strictly.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE = REPO_ROOT / "data_cache" / "smoke"


def _cache_ready() -> bool:
    return (CACHE / "items.json").exists() and (CACHE / "stats.json").exists()


@pytest.mark.skipif(not _cache_ready(), reason="smoke latent cache missing (run `make smoke-data`)")
def test_train_on_real_latents(tmp_path):
    import torch

    from training.data.dataset import Collator, LatentDataset
    from training.flow_lm_training import flow_lm_loss
    from training.train_flow_lm import EMA, build_model, param_groups

    model, arch = build_model(
        REPO_ROOT / "training" / "configs" / "smoke_arch.yaml", None, True
    )
    tokenizer = model.conditioner.tokenizer
    ds = LatentDataset(
        CACHE, tokenizer, max_frames=250, max_tokens=64, min_frames=8,
        holdout_ratio=0.2, holdout=False,
    )
    assert len(ds) >= 3, "smoke cache too small"
    dl = torch.utils.data.DataLoader(
        ds, batch_size=min(4, len(ds)), shuffle=True,
        collate_fn=Collator(ldim=arch.mimi.quantizer.dimension),
    )
    optimizer = torch.optim.AdamW(param_groups(model, 0.01), lr=2e-3, betas=(0.9, 0.95))
    ema = EMA(model, 0.99)

    losses = []
    for epoch in range(3):
        for batch in dl:
            loss, metrics = flow_lm_loss(model, batch, noise_std=1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            losses.append(metrics["latent_loss"])

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    # Checkpoint in the Pocket-TTS training format + EMA overlay logic
    payload = {
        "model": {f"flow_lm.{k}": v for k, v in model.state_dict().items()},
        "ema": ema.state_dict(),
        "step": 1,
    }
    ckpt = tmp_path / "checkpoint.pt"
    torch.save(payload, ckpt)
    from ava.utils.weights_loading import get_training_checkpoint_state_dicts

    flow_state, _ = get_training_checkpoint_state_dicts(ckpt, use_ema=True)
    assert flow_state, "EMA overlay produced an empty state"


@pytest.mark.skipif(not _cache_ready(), reason="smoke latent cache missing")
def test_export_roundtrip(tmp_path):
    import torch

    from ava.models.tts_model import TTSModel
    from ava.utils.config import load_config
    from ava.utils.weights_io import load_file, save_file_sharded
    from ava.utils.weights_loading import get_training_checkpoint_state_dicts
    from training.train_flow_lm import build_model

    model, _ = build_model(
        REPO_ROOT / "training" / "configs" / "smoke_arch.yaml", None, True
    )
    payload = {"model": {f"flow_lm.{k}": v for k, v in model.state_dict().items()}, "step": 1}
    ckpt = tmp_path / "checkpoint.pt"
    torch.save(payload, ckpt)
    flow_state, _ = get_training_checkpoint_state_dicts(ckpt, use_ema=True)

    arch = load_config(REPO_ROOT / "training" / "configs" / "smoke_arch.yaml")
    base = load_file(REPO_ROOT / "weights" / "model.safetensors")
    bundle = {f"flow_lm.{k}": v.float() for k, v in flow_state.items()}
    bundle.update({k: v.float() for k, v in base.items() if k.startswith("mimi.")})
    out = tmp_path / "model.safetensors"
    save_file_sharded(bundle, out, max_shard_size_mb=1e9)

    arch_cfg = load_config(REPO_ROOT / "training" / "configs" / "smoke_arch.yaml")
    arch_cfg.weights_path = str(out)
    loaded = TTSModel._from_pydantic_config_with_weights(
        arch_cfg, temp=0.3, sampler_decode_steps=1, noise_clamp=None,
        eos_threshold=-2.0, origin=REPO_ROOT / "training" / "configs" / "smoke_arch.yaml",
    )
    n = sum(t.numel() for t in loaded.state_dict().values())
    n_mimi = sum(t.numel() for k, t in loaded.state_dict().items() if k.startswith("mimi."))
    # tiny flow LM (~0.37M) + full frozen codec (12.37M)
    assert abs(n_mimi / 1e6 - 12.37) < 0.15
    assert 10e6 < n < 15e6
