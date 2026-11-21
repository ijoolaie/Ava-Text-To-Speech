"""Training-pipeline smoke test: a tiny flow LM must learn on real codec latents.

Uses the smoke architecture (training/configs/smoke_arch.yaml) on synthetic
latent targets (random walk shaped like speech latents) to keep the test fast
and hermetic — the real end-to-end run is exercised by tests/test_train_real.py
(only when the latent cache exists) and by `make smoke-train`.
"""

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ava.models.flow_lm import FlowLMModel  # noqa: E402
from ava.utils.config import load_config  # noqa: E402
from training.flow_lm_training import flow_lm_loss  # noqa: E402


def _build_tiny_model(tmp_path: Path) -> FlowLMModel:
    arch = load_config(REPO_ROOT / "training" / "configs" / "smoke_arch.yaml")
    arch.flow_lm.lookup_table.tokenizer_path = str(
        REPO_ROOT / "weights" / "tokenizer.json"
    )
    return FlowLMModel.from_pydantic_config(
        arch.flow_lm, latent_dim=arch.mimi.quantizer.dimension,
        insert_bos_before_voice=arch.flow_lm.insert_bos_before_voice,
    )


def _fake_batch(batch_size=2, frames=40, ldim=32):
    """Random-walk latents (temporal correlation, like speech latents)."""
    targets = torch.zeros(batch_size, frames, ldim)
    for b in range(batch_size):
        steps = np.cumsum(np.random.randn(frames, ldim) * 0.3, axis=0)
        targets[b] = torch.from_numpy(steps.astype(np.float32))
    inputs = torch.full((batch_size, frames, ldim), float("nan"))
    inputs[:, 1:] = targets[:, :-1]
    eos = torch.zeros(batch_size, frames)
    eos[:, -1] = 1.0
    mask = torch.ones(batch_size, frames, dtype=torch.bool)
    return {
        "tokens": torch.randint(0, 4000, (batch_size, 8)),
        "inputs": inputs,
        "targets": targets,
        "eos_labels": eos,
        "mask": mask,
    }


def test_drifting_loss_decreases(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_tiny_model(tmp_path)
    model.train()

    batch = _fake_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first_loss = None
    last_loss = None
    for step in range(60):
        loss, metrics = flow_lm_loss(model, batch, noise_std=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = metrics["latent_loss"]
        last_loss = metrics["latent_loss"]
    assert last_loss < first_loss * 0.5, (
        f"latent loss did not decrease enough: {first_loss:.4f} -> {last_loss:.4f}"
    )
    assert abs(metrics["pred_target_corr"]) < 1.0  # sanity


def test_stats_ema_updates_buffers():
    torch.manual_seed(0)
    np.random.seed(0)
    model = _build_tiny_model(tmp_path=None)
    model.train()
    from training.flow_lm_training import update_stats_ema

    batch = _fake_batch(frames=50)
    mean0 = model.emb_mean.clone()
    update_stats_ema(model, batch["targets"], batch["mask"], decay=0.9)
    assert not torch.equal(mean0, model.emb_mean)
    # targets are non-zero random walks -> the mean should have moved upward
    assert model.emb_mean.abs().sum() > 0
