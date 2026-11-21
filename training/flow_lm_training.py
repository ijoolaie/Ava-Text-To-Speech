"""Training forward/loss for the FlowLM "drifting" head — original code.

Objective (one-step drifting head, matching the released checkpoint's
``flow.type: drifting``):

- teacher forcing over continuous codec latents; position 0 is the BOS slot,
  encoded as NaN in the input and replaced by the learned ``bos_emb``;
- the transformer hidden state is projected to z (``z_proj`` → 64 dims);
- from Gaussian noise x_0 ~ N(0, noise_std²) the flow head directly predicts the
  next latent:  L_latent = ‖ flow_net(z_t, x_0) − x_{t+1} ‖²  (masked mean);
- ``out_eos`` is a binary head on z:  L_eos = BCE(logit_t, 1[t is last]).

Latent targets are normalized with the model's current ``emb_mean``/``emb_std``
buffers, which follow an EMA update each step (Kyutai's stats_ema recipe).
"""

from __future__ import annotations

import torch


def normalize_targets(targets: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (targets - mean) / std


def update_stats_ema(
    model,
    batch_targets_raw: torch.Tensor,
    mask: torch.Tensor,
    decay: float,
) -> None:
    """EMA update of emb_mean/emb_std from the raw (unnormalized) batch latents."""
    with torch.no_grad():
        valid = batch_targets_raw[mask]  # [N, 32]
        if valid.numel() < 2:
            return
        batch_mean = valid.mean(dim=0)
        batch_std = valid.std(dim=0).clamp_min(1e-4)
        model.emb_mean.mul_(decay).add_(batch_mean, alpha=1 - decay)
        model.emb_std.mul_(decay).add_(batch_std, alpha=1 - decay)


def flow_lm_loss(
    model,
    batch: dict,
    noise_std: float = 1.0,
    eos_loss_weight: float = 1.0,
    latent_loss_weight: float = 1.0,
    voice_cond_normalized: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute the total training loss. `model` is a FlowLMModel (train mode).

    When the model carries a ``speaker_proj_weight``, each clip conditions on its
    own voice: the codec latents are projected like the inference path's
    ``_encode_audio`` and prepended to the text embeddings. Pass
    ``voice_cond_normalized=True`` when the checkpoint/inference config does.
    """
    tokens = batch["tokens"]
    inputs = batch["inputs"].to(model.bos_emb.device)
    targets_raw = batch["targets"].to(model.bos_emb.device)
    eos_labels = batch["eos_labels"].to(model.bos_emb.device)
    mask = batch["mask"].to(model.bos_emb.device)

    # NaN sentinel -> learned BOS, identical to the inference path.
    sequence = torch.where(torch.isnan(inputs), model.bos_emb, inputs)
    # Teacher-forced targets in normalized space.
    targets = normalize_targets(targets_raw, model.emb_mean, model.emb_std)

    text_embeddings = model.conditioner(tokens)
    if hasattr(model, "speaker_proj_weight") and model.speaker_proj_weight is not None:
        # Voice prefix: the projection consumes codec latents (normalized only when
        # the config says so), matching TTSModel._encode_audio.
        cond_latents = targets_raw
        if voice_cond_normalized:
            cond_latents = normalize_targets(targets_raw, model.emb_mean, model.emb_std)
        audio_conditioning = torch.nn.functional.linear(cond_latents, model.speaker_proj_weight)
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

    input_ = model.input_linear(sequence)
    hidden = model.backbone(input_, text_embeddings, sequence, model_state=None)

    # Drifting head: one step from noise to the (normalized) next latent.
    noise = torch.randn(
        hidden.shape[0], hidden.shape[1], model.ldim,
        device=hidden.device, dtype=torch.float32,
    ) * noise_std
    pred = model.flow_net(hidden, noise)

    sq_err = ((pred - targets) ** 2).mean(dim=-1)
    n_valid = mask.sum().clamp_min(1)
    latent_loss = (sq_err * mask).sum() / n_valid

    eos_logits = model.out_eos(hidden).squeeze(-1)
    eos_per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        eos_logits, eos_labels, reduction="none"
    )
    eos_loss = (eos_per_item * mask).sum() / n_valid

    loss = latent_loss_weight * latent_loss + eos_loss_weight * eos_loss
    with torch.no_grad():
        eos_probs = torch.sigmoid(eos_logits)
        metrics = {
            "loss": loss.item(),
            "latent_loss": latent_loss.item(),
            "eos_loss": eos_loss.item(),
            # fraction of last positions whose EOS logit crosses the default threshold
            "eos_recall": (
                ((eos_logits > -2.0).float() * eos_labels * mask).sum()
                / eos_labels[mask].sum().clamp_min(1)
            ).item(),
            "false_eos_rate": (
                ((eos_logits > -2.0).float() * (1 - eos_labels) * mask).sum()
                / ((1 - eos_labels) * mask).sum().clamp_min(1)
            ).item(),
            "pred_target_corr": (
                torch.corrcoef(torch.stack([pred[mask].flatten(), targets[mask].flatten()]))[0, 1].item()
                if mask.any()
                else 0.0
            ),
            "n_frames": int(mask.sum().item()),
        }
    return loss, metrics


@torch.no_grad()
def teacher_forced_reconstruction(
    model,
    batch: dict,
    noise_std: float = 0.3,
    voice_cond_normalized: bool = False,
) -> torch.Tensor:
    """Mean cosine distance between predicted and target latents (evaluation)."""
    tokens = batch["tokens"]
    inputs = batch["inputs"].to(model.bos_emb.device)
    targets_raw = batch["targets"].to(model.bos_emb.device)
    mask = batch["mask"].to(model.bos_emb.device)

    sequence = torch.where(torch.isnan(inputs), model.bos_emb, inputs)
    targets = normalize_targets(targets_raw, model.emb_mean, model.emb_std)
    text_embeddings = model.conditioner(tokens)
    if hasattr(model, "speaker_proj_weight") and model.speaker_proj_weight is not None:
        cond_latents = targets_raw
        if voice_cond_normalized:
            cond_latents = normalize_targets(targets_raw, model.emb_mean, model.emb_std)
        audio_conditioning = torch.nn.functional.linear(cond_latents, model.speaker_proj_weight)
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)
    input_ = model.input_linear(sequence)
    hidden = model.backbone(input_, text_embeddings, sequence, model_state=None)
    noise = torch.randn(
        hidden.shape[0], hidden.shape[1], model.ldim,
        device=hidden.device, dtype=torch.float32,
    ) * noise_std
    pred = model.flow_net(hidden, noise)
    cos = torch.nn.functional.cosine_similarity(pred, targets, dim=-1)
    return cos[mask].mean()
