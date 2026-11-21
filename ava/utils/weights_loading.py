# Adapted from Kyutai pocket-tts v4.0.1 (MIT): name mapping between training
# checkpoints and this package's layout. Local files only; sharded bundles supported.
"""Weight name mapping between training checkpoints / legacy layouts and ava."""

from pathlib import Path

import torch

from ava.utils.weights_io import load_file


def get_flow_lm_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """flow_lm.* tensors from a bundle, renaming legacy training-layout keys when present."""
    state = load_file(path)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if key.startswith("flow.w_s_t.") or key.endswith(".learnt_padding") or key == "num_ema_updates":
            continue
        new_name = key
        if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
            new_name = "flow_lm.conditioner.embed.weight"
        if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
            new_name = "flow_lm.speaker_proj_weight"
        if key == "fuser.padding_value":
            new_name = "flow_lm.bos_before_voice"
        new_name = new_name.replace(".self_attn.in_proj_weight", ".self_attn.in_proj.weight")
        out[new_name] = tensor
    return {k: v for k, v in out.items() if k.startswith("flow_lm.")}


def get_mimi_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """mimi.* tensors from a bundle (already in this package's layout) or a legacy codec file."""
    state = load_file(path)
    if any(key.startswith("mimi.") for key in state):
        return {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
    out: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if (
            key.startswith("model.quantizer.vq.")
            or key == "model.quantizer.logvar_proj.weight"
            or "_codebook" in key
            or key.endswith(".weight_v")
            or key == "quantizer.logvar_proj.weight"
        ):
            continue
        if key.endswith(".weight_g"):
            key_g = key
            new_key = key.removesuffix("_g")
            key_v = new_key + "_v"
            weight_v = state[key_v]
            weight_g = state[key_g]
            new_key = new_key.replace(".conv.conv.", ".conv.").replace(".convtr.convtr.", ".convtr.")
            out[new_key] = torch._weight_norm(weight_v, weight_g, dim=0)
            continue
        if key in [
            "wavlm_emb_downsample.conv.conv.weight",
            "wavlm_input_resample.kernel",
            "wavlm_proj.weight",
            "quantizer.logvar_param",
        ] or "wavlm_emb_downsample" in key:
            continue
        out[
            key.removeprefix("model.")
            .replace(".conv.conv.", ".conv.")
            .replace(".convtr.convtr.", ".convtr.")
            .replace("in_proj_weight", "in_proj.weight")
        ] = tensor
    return out


def get_training_checkpoint_state_dicts(
    path: str | Path, use_ema: bool = True
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """(flow_lm, mimi) state dicts from a training checkpoint (.pt).

    Training checkpoints hold the trainable model under "model" and, when the run
    tracked one, its EMA shadow under "ema"; both are keyed with the "flow_lm." /
    "mimi." prefixes of the training wrapper. The shadow covers trained parameters
    only, so it is overlaid on "model" rather than used on its own.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = payload.get("model")
    if state is None:
        raise ValueError(f"{path} holds no 'model' weights")
    if use_ema and payload.get("ema"):
        state = {**state, **payload["ema"]}
    flow_lm = {k.removeprefix("flow_lm."): v for k, v in state.items() if k.startswith("flow_lm.")}
    mimi = {k.removeprefix("mimi."): v for k, v in state.items() if k.startswith("mimi.")}
    if not flow_lm:
        raise ValueError(f"no flow_lm.* weights in {path}")
    return flow_lm, mimi
