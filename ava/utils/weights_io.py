"""Safetensors I/O with multi-file (sharded) support — original code.

The repository ships the 107 MB model as two shards (< GitHub's 100 MB per-file
limit) plus a `model.safetensors.index.json` weight map, the same layout HF
repositories use. `load_file` transparently reads a plain file or a sharded one;
`save_file_sharded` writes the sharded layout.
"""

import json
import math
from pathlib import Path

import safetensors
import safetensors.torch
import torch

_SHARD_PATTERN = "{stem}-{index:05d}-of-{total:05d}.safetensors"


def _index_path(path: Path) -> Path:
    return path.parent / (path.name.removesuffix(".safetensors") + ".safetensors.index.json")


def load_file(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load a safetensors file, following a shard index when one sits next to it."""
    path = Path(path)
    index = _index_path(path)
    if index.exists():
        with open(index, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            state.update(safetensors.torch.load_file(path.parent / shard, device=device))
        return state
    return safetensors.torch.load_file(path, device=device)


def save_file_sharded(
    state: dict[str, torch.Tensor],
    dest: str | Path,
    max_shard_size_mb: float = 95.0,
    metadata: dict[str, str] | None = None,
) -> list[Path]:
    """Write `state` as one or more safetensors shards plus an index. Returns the files."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    n_shards = max(1, math.ceil(total_bytes / (max_shard_size_mb * 1024 * 1024)))
    if n_shards == 1:
        safetensors.torch.save_file(state, dest, metadata=metadata)
        # Remove a stale index if the model was previously sharded.
        index = _index_path(dest)
        if index.exists():
            for stale in dest.parent.glob(f"{dest.stem}-*.safetensors"):
                stale.unlink()
            index.unlink()
        return [dest]

    names = sorted(state.keys())
    target = total_bytes / n_shards
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in names:
        tensor = state[name]
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > target:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensor
        current_bytes += size
    if current:
        shards.append(current)

    stem = dest.name.removesuffix(".safetensors")
    weight_map: dict[str, str] = {}
    written: list[Path] = []
    for i, shard_state in enumerate(shards, start=1):
        shard_name = _SHARD_PATTERN.format(stem=stem, index=i, total=len(shards))
        shard_path = dest.parent / shard_name
        safetensors.torch.save_file(shard_state, shard_path, metadata=metadata)
        for name in shard_state:
            weight_map[name] = shard_name
        written.append(shard_path)

    index = {
        "metadata": {
            "total_size": total_bytes,
            **(metadata or {}),
        },
        "weight_map": weight_map,
    }
    with open(_index_path(dest), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    return written
