#!/usr/bin/env python3
"""Split the single model.safetensors into <100MB shards + index (GitHub-friendly),
or merge shards back into a single file (the browser demo needs one file).

Usage:
    python scripts/split_weights.py split [--src path] [--dest dir] [--max-mb 95]
    python scripts/split_weights.py merge [--dest path]   # reads weights/ shards
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ava.utils.weights_io import load_file, save_file_sharded  # noqa: E402


def do_split(src: Path, dest_dir: Path, max_mb: float) -> None:
    print(f"Loading {src} ...")
    state = load_file(src)
    total = sum(t.numel() * t.element_size() for t in state.values())
    print(f"{len(state)} tensors, {total / 1e6:.1f} MB")
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / "model.safetensors"
    written = save_file_sharded(state, target, max_shard_size_mb=max_mb)
    print("Wrote:")
    for p in written:
        print(f"  {p}  {p.stat().st_size / 1e6:.1f} MB")
    index = dest_dir / "model.safetensors.index.json"
    if index.exists():
        print(f"  {index}")
    # verify round-trip
    again = load_file(target)
    assert set(again) == set(state), "round-trip tensor names differ"
    for k in state:
        assert torch_equal(state[k], again[k]), f"round-trip mismatch on {k}"
    print("Round-trip verification passed.")


def torch_equal(a, b) -> bool:
    import torch

    return a.shape == b.shape and bool(torch.equal(a, b))


def do_merge(dest: Path) -> None:
    src = REPO_ROOT / "weights" / "model.safetensors"
    state = load_file(src)
    import safetensors.torch

    dest.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(state, dest)
    print(f"Merged {len(state)} tensors -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Split/merge Ava weights")
    ap.add_argument("mode", choices=["split", "merge"])
    ap.add_argument("--src", default=str(REPO_ROOT / "weights" / "model.safetensors"))
    ap.add_argument("--dest", default=None, help="split: target dir; merge: output file")
    ap.add_argument("--max-mb", type=float, default=95.0)
    args = ap.parse_args()
    if args.mode == "split":
        dest_dir = Path(args.dest) if args.dest else REPO_ROOT / "weights"
        do_split(Path(args.src), dest_dir, args.max_mb)
    else:
        dest = Path(args.dest) if args.dest else REPO_ROOT / "web" / "model.safetensors"
        do_merge(dest)


if __name__ == "__main__":
    main()
