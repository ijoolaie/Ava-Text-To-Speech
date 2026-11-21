"""Dataset over the prepared latent cache: (text tokens, latents, EOS labels)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

LUT_PAD = 4005  # the n_bins id: the LUT embed has n_bins + 1 rows; the last one pads


@dataclass
class TrainItem:
    stem: str
    frames: int
    text: str
    source: str


def load_items(cache_dir: Path) -> tuple[list[TrainItem], dict]:
    items = [TrainItem(**d) for d in json.loads((cache_dir / "items.json").read_text("utf-8"))]
    stats = json.loads((cache_dir / "stats.json").read_text("utf-8"))
    return items, stats


class LatentDataset(torch.utils.data.Dataset):
    """Each sample: full clip latents (raw, not normalized) + normalized text.

    Normalization happens in the collate/training step with the *current* EMA
    stats, so a moving mean/std stays consistent inside the loss.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        tokenizer,
        max_frames: int = 250,
        max_tokens: int = 64,
        min_frames: int = 10,
        holdout_ratio: float = 0.0,
        holdout: bool = False,
        seed: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_tokens = max_tokens
        self.min_frames = min_frames

        items, self.stats = load_items(self.cache_dir)
        rng = random.Random(seed)
        rng.shuffle(items)
        if holdout_ratio > 0 and len(items) > 20:
            n_holdout = max(1, int(len(items) * holdout_ratio))
            items = items[:n_holdout] if holdout else items[n_holdout:]

        self.items: list[TrainItem] = []
        skipped_long = 0
        for it in items:
            if not (self.min_frames <= it.frames <= self.max_frames):
                skipped_long += 1
                continue
            if len(self.tokenizer.encode(it.text)) > self.max_tokens:
                skipped_long += 1
                continue
            self.items.append(it)
        if skipped_long:
            print(f"  dataset({'holdout' if holdout else 'train'}): skipped {skipped_long} "
                  f"clips (length/token cap), kept {len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        data = np.load(self.cache_dir / f"{it.stem}.npz")
        latents = data["latents"].astype(np.float32)  # [T, 32] raw
        tokens = self.tokenizer.encode(it.text)
        return {"latents": latents, "tokens": tokens, "stem": it.stem}


class Collator:
    """Pads a batch and builds teacher-forced inputs/targets.

    Positions: inputs = [BOS, z_1 … z_{T-1}], targets = [z_1 … z_T]; the EOS
    label is 1 exactly on the last position (nothing follows the final latent).
    BOS is represented as NaN in the padded input, mirroring inference.
    """

    def __init__(self, ldim: int = 32, pad_id: int = LUT_PAD):
        self.ldim = ldim
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        lens = [b["latents"].shape[0] for b in batch]
        T = max(lens)
        tok_lens = [len(b["tokens"]) for b in batch]
        L = max(tok_lens)

        # Text tokens, padded with the extra (n_bins) id the LUT embed reserves.
        tokens = torch.full((B, L), self.pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            tokens[i, : tok_lens[i]] = torch.tensor(b["tokens"], dtype=torch.long)
        text_mask = torch.zeros(B, L, dtype=torch.bool)
        for i in range(B):
            text_mask[i, : tok_lens[i]] = True

        latents = torch.zeros(B, T, self.ldim)
        targets = torch.zeros(B, T, self.ldim)
        eos_labels = torch.zeros(B, T)
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, b in enumerate(batch):
            t = lens[i]
            z = torch.from_numpy(b["latents"])
            latents[i, :t] = z[:t]
            targets[i, :t] = z
            mask[i, :t] = True
            eos_labels[i, t - 1] = 1.0
        # Teacher-forced inputs: BOS (NaN sentinel) then z_1..z_{T-1}.
        inputs = torch.full((B, T, self.ldim), float("nan"))
        for i, b in enumerate(batch):
            t = lens[i]
            inputs[i, 1:t] = latents[i, : t - 1]

        return {
            "tokens": tokens,
            "text_mask": text_mask,
            "inputs": inputs,        # [B, T, 32], NaN = BOS slot
            "targets": targets,      # [B, T, 32] raw latents
            "eos_labels": eos_labels,
            "mask": mask,
            "frames": torch.tensor(lens),
        }
