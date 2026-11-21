# Adapted from Kyutai pocket-tts v4.0.1 (MIT). Local tokenizer files only.
"""Text tokenizer backends and the lookup-table text conditioner."""

import logging
from collections.abc import Callable

import sentencepiece
import tokenizers
import torch
from torch import nn

from ava.utils.utils import resolve_path

logger = logging.getLogger(__name__)


class SentencePieceTokenizer:
    """SentencePiece model file (``tokenizer.model``).

    Args:
        nbins (int): should be equal to the vocab size of the sentencepiece model.
        tokenizer_path (str): path to the sentencepiece tokenizer model.
    """

    def __init__(self, nbins: int, tokenizer_path: str):
        logger.info("Loading sentencepiece tokenizer from %s", tokenizer_path)
        local_path = resolve_path(tokenizer_path)
        self.sp = sentencepiece.SentencePieceProcessor(str(local_path))
        assert nbins == self.sp.vocab_size(), (
            f"sentencepiece tokenizer has vocab size={self.sp.vocab_size()} but nbins={nbins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def serialize(self) -> tuple[str, bytes]:
        return "sentencepiece", self.sp.serialized_model_proto()

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text))[None, :]


class JsonTokenizer:
    """A ``tokenizer.json`` (HuggingFace tokenizers) holding the same vocabulary.

    Converted from the SentencePiece model, so it produces identical token ids;
    it loads without the sentencepiece runtime and is the format the released
    config points at. ``.model`` files stay supported for models trained elsewhere.
    """

    def __init__(self, nbins: int, tokenizer_path: str):
        logger.info("Loading tokenizers json from %s", tokenizer_path)
        local_path = resolve_path(tokenizer_path)
        self.tokenizer = tokenizers.Tokenizer.from_file(str(local_path))
        vocab_size = self.tokenizer.get_vocab_size()
        assert nbins == vocab_size, (
            f"tokenizer has vocab size={vocab_size} but nbins={nbins} was specified"
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def serialize(self) -> tuple[str, bytes]:
        return "tokenizers", self.tokenizer.to_str().encode()

    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text))[None, :]


Tokenizer = SentencePieceTokenizer | JsonTokenizer


def encoder_from_serialized(kind: str, payload: bytes) -> Callable[[str], list[int]]:
    """Rebuild just the text -> ids function in a worker process."""
    if kind == "tokenizers":
        tokenizer = tokenizers.Tokenizer.from_str(payload.decode())
        return lambda text: tokenizer.encode(text).ids
    sp = sentencepiece.SentencePieceProcessor()
    sp.load_from_serialized_proto(payload)
    return lambda text: sp.encode(text, out_type=int)


def build_tokenizer(nbins: int, tokenizer_path: str, kind: str = "sentencepiece") -> Tokenizer:
    """Pick the backend from the config's `tokenizer` field, else from the suffix."""
    if kind in ("tokenizers", "json") or str(tokenizer_path).split("@")[0].endswith(".json"):
        return JsonTokenizer(nbins, tokenizer_path)
    return SentencePieceTokenizer(nbins, tokenizer_path)


class LUTConditioner(nn.Module):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer backend ("sentencepiece" or "tokenizers").
        rank (int or None): If set, the table is low-rank: `rank`-wide embeddings followed by
            a bias-free projection to `dim`.
    """

    def __init__(
        self,
        n_bins: int,
        tokenizer_path: str,
        dim: int,
        output_dim: int,
        tokenizer: str = "sentencepiece",
        rank: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.tokenizer = build_tokenizer(n_bins, tokenizer_path, tokenizer)
        self.embed = nn.Embedding(n_bins + 1, rank or self.dim)  # n_bins + 1 for padding.
        self.proj = None if rank is None else nn.Linear(rank, self.dim, bias=False)

    def prepare(self, x: str) -> torch.Tensor:
        return self.tokenizer(x).to(self.embed.weight.device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        return x if self.proj is None else self.proj(x)
