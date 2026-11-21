# -*- coding: utf-8 -*-
"""Character tokenizer over the Persian language-engine alphabet (AVA Native).

The vocabulary is deliberately tiny and dependency-free: Persian letters,
half-space (ZWNJ), space and the kept punctuation, plus specials. Normalized
text from `ava.language` encodes one character per token.
"""
from __future__ import annotations

import json

from ava.language.chars import KEPT_PUNCT, PERSIAN_LETTERS, ZWNJ

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]


class CharTokenizer:
    def __init__(self):
        self.vocab: list[str] = SPECIALS + list(PERSIAN_LETTERS + " " + ZWNJ + KEPT_PUNCT)
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.pad_id = self.stoi[PAD]
        self.unk_id = self.stoi[UNK]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.bos_id] + [self.stoi.get(c, self.unk_id) for c in text] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i in (self.bos_id, self.eos_id, self.pad_id):
                continue
            c = self.vocab[i] if 0 <= i < len(self.vocab) else UNK
            out.append(c)
        return "".join(out)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "ava-n1-char", "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        tok = cls.__new__(cls)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok.vocab = data["vocab"]
        tok.stoi = {c: i for i, c in enumerate(tok.vocab)}
        tok.pad_id, tok.unk_id = tok.stoi[PAD], tok.stoi[UNK]
        tok.bos_id, tok.eos_id = tok.stoi[BOS], tok.stoi[EOS]
        return tok
