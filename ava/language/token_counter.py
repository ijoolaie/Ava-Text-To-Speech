# -*- coding: utf-8 -*-
"""BPE token counting for model chunk sizing (moved from ava.text_norm).

Same logic as the browser implementation: greedy merges over the model's
vocab/merges tables, counting runs of unknown pieces as one token.
"""
from __future__ import annotations

from .chars import ZWNJ


class TokenCounter:
    def __init__(self, tokenizer_json: dict):
        model = tokenizer_json["model"]
        self.vocab: set[str] = set(model["vocab"])
        self.rank: dict[tuple[str, str], int] = {}
        for i, m in enumerate(model["merges"]):
            a, b = m if isinstance(m, list) else tuple(m.split(" "))
            self.rank[(a, b)] = i
        self.cache: dict[str, int] = {}

    def _word_tokens(self, word: str) -> int:
        if word in self.cache:
            return self.cache[word]
        parts = list(word)
        while True:
            best, best_rank = -1, 1 << 30
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]), None)
                if r is not None and r < best_rank:
                    best_rank, best = r, i
            if best < 0:
                break
            parts = parts[:best] + [parts[best] + parts[best + 1]] + parts[best + 2 :]
        n, prev_unknown = 0, False
        for p in parts:
            unknown = p not in self.vocab
            if not (unknown and prev_unknown):
                n += 1
            prev_unknown = unknown
        self.cache[word] = n
        return n

    def __call__(self, text: str) -> int:
        return sum(self._word_tokens("▁" + w) for w in text.replace(ZWNJ, " ").split(" ") if w)
