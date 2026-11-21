# -*- coding: utf-8 -*-
"""Pronunciation lexicon for the AVA Native language engine.

A lexicon maps written forms to their spoken form: Latin acronyms, brand and
proper names, and orthographic irregularities. Entries are applied on cleaned
text before numbers/dates are expanded, on a whole-token basis, plus optional
regex entries for patterned fixes.

Ships with a small starter lexicon (`data/pronunciation_fa.json`) and can be
extended at runtime or with a user-provided JSON file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_DATA_DIR = Path(__file__).parent / "data"
STARTER_LEXICON = _DATA_DIR / "pronunciation_fa.json"


class PronunciationLexicon:
    def __init__(self, words: dict[str, str] | None = None, regex: list[list[str]] | None = None):
        self.words: dict[str, str] = dict(words or {})
        self.regex: list[tuple[re.Pattern, str]] = [
            (re.compile(pat), repl) for pat, repl in (regex or [])
        ]
        # normalize lookup keys: latin keys are matched case-insensitively
        self._lower: dict[str, str] = {}
        for k, v in self.words.items():
            if re.fullmatch(r"[A-Za-z0-9\- ]+", k):
                self._lower.setdefault(k.lower(), v)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PronunciationLexicon":
        path = Path(path) if path else STARTER_LEXICON
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(words=data.get("words"), regex=data.get("regex"))

    def add(self, written: str, spoken: str) -> None:
        self.words[written] = spoken
        if re.fullmatch(r"[A-Za-z0-9\- ]+", written):
            self._lower.setdefault(written.lower(), spoken)

    def _lookup(self, token: str) -> str | None:
        if token in self.words:
            return self.words[token]
        return self._lower.get(token.lower())

    def apply(self, text: str) -> str:
        if not self.words and not self.regex:
            return text
        out: list[str] = []
        for token in text.split(" "):
            spoken = self._lookup(token)
            out.append(spoken if spoken else token)
        text = " ".join(out)
        for pattern, repl in self.regex:
            text = pattern.sub(repl, text)
        return text

    def to_json(self) -> dict:
        return {"words": self.words, "regex": [[p.pattern, r] for p, r in self.regex]}
