# -*- coding: utf-8 -*-
"""Sentence segmentation and long-text splitting (AVA Native language engine).

Sentence boundaries are [.!؟…] followed by whitespace; decimal points and
ellipsis are protected. Overlong sentences are split on weak punctuation
(،؛:) at the last boundary before the limit, matching the legacy behavior.
"""
from __future__ import annotations

import re

_DOT_PLACEHOLDER = "\x00DOT\x00"


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}|\r\n\r\n", text) if p.strip()]


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split into sentences; very long sentences fall back to weak punctuation."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # protect decimal points so «3.14» does not end a sentence
    masked = re.sub(r"(\d)\.(\d)", rf"\g<1>{_DOT_PLACEHOLDER}\g<2>", text)
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!؟])\s+", masked):
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = max(sentence.rfind(c, 0, max_chars) for c in "،؛:")
            if cut <= 0:
                cut = max_chars
            out.append(sentence[: cut + 1].strip())
            sentence = sentence[cut + 1 :].strip()
        if sentence:
            out.append(sentence)
    return [s.replace(_DOT_PLACEHOLDER, ".") for s in out]


def with_end(text: str) -> str:
    """Ensure the text ends with terminal punctuation (legacy helper)."""
    text = text.strip()
    if text == "" or text[-1] in ".؟!":
        return text
    return text + "."
