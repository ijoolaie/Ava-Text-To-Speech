"""Tokenizer, text-chunking and voice-file format tests."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ava.modules.text_conditioner import build_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer(4005, str(REPO_ROOT / "weights" / "tokenizer.json"), "tokenizers")


def test_vocab_size_and_roundtrip(tokenizer):
    assert tokenizer.tokenizer.get_vocab_size() == 4005
    text = "سلام دنیا. این یک آزمون است."
    ids = tokenizer.encode(text)
    assert ids and all(0 <= i < 4005 for i in ids)
    assert tokenizer.decode(ids) == text


def test_chunking_respects_max_tokens(tokenizer):
    from ava.models.text_chunking import split_into_best_sentences

    long_text = ("جملهٔ آزمایشی شمارهٔ یک است. " * 30).strip()
    chunks = split_into_best_sentences(
        tokenizer, long_text, max_tokens=50,
        pad_with_spaces_for_short_inputs=False, remove_semicolons=False,
    )
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= 50, chunk[:40]


def test_prepare_text_prompt_adds_punctuation():
    from ava.models.text_chunking import prepare_text_prompt

    text, frames = prepare_text_prompt(
        "سلام", pad_with_spaces_for_short_inputs=False, remove_semicolons=False,
        append_terminal_punctuation=True, capitalize_first_letter=False,
    )
    assert text.endswith(".")
    assert frames == 3  # short text => longer tail after EOS


def test_voice_file_format():
    import torch
    from ava.utils.weights_io import load_file

    path = REPO_ROOT / "weights" / "voices" / "nava.safetensors"
    state = load_file(path)
    assert set(state) == {"audio_prompt"}
    tensor = state["audio_prompt"]
    assert tensor.ndim == 2 and tensor.shape[-1] == 512, tensor.shape
    assert tensor.dtype == torch.float32
