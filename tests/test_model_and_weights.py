"""Architecture tests: parameter counts and strict compatibility with the
shipped Pocket-TTS checkpoint."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ava.models.tts_model import TTSModel  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return TTSModel.load_model(REPO_ROOT / "config" / "fa.yaml")


def test_param_counts_match_official_numbers(model):
    """Official card: 18.2M trainable (15.6M flow LM + 2.6M decoder),
    27.9M total (the extra 9.7M is the Mimi encoder)."""
    state = model.state_dict()
    flow = sum(t.numel() for n, t in state.items() if n.startswith("flow_lm."))
    enc = sum(
        t.numel() for n, t in state.items()
        if n.startswith("mimi.encoder.") or n.startswith("mimi.encoder_transformer.")
        or n.startswith("mimi.downsample.")
    )
    dec = sum(
        t.numel() for n, t in state.items()
        if n.startswith("mimi.decoder.") or n.startswith("mimi.decoder_transformer.")
        or n.startswith("mimi.quantizer.") or n.startswith("mimi.upsample.")
    )
    total = sum(t.numel() for t in state.values())
    assert abs(flow / 1e6 - 15.6) < 0.15, f"flow LM {flow / 1e6:.2f}M"
    assert abs(dec / 1e6 - 2.6) < 0.15, f"decoder {dec / 1e6:.2f}M"
    assert abs(enc / 1e6 - 9.7) < 0.25, f"encoder {enc / 1e6:.2f}M"
    assert abs(total / 1e6 - 27.9) < 0.3, f"total {total / 1e6:.2f}M"


def test_checkpoint_tensor_shapes(model):
    """Every checkpoint tensor must map 1:1 onto the built model (strict load ran
    in the fixture; here we double-check a few architecture-defining shapes)."""
    sd = model.state_dict()
    assert sd["flow_lm.transformer.layers.7.self_attn.in_proj.weight"].shape == (1536, 512)
    assert sd["flow_lm.conditioner.embed.weight"].shape == (4006, 512)
    assert sd["flow_lm.z_proj.weight"].shape == (64, 512)
    assert sd["flow_lm.speaker_proj_weight"].shape == (512, 32)
    assert sd["mimi.downsample.conv.conv.weight"].shape == (32, 512, 32)
    assert sd["mimi.upsample.convtr.convtr.weight"].shape == (128, 1, 32)
    assert sd["mimi.decoder_transformer.transformer.layers.11.self_attn.in_proj.weight"].shape == (384, 128)
    assert len(model.flow_lm.transformer.layers) == 8
    assert len(model.mimi.decoder_transformer.transformer.layers) == 12
    assert len(model.mimi.encoder_transformer.transformer.layers) == 2


def test_sharded_index_consistency():
    from ava.utils.weights_io import load_file

    path = REPO_ROOT / "weights" / "model.safetensors"
    index = REPO_ROOT / "weights" / "model.safetensors.index.json"
    assert index.exists(), "sharded index missing"
    import json

    wm = json.loads(index.read_text())["weight_map"]
    state = load_file(path)
    assert len(state) == 302
    assert set(state) == set(wm)
    shard_sizes = {
        (REPO_ROOT / "weights" / s).stat().st_size for s in set(wm.values())
    }
    assert max(shard_sizes) < 100_000_000, "a shard exceeds GitHub's 100 MB limit"


def test_normalizer_basic():
    from ava.text_norm import normalize, split_sentences

    text = normalize("۲۳ کتاب دارم.")
    assert "بیست و سه" in text
    assert "۲" not in text
    text2 = normalize("سلام! % این یک تست است.")
    assert "%" not in text2
    pieces = split_sentences("جمله اول. جمله دوم؟ جمله سوم!")
    assert len(pieces) == 3
