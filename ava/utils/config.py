# Adapted from Kyutai pocket-tts v4.0.1 (MIT) — same schema, local paths only.
# New in Ava: `languages` and `voices_dir` fields; no hf:// downloads.
"""Configuration models for loading YAML config files (pydantic, strict)."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from ava.utils.utils import resolve_path


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# Flow configuration
class FlowConfig(StrictModel):
    dim: int
    depth: int
    # "lsd" (2 time conditions, 1-step decode), "flow_matching" (1 time condition, Euler
    # integration; needs >= 16 decode steps), "drifting" (no time condition, the head maps
    # noise to a sample in one step) or "pointwise" (no time condition, one step).
    type: str = "lsd"


# Transformer configuration for FlowLM
class FlowLMTransformerConfig(StrictModel):
    hidden_scale: int
    max_period: float
    d_model: int
    num_heads: int
    num_layers: int
    attn_dim: int | None = None
    num_kv_heads: int | None = None
    dim_feedforward: int | None = None


class LookupTable(StrictModel):
    dim: int
    n_bins: int
    tokenizer: str
    tokenizer_path: str
    rank: int | None = None


# Root configuration for the flow language model
class FlowLMConfig(StrictModel):
    """Root configuration model for the flow LM section of a YAML config file."""

    dtype: str

    flow: FlowConfig
    transformer: FlowLMTransformerConfig
    z_dim: int | None = None

    # conditioning
    lookup_table: LookupTable
    weights_path: str | None = None
    insert_bos_before_voice: bool = False


# SEANet configuration
class SEANetConfig(StrictModel):
    dimension: int
    channels: int
    n_filters: int
    n_residual_layers: int
    ratios: list[int]
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    dilation_base: int
    pad_mode: str
    compress: int


# Transformer configuration for Mimi
class MimiTransformerConfig(StrictModel):
    d_model: int
    input_dimension: int
    output_dimensions: tuple[int, ...]
    num_heads: int
    num_layers: int
    layer_scale: float
    context: int
    max_period: float = 10000.0
    dim_feedforward: int


# Quantizer configuration
class QuantizerConfig(StrictModel):
    dimension: int
    output_dimension: int


# Root configuration for the Mimi codec
class MimiConfig(StrictModel):
    """Root configuration model for Mimi YAML config files."""

    dtype: str

    sample_rate: int
    channels: int
    frame_rate: float

    seanet: SEANetConfig
    transformer: MimiTransformerConfig

    quantizer: QuantizerConfig
    weights_path: str | None = None
    inner_dim: int | None = None
    outer_dim: int | None = None
    # Decoder-side SEANet and transformer, when they differ from the encoder's. The decoder
    # side also includes quantizer.output_dimension and outer_dim.
    decoder_seanet: SEANetConfig | None = None
    decoder_transformer: MimiTransformerConfig | None = None


class Config(StrictModel):
    flow_lm: FlowLMConfig
    mimi: MimiConfig
    weights_path: str | None = None
    pad_with_spaces_for_short_inputs: bool = False
    remove_semicolons: bool = False
    append_terminal_punctuation: bool = True
    # Upper-casing the first letter is an orthographic convention of Latin-script languages.
    # Persian has no case, so the shipped fa.yaml sets this to False.
    capitalize_first_letter: bool = True
    # Per-character rewrites applied before tokenization ("" deletes).
    replace_characters: dict[str, str] = {}
    model_recommended_frames_after_eos: int | None = None
    # Sampling temperature used when the caller does not pass one.
    default_temperature: float = 0.3
    # EOS threshold used when the caller does not pass one (generation stops once the EOS
    # logit exceeds it).
    default_eos_threshold: float = -2.0
    languages: list[str] = ["fa"]
    # Directory holding precomputed voice prompts (*.safetensors with an `audio_prompt`
    # tensor, or exported module states). Relative to the repository root.
    voices_dir: str | None = None
    # Voice conditioning over *normalized* latents instead of raw ones. Upstream
    # pocket-tts uses raw latents; some third-party small-model checkpoints were
    # exported with normalized ones. See docs/ARCHITECTURE.md.
    voice_cond_normalized: bool = False


def load_config(yaml_path: str | Path) -> Config:
    """Load a YAML config file. Only local paths are supported (self-contained project)."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        # Allow paths relative to the repository root when run from elsewhere.
        from ava.utils.utils import repo_root

        alt = repo_root() / yaml_path
        if alt.exists():
            yaml_path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def config_base_dir(yaml_path: str | Path) -> Path:
    """Base directory for resolving relative asset paths: the repository root.

    Found by walking up to the ancestor holding pyproject.toml; falls back to the
    config file's directory.
    """
    yaml_path = Path(yaml_path).resolve()
    for ancestor in yaml_path.parents:
        if (ancestor / "pyproject.toml").exists():
            return ancestor
    return yaml_path.parent


__all__ = [
    "Config",
    "FlowConfig",
    "FlowLMConfig",
    "FlowLMTransformerConfig",
    "LookupTable",
    "MimiConfig",
    "MimiTransformerConfig",
    "QuantizerConfig",
    "SEANetConfig",
    "config_base_dir",
    "load_config",
    "resolve_path",
]
