# NOTICE — attribution and licensing

The Ava project mixes original code, adapted open-source components, and
pre-trained assets. This file states exactly what comes from where, under which
license.

## 1. Code

| Component | Source | License |
| --- | --- | --- |
| `ava/modules/` (attention, conv, resample, rope, seanet, stateful_module, layer_scale, transformer, mlp, dummy_quantizer, text_conditioner), `ava/models/` (flow_lm, mimi, tts_model, model_state, text_chunking), `ava/data/audio_utils.py`, `ava/utils/config.py`, `ava/utils/weights_loading.py` | Adapted from **Kyutai pocket-tts v4.0.1** (<https://github.com/kyutai-labs/pocket-tts>) — the model architecture and streaming runtime of the Pocket TTS format | MIT (© 2025 Kyutai). Adaptations for Ava: renamed imports, local-only file loading, sharded-safetensors support, repository-shipped voices, removed HF downloads / server / quantization extras. |
| `ava/modules/mlp.py` | Adapted from pocket-tts, which took it from MAR (`https://github.com/LTH14/mar`, `models/diffloss.py`) | MIT |
| `web/vendor/phonon-tts/` | **phonon-tts 0.7.1** npm package (Kyutai's browser/WASM runtime for Pocket-TTS-format models), vendored unmodified | Apache-2.0 / MIT (see `web/vendor/phonon-tts/LICENSE-*` and `THIRD_PARTY_LICENSES.md`) |
| Training pipeline (`training/`), sharded weights I/O (`ava/utils/weights_io.py`), CLI scripts (`scripts/`), Persian text normalizer (`ava/text_norm.py`), web UI (`web/app.js`, `web/index.html`, `web/persian.js`, `web/persian_tables.js`), tests, docs | Original work of this repository | MIT (see `LICENSE`) |

## 2. Pre-trained assets (`weights/`)

| File | Origin | License |
| --- | --- | --- |
| `weights/model.safetensors` (sharded) + `weights/tokenizer.json` | Released 18.2M-parameter Persian checkpoint built on Kyutai's Pocket TTS small-model recipe; the Mimi codec and the 2.6M-parameter decoder inside the checkpoint are Kyutai's. Upstream checkpoint (CC BY-NC 4.0): <https://huggingface.co/mehdi-hf/cilla-mini> | **CC BY-NC 4.0** — non-commercial use only. Commercial use requires permission from the upstream licensor (see the model card). |
| `weights/voices/*.safetensors` | Precomputed voice-conditioning vectors shipped with this repository | Same as the model weights |

**Summary:** all *code* in this repo is MIT/Apache-2.0 and free for any use,
including commercial. The *model weights* are CC BY-NC 4.0: using the model to
speak is fine for personal, research and educational purposes; any commercial
deployment of the model itself needs the upstream licensor's permission. The
CC BY-NC 4.0 license requires keeping this attribution notice when the weights
are redistributed. When you train your own flow LM with the provided pipeline
(on your own data), the trained part is yours — but a bundle that reuses the
frozen Mimi codec inherits the non-commercial restriction.

## 3. Citation

- Upstream Pocket TTS (Kyutai): <https://github.com/kyutai-labs/pocket-tts>
- Mimi codec (Kyutai, Moshi): <https://github.com/kyutai-labs/moshi>
- Upstream Persian checkpoint (CC BY-NC 4.0): <https://huggingface.co/mehdi-hf/cilla-mini>
