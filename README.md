# Ava

**Persian text-to-speech in the Pocket TTS format — a self-contained repository.**

Ava is an 18.2M-parameter Persian (Farsi) TTS model built on [Kyutai's Pocket TTS](https://github.com/kyutai-labs/pocket-tts) small-model recipe. It runs on CPU, phones and browsers, and speaks in the voice of a short audio prompt.

This repository is **fully self-contained**: model weights, tokenizer, voices, web demo, and the complete training pipeline ship together — nothing is downloaded at runtime and nothing depends on the gated Hugging Face repository.

| | |
|---|---|
| Flow language model | 15.6M params — 8 layers, d_model 512, one-step "drifting" sampler head |
| Audio decoder (Mimi) | 2.6M params — 12-layer decoder transformer @ 12.5 Hz |
| Speaker encoder (Mimi) | 9.7M params — encodes the voice prompt |
| Sample rate / frame rate | 24 kHz mono / 12.5 Hz |
| Total checkpoint | 27.9M params (~107 MB float32, shipped as 3 shards < 100 MB) |
| Speed (2-core CPU) | ≈ 4× real time (measured: 4.3 s of audio in 1.0 s) |
| Weights license | CC BY-NC 4.0 (non-commercial); code MIT — see [NOTICE.md](NOTICE.md) |

## Quick start

```bash
pip install -e .            # torch, numpy, scipy, safetensors, sentencepiece, tokenizers, pydantic, pyyaml

# Speak with one of the shipped voices (nava / shirin / arash)
python scripts/speak_fa.py --voice shirin --text "سلام، حال شما چطور است؟" --out hello.wav

# Or clone any voice from a 3-10 s WAV
python scripts/speak_fa.py --voice my_voice.wav --text "این یک آزمایش است." --out mine.wav
```

Python API (mirrors `pocket_tts`):

```python
from ava import TTSModel

model = TTSModel.load_model("config/fa.yaml")
voice = model.get_state_for_audio_prompt("weights/voices/nava.safetensors")
audio = model.generate_audio(voice, "سلام دنیا. این مدل فارسی روی پردازنده اجرا می‌شود.")
# audio: float32 tensor [samples] at model.sample_rate (24000)
```

Web demo (runs the model in the browser via the WASM runtime):

```bash
python scripts/serve.py          # http://localhost:8000
```

## What is inside

```
ava/                  Python package (inference API, vendored & adapted from pocket-tts)
training/               Full training pipeline (data prep, tokenizer, flow LM, codec, export, eval)
config/fa.yaml          Model architecture + runtime configuration
weights/                model shards + index, tokenizer, voice files, runtime config
web/                    Browser demo (phonon-tts WASM runtime, RTL Persian UI)
scripts/                speak_fa, export_voice, serve, prepare_web, split/merge weights, smoke data
tests/                  12 tests: shapes, strict load, tokenizer, chunking, training, export
docs/                   ARCHITECTURE, TRAINING, DATA, PUBLISH
samples/                Generated sample WAVs
```

## Training your own model

The training pipeline follows Kyutai's small-model recipe: the Mimi codec stays **frozen**, and the flow LM (15.6M params) is trained on continuous 12.5 Hz codec latents of real speech with a one-step "drifting" sampler head plus an EOS head. Everything was verified end-to-end in this sandbox:

```bash
# 1) Prepare data: a CSV manifest (audio_path, transcript) -> frozen-codec latent cache
python -m training.data.prepare_data --manifest my_data.csv --out data_cache/ava-fa

# 2) Train the flow LM (real config; from scratch or fine-tune with model.init_from)
python -m training.train_flow_lm --config training/configs/ava_fa_flow_lm.yaml

# 3) Export any training step to the Pocket TTS bundle format
python -m training.export --checkpoint runs/ava-fa/checkpoint-latest.pt

# 4) Evaluate + generate sample WAVs
python -m training.evaluate --checkpoint runs/ava-fa/checkpoint-latest.pt
```

A CPU smoke run (`make smoke-train`) bootstraps 16 clips with the released model, trains a miniature flow LM on them and shows the loss falling (measured: latent MSE 1.33 → 0.52, prediction/target correlation 0.00 → 0.70 in 150 steps / 2 seconds). See [docs/TRAINING.md](docs/TRAINING.md) for the objective, schedules and the honest limits of this sandbox verification, and [docs/DATA.md](docs/DATA.md) for dataset guidance (Common Voice fa, your own recordings).

## Repository layout notes

- `weights/model.safetensors` is sharded (`model-0000X-of-00003.safetensors` + index) because GitHub rejects files > 100 MB. `ava` loads the index transparently; `scripts/split_weights.py merge` produces the single file the WASM demo fetches (built automatically by `scripts/serve.py`).
- The three voice files are precomputed voice conditionings — the exact format the browser runtime consumes — so generation never re-encodes audio for them.
- `ava/text_norm.py` normalizes Persian text before synthesis (Arabic→Persian letters, numbers→words, percent, punctuation, ZWNJ handling) — the same behavior as the demo's `persian.js`.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the model, tensor-by-tensor, and how this repo was verified against the checkpoint
- [docs/TRAINING.md](docs/TRAINING.md) — objective, data pipeline, schedules, export, smoke-run evidence
- [docs/DATA.md](docs/DATA.md) — datasets, manifest format, Persian text normalization for training
- [docs/PUBLISH.md](docs/PUBLISH.md) — how to push this to GitHub (shards, LFS notes, Pages limits)
- [NOTICE.md](NOTICE.md) — file-by-file provenance and licensing (MIT code, CC BY-NC weights)

## Credits

- [Kyutai pocket-tts](https://github.com/kyutai-labs/pocket-tts) (MIT) — the model architecture, streaming runtime and format; the modules in `ava/` are adapted from it.
- [Kyutai Mimi codec](https://github.com/kyutai-labs/moshi) — the codec inside the checkpoint.
- [phonon-tts](https://www.npmjs.com/package/phonon-tts) (Apache-2.0/MIT) — the browser WASM runtime vendored under `web/vendor/`.
- Released Persian model weights (CC BY-NC 4.0), included in `weights/` — upstream checkpoint: [`mehdi-hf/cilla-mini`](https://huggingface.co/mehdi-hf/cilla-mini).
