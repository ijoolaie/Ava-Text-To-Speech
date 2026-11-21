# Training

The training pipeline reproduces Kyutai's Pocket TTS small-model recipe for a
Persian model, is fully self-contained (the frozen codec ships in
`weights/`), and was verified end-to-end on CPU in this repository.

## The recipe in one paragraph

The Mimi codec stays **frozen**. The flow LM (15.6M params) is trained
autoregressively on continuous 12.5 Hz codec latents of real speech: teacher
forcing over a sentence's latents, position 0 = BOS, and for every position the
one-step "drifting" head maps Gaussian noise x₀ ~ N(0, σ²) directly to the next
(normalized) latent — L_latent = MSE(head(z_t, x₀), x_{t+1}) — while a binary
EOS head learns to fire on the final position (L_eos = BCE). Each clip also
conditions on its own voice: the same projection inference uses
(`speaker_proj` over codec latents, raw or normalized per
`voice_cond_normalized`) is prepended to the text embeddings, which is how voice
cloning is learned. Latent statistics (`emb_mean`/`emb_std`) are EMA-tracked
during training and exported in the checkpoint, exactly like the released
model's buffers.

## Pipeline

```
manifest.csv ──prepare_data──> latent cache (npz per clip + stats.json)
                                      │
                              train_flow_lm  ← training/configs/*.yaml
                                      │  checkpoints .pt {model, ema, step, stats}
                                      ▼
                                   export  → weights/model.safetensors (bundle + shards)
                                      │
                                   evaluate → holdout cosine + sample WAVs
```

### 1) Data preparation

```bash
python -m training.data.prepare_data \
    --manifest data/my_fa.csv \
    --arch config/fa.yaml \
    --out data_cache/ava-fa
```

Manifest: CSV/TSV with two columns — `(audio_path, transcript)`. Each clip is
loaded, resampled to 24 kHz mono, encoded with the **frozen Mimi encoder** from
the config's bundle, and stored as raw latents `[T, 32]` with the (normalized)
transcript. Dataset-level latent mean/std are written to `stats.json` and seed
the model's EMA stats. Clips are skipped with reasons in `skipped.json`
(missing file, too short/long, empty text).

### 2) Tokenizer (only for a from-scratch language)

```bash
python -m training.train_tokenizer --corpus transcripts.txt --vocab-size 4005 --out-dir weights
```

Trains a SentencePiece unigram model (4005 = `n_bins`; the LUT embedding has
+1 padding row) and converts it to `tokenizer.json` with an id-match check. The
shipped Persian model already has both files — skip this unless you change
languages.

### 3) Flow LM training

```bash
python -m training.train_flow_lm --config training/configs/ava_fa_flow_lm.yaml
# quick overrides: --override optim.total_steps=5000 train.batch_size=8
```

Real config (`ava_fa_flow_lm.yaml`): AdamW lr 5e-4 (β 0.9/0.95, wd 0.01 with
no-decay groups for norms/embeddings), 500-step warmup, cosine decay, grad clip
1.0, batch 16, EMA 0.999 after step 100, stats EMA 0.999, fp32 (bf16 autocast
on CUDA). Checkpoints save `{model: flow_lm.*, ema: flow_lm.*, step, stats,
optimizer}` — the same layout `pocket-tts`'s
`get_training_checkpoint_state_dicts` reads, with the EMA overlay rule.

**Fine-tuning the released model**: set `model.init_from: weights/model.safetensors`.
All flow-LM weights, the speaker projection and the stats buffers load from the
bundle; training then adapts the model to your data with far less of it. This is
the recommended path for small datasets — quality carries over and the loss
drops within minutes.

### 4) Codec fine-tuning (optional, off-recipe)

`training/train_codec.py` fine-tunes the whole Mimi autoencoder with waveform +
multi-resolution STFT reconstruction loss. The released recipe does **not**
train the codec (Kyutai's codec, trained with adversarial losses we do not
include, is kept frozen); use this only if you know why.

### 5) Export

```bash
python -m training.export --checkpoint runs/ava-fa/checkpoint-latest.pt \
    --arch config/fa.yaml --out weights/model.safetensors
```

Assembles `flow_lm.*` (EMA by default) + `mimi.*` (frozen, from the base
bundle) into the Pocket TTS bundle, writes it sharded (< 100 MB per file) and
re-loads it `strict=True` to verify. The exported bundle works with
`TTSModel.load_model`, `scripts/speak_fa.py`, and the web demo. Note the frozen
Mimi part stays under the CC BY-NC license — see NOTICE.md.

### 6) Evaluation

```bash
python -m training.evaluate --checkpoint runs/ava-fa/checkpoint-latest.pt
```

Reports teacher-forced cosine similarity between the drifting head's prediction
and the true next latent on a held-out split, EOS hit/false-alarm rates, and
generates sample WAVs with a chosen voice into `runs/<run>/eval/`.

## What was verified in this sandbox (smoke run, CPU, 2 cores)

| Check | Result |
|---|---|
| Latent cache from real audio (16 clips bootstrapped with the released model) | 16/16 encoded, stats sane |
| Tiny flow LM (`smoke_cpu.yaml`) training loss | 1.33 → 0.52 in 150 steps (~2 s) |
| Prediction/target correlation | 0.00 → 0.70 |
| EOS loss | 0.57 → 0.0016 (fires on final frames) |
| Held-out cosine (eval during run) | 0.64 |
| Export → strict re-load round trip | ✓ |
| Full pytest suite | 12/12 (incl. real-latent training + export tests) |

**Honest limits**: the smoke run proves the *pipeline* (data → loss decrease →
export → reload). It does not produce speech quality — 16 bootstrapped clips
(≈ 45 s of audio) are nowhere near what a real model needs. For a real Persian
model, train on several hundred hours of transcribed speech (docs/DATA.md) on a
GPU; the config's `total_steps: 200000` is a starting point, not a ceiling.
